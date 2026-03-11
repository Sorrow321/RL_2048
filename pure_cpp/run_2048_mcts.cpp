/****************************************************
 * run_2048_mcts.cpp
 *
 * MCTS for 2048 with:
 *   - Bitboard representation (uint64_t, 4-bit nibbles)
 *   - Lookup-table slide/merge
 *   - Heuristic leaf evaluation (no random rollouts)
 *   - Sampled chance nodes (not fully enumerated)
 *   - Sequential iteration (bitboard + O(1) eval = high throughput)
 *
 * Compile: g++ -std=c++17 -O3 run_2048_mcts.cpp -o run_2048_mcts
 ****************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>
#include <cassert>
#include <array>
#include <unordered_map>
#include <cstdint>
#include <chrono>

////////////////////////////////////////////////////////////////////////////////
// 1) Bitboard 2048 Engine
//
// Board = uint64_t, 16 nibbles. Each nibble stores log2(tile_value), with 0
// meaning empty. So nibble=1 means tile 2, nibble=11 means tile 2048, etc.
// Layout: nibble 0 = board[0][0], nibble 1 = board[0][1], ..., nibble 15 = board[3][3]
// i.e. row-major: nibble index = row*4 + col.
//
// A "row" is a 16-bit value (4 nibbles). We precompute tables for sliding
// and merging a single row to the left, then derive all 4 directions.
////////////////////////////////////////////////////////////////////////////////

using Board = uint64_t;
using Row = uint16_t;

static Row   slide_left_table[65536];
static Row   slide_right_table[65536];
static int   merge_score_table[65536];
static double heuristic_table[65536];

static inline int nibble(Board b, int pos) {
    return (b >> (pos * 4)) & 0xF;
}

static inline Board set_nibble(Board b, int pos, int val) {
    Board mask = ~(0xFULL << (pos * 4));
    return (b & mask) | ((Board)val << (pos * 4));
}

static inline Row get_row(Board b, int row) {
    return (Row)(b >> (row * 16));
}

static inline Board set_row(Board b, int row, Row r) {
    Board mask = ~(0xFFFFULL << (row * 16));
    return (b & mask) | ((Board)r << (row * 16));
}

static Row reverse_row(Row r) {
    return ((r & 0xF) << 12) | (((r >> 4) & 0xF) << 8) |
           (((r >> 8) & 0xF) << 4) | ((r >> 12) & 0xF);
}

static void init_tables() {
    for (int r = 0; r < 65536; r++) {
        Row row = (Row)r;
        int c[4] = {row & 0xF, (row >> 4) & 0xF, (row >> 8) & 0xF, (row >> 12) & 0xF};

        // --- Slide left + merge ---
        // Compact left
        int a[4] = {0, 0, 0, 0};
        int pos = 0;
        for (int i = 0; i < 4; i++) {
            if (c[i] != 0) a[pos++] = c[i];
        }
        // Merge adjacent equal pairs
        int score = 0;
        int b[4] = {0, 0, 0, 0};
        int bpos = 0;
        for (int i = 0; i < 4; ) {
            if (i < 3 && a[i] != 0 && a[i] == a[i + 1]) {
                b[bpos++] = a[i] + 1;
                score += (1 << (a[i] + 1));
                i += 2;
            } else {
                b[bpos++] = a[i];
                i++;
            }
        }

        Row result = (Row)(b[0] | (b[1] << 4) | (b[2] << 8) | (b[3] << 12));
        slide_left_table[r] = result;
        slide_right_table[reverse_row(row)] = reverse_row(result);
        merge_score_table[r] = score;

        // --- Row heuristic contribution (on ORIGINAL row values c[], not post-slide b[]) ---
        double empty_count = 0;
        double mono_lr = 0, mono_rl = 0;
        double merge_potential = 0;
        double sum_val = 0;
        for (int i = 0; i < 4; i++) {
            if (c[i] == 0) empty_count += 1.0;
            if (c[i] > 0) sum_val += (1 << c[i]);
        }
        for (int i = 0; i < 3; i++) {
            if (c[i] > 0 && c[i + 1] > 0) {
                if (c[i] > c[i + 1])      mono_lr += (1 << c[i]) - (1 << c[i + 1]);
                else if (c[i] < c[i + 1]) mono_rl += (1 << c[i + 1]) - (1 << c[i]);
            }
            if (c[i] != 0 && c[i] == c[i + 1])
                merge_potential += (1 << (c[i] + 1));
        }
        double monotonicity = -std::min(mono_lr, mono_rl);

        heuristic_table[r] = 200.0 * empty_count
                           + 1.0 * monotonicity
                           + 1.0 * merge_potential;
    }
}

static Board transpose(Board b) {
    // Transpose the 4x4 grid so rows become columns.
    // nibble(row, col) -> nibble(col, row)
    Board t = 0;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            t = set_nibble(t, c * 4 + r, nibble(b, r * 4 + c));
    return t;
}

struct MoveResult {
    Board board;
    int score;
    bool changed;
};

static MoveResult move_left(Board b) {
    Board result = 0;
    int score = 0;
    bool changed = false;
    for (int r = 0; r < 4; r++) {
        Row row = get_row(b, r);
        Row newrow = slide_left_table[row];
        score += merge_score_table[row];
        if (newrow != row) changed = true;
        result = set_row(result, r, newrow);
    }
    return {result, score, changed};
}

static MoveResult move_right(Board b) {
    Board result = 0;
    int score = 0;
    bool changed = false;
    for (int r = 0; r < 4; r++) {
        Row row = get_row(b, r);
        Row rev = reverse_row(row);
        Row newrow = slide_right_table[row];
        score += merge_score_table[rev];
        if (newrow != row) changed = true;
        result = set_row(result, r, newrow);
    }
    return {result, score, changed};
}

static MoveResult move_up(Board b) {
    Board t = transpose(b);
    auto [res, sc, ch] = move_left(t);
    return {transpose(res), sc, ch};
}

static MoveResult move_down(Board b) {
    Board t = transpose(b);
    auto [res, sc, ch] = move_right(t);
    return {transpose(res), sc, ch};
}

// action: 0=down, 1=up, 2=right, 3=left
static MoveResult do_move(Board b, int action) {
    switch (action) {
        case 0: return move_down(b);
        case 1: return move_up(b);
        case 2: return move_right(b);
        case 3: return move_left(b);
        default: return {b, 0, false};
    }
}

static int count_empty(Board b) {
    int cnt = 0;
    for (int i = 0; i < 16; i++)
        if (nibble(b, i) == 0) cnt++;
    return cnt;
}

static int max_tile_log(Board b) {
    int mx = 0;
    for (int i = 0; i < 16; i++)
        mx = std::max(mx, nibble(b, i));
    return mx;
}

static bool can_move(Board b) {
    for (int a = 0; a < 4; a++) {
        if (do_move(b, a).changed) return true;
    }
    return false;
}

static Board spawn_tile(Board b, std::mt19937 &rng) {
    int empty = count_empty(b);
    if (empty == 0) return b;
    std::uniform_int_distribution<int> pos_dist(0, empty - 1);
    int target = pos_dist(rng);
    int cur = 0;
    for (int i = 0; i < 16; i++) {
        if (nibble(b, i) == 0) {
            if (cur == target) {
                std::uniform_real_distribution<double> val_dist(0.0, 1.0);
                int val = (val_dist(rng) < 0.1) ? 2 : 1; // 2 = log2(4), 1 = log2(2)
                return set_nibble(b, i, val);
            }
            cur++;
        }
    }
    return b;
}

static std::vector<int> get_possible_actions(Board b) {
    std::vector<int> acts;
    for (int a = 0; a < 4; a++) {
        if (do_move(b, a).changed) acts.push_back(a);
    }
    return acts;
}

////////////////////////////////////////////////////////////////////////////////
// 2) Heuristic Evaluation
////////////////////////////////////////////////////////////////////////////////

// 4 snake patterns (one per corner), each traversing the board in a
// monotonically decreasing path. We evaluate all 4 and take the max.
static const double snake_patterns[4][16] = {
    // Top-left corner, snaking right-down
    { 15, 14, 13, 12,
       8,  9, 10, 11,
       7,  6,  5,  4,
       0,  1,  2,  3 },
    // Top-right corner, snaking left-down
    { 12, 13, 14, 15,
      11, 10,  9,  8,
       4,  5,  6,  7,
       3,  2,  1,  0 },
    // Bottom-left corner, snaking right-up
    {  0,  1,  2,  3,
       7,  6,  5,  4,
       8,  9, 10, 11,
      15, 14, 13, 12 },
    // Bottom-right corner, snaking left-up
    {  3,  2,  1,  0,
       4,  5,  6,  7,
      11, 10,  9,  8,
      12, 13, 14, 15 },
};

static constexpr double HEURISTIC_SCALE = 200000.0;

static double evaluate_board(Board b) {
    if (!can_move(b)) return -1.0;

    double score = 0.0;

    for (int r = 0; r < 4; r++) {
        Row row = get_row(b, r);
        score += heuristic_table[row];
    }
    Board t = transpose(b);
    for (int r = 0; r < 4; r++) {
        Row row = get_row(t, r);
        score += heuristic_table[row];
    }

    double best_snake = -1e18;
    for (int p = 0; p < 4; p++) {
        double s = 0.0;
        for (int i = 0; i < 16; i++) {
            int v = nibble(b, i);
            if (v > 0) s += snake_patterns[p][i] * (1 << v);
        }
        best_snake = std::max(best_snake, s);
    }
    score += best_snake * 0.25;

    return score / HEURISTIC_SCALE;
}

static const char* action_names[4] = {"down", "up", "right", "left"};

////////////////////////////////////////////////////////////////////////////////
// 3) Data Structures for MCTS (with sampled chance nodes)
////////////////////////////////////////////////////////////////////////////////

struct ActionNode;

struct StateNode {
    Board state;
    int n_visits = 0;
    bool terminal = false;
    double terminal_value = 0.0;
    int depth = 0;

    std::vector<int> actions;
    std::vector<std::shared_ptr<ActionNode>> children_actions;

    std::weak_ptr<ActionNode> parent_action;

    StateNode(Board s) : state(s) {}
};

struct ActionNode {
    int action_id = -1;
    int n_visits = 0;
    double accumulated_value = 0.0;
    int merge_score = 0;
    Board after_move;

    // Sampled chance children: map from board -> StateNode
    std::unordered_map<Board, std::shared_ptr<StateNode>> children_map;
    // For backprop path, we also keep a vector for iteration
    std::vector<std::shared_ptr<StateNode>> children_vec;

    std::weak_ptr<StateNode> parent_state;
};

////////////////////////////////////////////////////////////////////////////////
// 4) MCTS with Heuristic Eval + Sampled Chance Nodes
//
// With O(1) heuristic evaluation, each MCTS iteration is very fast.
// We use a single mutex and run iterations sequentially to avoid contention.
// Multiple games can be parallelized instead, or we can use batch iterations
// with coarse-grained parallelism via independent trees.
//
// For a single game, sequential MCTS with bitboard + heuristic can easily
// do 50k+ iterations per move in under a second.
////////////////////////////////////////////////////////////////////////////////

class MCTS {
private:
    double explore_coef;
    std::shared_ptr<StateNode> root;
    std::mt19937 rng;

public:
    MCTS(double expc = 1.41)
        : explore_coef(expc), rng(std::random_device{}())
    {}

    void init_root(Board st) {
        root = std::make_shared<StateNode>(st);
        root->depth = 0;
        root->terminal = !can_move(st);
        root->terminal_value = root->terminal ? -1.0 : 0.0;
    }

    bool has_root() const { return root != nullptr; }

    void expand_node(std::shared_ptr<StateNode> node) {
        if (node->terminal) return;
        if (!node->children_actions.empty()) return;

        auto acts = get_possible_actions(node->state);
        if (acts.empty()) {
            node->terminal = true;
            node->terminal_value = -1.0;
            return;
        }
        node->actions = acts;
        for (int a : acts) {
            auto anode = std::make_shared<ActionNode>();
            anode->action_id = a;
            anode->parent_state = node;
            auto [board, sc, ch] = do_move(node->state, a);
            anode->after_move = board;
            anode->merge_score = sc;
            node->children_actions.push_back(anode);
        }
    }

    int select_action_idx(const StateNode *node) const {
        double best_val = -1e18;
        int best_idx = 0;
        double log_parent = (node->n_visits > 0) ? std::log((double)node->n_visits) : 0.0;
        for (int i = 0; i < (int)node->children_actions.size(); i++) {
            auto &actnode = node->children_actions[i];
            if (actnode->n_visits == 0) return i;
            double mean_val = actnode->accumulated_value / (double)actnode->n_visits;
            double bonus = explore_coef * std::sqrt(log_parent / (double)actnode->n_visits);
            double score = mean_val + bonus;
            if (score > best_val) {
                best_val = score;
                best_idx = i;
            }
        }
        return best_idx;
    }

    // Progressive widening: only create a new chance child when
    // visits^pw_alpha > current number of children. Otherwise, revisit
    // an existing child uniformly at random.
    static constexpr double pw_alpha = 0.5;

    std::shared_ptr<StateNode> sample_or_create_child(
        std::shared_ptr<ActionNode> anode, int parent_depth)
    {
        int n = anode->n_visits + 1;
        int k = (int)anode->children_vec.size();
        bool should_widen = (k == 0) || (std::pow((double)n, pw_alpha) > (double)k);

        if (should_widen) {
            Board spawned = spawn_tile(anode->after_move, rng);
            auto it = anode->children_map.find(spawned);
            if (it != anode->children_map.end()) return it->second;

            auto sn = std::make_shared<StateNode>(spawned);
            sn->depth = parent_depth + 1;
            sn->terminal = !can_move(spawned);
            sn->terminal_value = sn->terminal ? -1.0 : 0.0;
            sn->parent_action = anode;
            anode->children_map[spawned] = sn;
            anode->children_vec.push_back(sn);
            return sn;
        }

        // Revisit an existing child
        std::uniform_int_distribution<int> dist(0, k - 1);
        return anode->children_vec[dist(rng)];
    }

    void do_one_iteration() {
        auto node = root;
        while (!node->terminal && !node->children_actions.empty()) {
            int idx = select_action_idx(node.get());
            auto &action_node = node->children_actions[idx];
            node = sample_or_create_child(action_node, node->depth);
        }

        if (!node->terminal && node->children_actions.empty()) {
            expand_node(node);
        }

        double val;
        if (node->terminal) {
            val = -1.0;
        } else {
            val = evaluate_board(node->state);
        }

        auto cur = node;
        while (true) {
            cur->n_visits += 1;
            auto pa = cur->parent_action.lock();
            if (!pa) break;
            pa->n_visits += 1;
            pa->accumulated_value += val;
            auto st_parent = pa->parent_state.lock();
            if (!st_parent) break;
            cur = st_parent;
        }
    }

    void run_mcts(int n_iterations) {
        if (!root) return;
        for (int i = 0; i < n_iterations; i++) {
            do_one_iteration();
        }
    }

    int get_best_action() const {
        if (!root || root->children_actions.empty()) return -1;
        int best_visits = -1;
        int best_idx = -1;
        for (int i = 0; i < (int)root->children_actions.size(); i++) {
            auto &an = root->children_actions[i];
            if (an->n_visits > best_visits) {
                best_visits = an->n_visits;
                best_idx = i;
            }
        }
        if (best_idx < 0) return -1;
        return root->actions[best_idx];
    }

    void print_root_stats() const {
        if (!root) return;
        for (int i = 0; i < (int)root->children_actions.size(); i++) {
            auto &an = root->children_actions[i];
            double q = (an->n_visits > 0) ? an->accumulated_value / an->n_visits : 0;
            std::cout << "  " << action_names[root->actions[i]]
                      << ": visits=" << an->n_visits
                      << " Q=" << q << "\n";
        }
    }

    void select_branch(int action, Board new_state) {
        if (!root) return;

        int aidx = -1;
        for (int i = 0; i < (int)root->actions.size(); i++) {
            if (root->actions[i] == action) { aidx = i; break; }
        }
        if (aidx < 0) { root.reset(); return; }

        auto anode = root->children_actions[aidx];
        auto it = anode->children_map.find(new_state);
        if (it == anode->children_map.end()) { root.reset(); return; }

        auto new_root = it->second;
        new_root->parent_action.reset();
        root = new_root;
    }
};

////////////////////////////////////////////////////////////////////////////////
// 5) Utility
////////////////////////////////////////////////////////////////////////////////

static void print_board(Board b) {
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            int val = nibble(b, r * 4 + c);
            if (c > 0) std::cout << "\t";
            std::cout << (val ? (1 << val) : 0);
        }
        std::cout << "\n";
    }
}

////////////////////////////////////////////////////////////////////////////////
// 6) Main
////////////////////////////////////////////////////////////////////////////////

int main() {
    init_tables();

    double explore_coef = 1.5;
    int n_iter = 10000;
    int print_every = 50;

    MCTS mcts(explore_coef);
    std::mt19937 game_rng(std::random_device{}());
    Board state = 0;
    state = spawn_tile(state, game_rng);
    state = spawn_tile(state, game_rng);
    mcts.init_root(state);

    int step = 0;
    int total_merge_score = 0;
    auto t_start = std::chrono::steady_clock::now();

    while (true) {
        if (step % print_every == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_start).count();
            std::cout << "Step " << step
                      << "  Score=" << total_merge_score
                      << "  MaxTile=" << (1 << max_tile_log(state))
                      << "  Time=" << elapsed << "s\n";
            print_board(state);
            std::cout << "\n";
        }

        mcts.run_mcts(n_iter);

        int best_a = mcts.get_best_action();
        if (best_a < 0) {
            std::cout << "No moves left at step " << step << ".\n";
            break;
        }

        auto [new_board, mscore, changed] = do_move(state, best_a);
        total_merge_score += mscore;
        Board spawned = spawn_tile(new_board, game_rng);

        mcts.select_branch(best_a, spawned);
        if (!mcts.has_root()) mcts.init_root(spawned);

        state = spawned;
        step++;

        if (!can_move(state)) {
            std::cout << "Game over at step " << step << "!\n";
            break;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\nFinal board:\n";
    print_board(state);
    std::cout << "Score: " << total_merge_score << "\n";
    std::cout << "Max tile: " << (1 << max_tile_log(state)) << "\n";
    std::cout << "Steps: " << step << "\n";
    std::cout << "Time: " << elapsed << "s\n";

    return 0;
}
