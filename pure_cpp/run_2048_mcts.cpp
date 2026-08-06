/****************************************************
 * run_2048_mcts.cpp
 *
 * MCTS for 2048, hyper-optimized:
 *   - Bitboard representation (uint64_t, 4-bit nibbles)
 *   - Merged per-row lookup table (slide L/R + scores + heuristic
 *     in one 16-byte entry -> one cache line per row query)
 *   - Bit-trick transpose, branch-free tile spawn via BMI2 pdep
 *   - splitmix64 RNG (vs mt19937)
 *   - Arena-allocated flat tree, reset per move (no malloc/free,
 *     no destructor walks; chance branching makes tree reuse
 *     retain only ~1% of visits, so fresh trees cost ~nothing)
 *   - UCB via precomputed sqrt(log n), 1/sqrt(n), 1/n tables
 *     (no log/sqrt/div in the selection loop)
 *   - Progressive widening as an integer compare (n > k*k)
 *   - Root parallelism: persistent worker pool, each worker builds
 *     an independent tree for the same root; per-action visit
 *     counts are summed to pick the move
 *
 * Env overrides: MCTS_ITERS (total iterations/move, default 10000)
 *                MCTS_THREADS (parallel trees, default = physical-ish cores)
 *
 * Compile: g++ -std=c++17 -O3 -march=native -pthread run_2048_mcts.cpp -o run_2048_mcts
 ****************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <atomic>
#if defined(__BMI2__)
#include <immintrin.h>
#endif

////////////////////////////////////////////////////////////////////////////////
// 1) Bitboard 2048 Engine
//
// Board = uint64_t, 16 nibbles. Each nibble stores log2(tile_value), with 0
// meaning empty. Layout: row-major, nibble index = row*4 + col.
////////////////////////////////////////////////////////////////////////////////

using Board = uint64_t;
using Row = uint16_t;

// Everything a row query needs, in one 16-byte entry.
struct RowEntry {
    Row      left;        // row after sliding left
    Row      right;       // row after sliding right
    uint32_t score_left;  // merge score of sliding left
    uint32_t score_right; // merge score of sliding right
    float    heur;        // row heuristic (empties/monotonicity/merge potential)
};
static_assert(sizeof(RowEntry) == 16, "RowEntry should be 16 bytes");

static RowEntry row_tbl[65536];
static double pow2_tile[16];

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

static inline Row reverse_row(Row r) {
    return ((r & 0xF) << 12) | (((r >> 4) & 0xF) << 8) |
           (((r >> 8) & 0xF) << 4) | ((r >> 12) & 0xF);
}

static void init_tables() {
    pow2_tile[0] = 0.0;
    for (int v = 1; v < 16; v++) pow2_tile[v] = (double)(1 << v);

    for (int r = 0; r < 65536; r++) {
        Row row = (Row)r;
        int c[4] = {row & 0xF, (row >> 4) & 0xF, (row >> 8) & 0xF, (row >> 12) & 0xF};

        // --- Slide left + merge ---
        int a[4] = {0, 0, 0, 0};
        int pos = 0;
        for (int i = 0; i < 4; i++) {
            if (c[i] != 0) a[pos++] = c[i];
        }
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
        Row rev = reverse_row(row);
        row_tbl[r].left = result;
        row_tbl[r].score_left = (uint32_t)score;
        row_tbl[rev].right = reverse_row(result);
        row_tbl[rev].score_right = (uint32_t)score;

        // --- Row heuristic (on ORIGINAL row values, not post-slide) ---
        // All components are integers < 2^24, so float holds them exactly.
        double empty_count = 0;
        double mono_lr = 0, mono_rl = 0;
        double merge_potential = 0;
        for (int i = 0; i < 4; i++) {
            if (c[i] == 0) empty_count += 1.0;
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

        row_tbl[r].heur = (float)(200.0 * empty_count
                                + 1.0 * monotonicity
                                + 1.0 * merge_potential);
    }
}

// Bit-trick transpose (~12 ops). Stage 1 transposes nibbles within each 2x2
// block, stage 2 swaps the off-diagonal 2x2 blocks.
static inline Board transpose(Board x) {
    Board a1 = x & 0xF0F00F0FF0F00F0FULL;
    Board a2 = x & 0x0000F0F00000F0F0ULL;
    Board a3 = x & 0x0F0F00000F0F0000ULL;
    Board a  = a1 | (a2 << 12) | (a3 >> 12);
    Board b1 = a & 0xFF00FF0000FF00FFULL;
    Board b2 = a & 0x00FF00FF00000000ULL;
    Board b3 = a & 0x00000000FF00FF00ULL;
    return b1 | (b2 >> 24) | (b3 << 24);
}

struct MoveResult {
    Board board;
    int score;
    bool changed;
};

static inline MoveResult move_left(Board b) {
    Board result = 0;
    int score = 0;
    bool changed = false;
    for (int r = 0; r < 4; r++) {
        Row row = get_row(b, r);
        const RowEntry &e = row_tbl[row];
        if (e.left != row) changed = true;
        score += e.score_left;
        result = set_row(result, r, e.left);
    }
    return {result, score, changed};
}

static inline MoveResult move_right(Board b) {
    Board result = 0;
    int score = 0;
    bool changed = false;
    for (int r = 0; r < 4; r++) {
        Row row = get_row(b, r);
        const RowEntry &e = row_tbl[row];
        if (e.right != row) changed = true;
        score += e.score_right;
        result = set_row(result, r, e.right);
    }
    return {result, score, changed};
}

static MoveResult move_up(Board b) {
    Board t = transpose(b);
    MoveResult m = move_left(t);
    return {transpose(m.board), m.score, m.changed};
}

static MoveResult move_down(Board b) {
    Board t = transpose(b);
    MoveResult m = move_right(t);
    return {transpose(m.board), m.score, m.changed};
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

// All 4 directions with a single transpose. Indexed by action id.
static void compute_all_moves(Board b, MoveResult out[4]) {
    Board t = transpose(b);
    MoveResult mu = move_left(t);
    MoveResult md = move_right(t);
    out[0] = md.changed ? MoveResult{transpose(md.board), md.score, true}
                        : MoveResult{b, 0, false};
    out[1] = mu.changed ? MoveResult{transpose(mu.board), mu.score, true}
                        : MoveResult{b, 0, false};
    out[2] = move_right(b);
    out[3] = move_left(b);
}

static bool can_move(Board b) {
    for (int r = 0; r < 4; r++) {
        Row row = get_row(b, r);
        const RowEntry &e = row_tbl[row];
        if (e.left != row || e.right != row) return true;
    }
    Board t = transpose(b);
    for (int r = 0; r < 4; r++) {
        Row row = get_row(t, r);
        const RowEntry &e = row_tbl[row];
        if (e.left != row || e.right != row) return true;
    }
    return false;
}

static int max_tile_log(Board b) {
    int mx = 0;
    for (int i = 0; i < 16; i++)
        mx = std::max(mx, nibble(b, i));
    return mx;
}

////////////////////////////////////////////////////////////////////////////////
// 2) Fast RNG + tile spawn
////////////////////////////////////////////////////////////////////////////////

// splitmix64: 3 multiplies per draw, passes BigCrush, plenty for MCTS sampling.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    inline uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    // Lemire multiply-shift bounded draw (bias < 2^-32, irrelevant at n <= 30)
    inline uint32_t below(uint32_t n) {
        return (uint32_t)(((next() & 0xFFFFFFFFULL) * (uint64_t)n) >> 32);
    }
};

// Uniform empty cell, 90% tile 2 / 10% tile 4. Empty-cell mask is computed
// branch-free; the idx-th empty cell is selected with pdep.
static inline Board spawn_tile(Board b, Rng &rng) {
    uint64_t t = b;
    t |= t >> 2;
    t |= t >> 1;
    uint64_t empty = ~t & 0x1111111111111111ULL; // bit0 of each empty nibble
    int n = __builtin_popcountll(empty);
    if (n == 0) return b;

    uint64_t r = rng.next();
    uint32_t idx = (uint32_t)(((r & 0xFFFFFFFFULL) * (uint64_t)n) >> 32);
#if defined(__BMI2__)
    uint64_t bit = _pdep_u64(1ULL << idx, empty);
#else
    uint64_t e = empty;
    for (uint32_t i = 0; i < idx; i++) e &= e - 1;
    uint64_t bit = e & (~e + 1);
#endif
    // (r>>32) < 0.1 * 2^32  ->  10% chance of a 4 (nibble value 2)
    uint64_t val = ((r >> 32) < 429496730ULL) ? 2 : 1;
    return b | (bit * val);
}

////////////////////////////////////////////////////////////////////////////////
// 3) Heuristic Evaluation
////////////////////////////////////////////////////////////////////////////////

// 4 snake patterns (one per corner); evaluate all 4 and take the max.
static const float snake_patterns[4][16] = {
    { 15, 14, 13, 12,
       8,  9, 10, 11,
       7,  6,  5,  4,
       0,  1,  2,  3 },
    { 12, 13, 14, 15,
      11, 10,  9,  8,
       4,  5,  6,  7,
       3,  2,  1,  0 },
    {  0,  1,  2,  3,
       7,  6,  5,  4,
       8,  9, 10, 11,
      15, 14, 13, 12 },
    {  3,  2,  1,  0,
       4,  5,  6,  7,
      11, 10,  9,  8,
      12, 13, 14, 15 },
};

static constexpr float HEURISTIC_SCALE = 200000.0f;

// Callers guarantee the board is non-terminal (dead boards back up -1 directly).
static float evaluate_board(Board b) {
    float score = 0.0f;

    for (int r = 0; r < 4; r++)
        score += row_tbl[get_row(b, r)].heur;
    Board t = transpose(b);
    for (int r = 0; r < 4; r++)
        score += row_tbl[get_row(t, r)].heur;

    float vals[16];
    for (int i = 0; i < 16; i++)
        vals[i] = (float)pow2_tile[nibble(b, i)];

    float best_snake = -1e30f;
    for (int p = 0; p < 4; p++) {
        float s = 0.0f;
        for (int i = 0; i < 16; i++)
            s += snake_patterns[p][i] * vals[i];
        best_snake = std::max(best_snake, s);
    }
    score += best_snake * 0.25f;

    return score / HEURISTIC_SCALE;
}

static const char* action_names[4] = {"down", "up", "right", "left"};

////////////////////////////////////////////////////////////////////////////////
// 4) Arena-allocated MCTS tree (one per worker, reset each move)
////////////////////////////////////////////////////////////////////////////////

struct StateNode;

struct ChildRef {
    Board state;
    StateNode *node;
};

struct ActionNode {
    Board after_move;
    StateNode *parent_state;
    ChildRef *children;    // arena array of distinct sampled spawns
    float acc_value;
    int32_t n_visits;
    int32_t n_children;
    int32_t cap_children;
    uint8_t action_id;
};

struct StateNode {
    Board state;
    ActionNode *parent_action;
    ActionNode *actions;   // arena array, n_actions entries
    int32_t n_visits;
    int16_t n_actions;     // -1 = unexpanded, 0 = terminal, >0 = expanded
};

class Arena {
    std::unique_ptr<uint8_t[]> buf;
    size_t cap = 0, used = 0;
public:
    void init(size_t capacity) {
        buf.reset(new uint8_t[capacity]);
        cap = capacity;
        used = 0;
    }
    inline void *alloc(size_t sz) {
        sz = (sz + 15) & ~(size_t)15;
        if (used + sz > cap) {
            std::cerr << "arena overflow\n";
            std::abort(); // sized for worst case (one node per iteration); unreachable
        }
        void *p = buf.get() + used;
        used += sz;
        return p;
    }
    inline void reset() { used = 0; }
};

// UCB lookup tables, indexed by visit count (bounded by iterations per tree):
//   sqrt_log_tbl[n] = sqrt(log(n)),  rsqrt_tbl[n] = 1/sqrt(n),  inv_tbl[n] = 1/n
static std::vector<float> sqrt_log_tbl, rsqrt_tbl, inv_tbl;

static void init_ucb_tables(int max_visits) {
    sqrt_log_tbl.resize(max_visits + 2);
    rsqrt_tbl.resize(max_visits + 2);
    inv_tbl.resize(max_visits + 2);
    sqrt_log_tbl[0] = 0.0f;
    rsqrt_tbl[0] = 0.0f;
    inv_tbl[0] = 0.0f;
    for (int n = 1; n < max_visits + 2; n++) {
        sqrt_log_tbl[n] = (float)std::sqrt(std::log((double)n));
        rsqrt_tbl[n] = (float)(1.0 / std::sqrt((double)n));
        inv_tbl[n] = (float)(1.0 / (double)n);
    }
}

class Tree {
    Arena arena;
    Rng rng;
    float explore_coef;

    inline StateNode *new_state(Board b) {
        StateNode *sn = (StateNode *)arena.alloc(sizeof(StateNode));
        sn->state = b;
        sn->parent_action = nullptr;
        sn->actions = nullptr;
        sn->n_visits = 0;
        sn->n_actions = -1;
        return sn;
    }

    // Computes all 4 moves once: legality, child boards and terminality
    // in a single pass.
    void expand(StateNode *node) {
        MoveResult mr[4];
        compute_all_moves(node->state, mr);

        int n_legal = (int)mr[0].changed + mr[1].changed + mr[2].changed + mr[3].changed;
        if (n_legal == 0) {
            node->n_actions = 0;
            return;
        }

        ActionNode *arr = (ActionNode *)arena.alloc(n_legal * sizeof(ActionNode));
        int j = 0;
        for (int a = 0; a < 4; a++) {
            if (!mr[a].changed) continue;
            ActionNode &an = arr[j++];
            an.after_move = mr[a].board;
            an.parent_state = node;
            an.children = nullptr;
            an.acc_value = 0.0f;
            an.n_visits = 0;
            an.n_children = 0;
            an.cap_children = 0;
            an.action_id = (uint8_t)a;
        }
        node->actions = arr;
        node->n_actions = (int16_t)n_legal;
    }

    inline int select_idx(const StateNode *node) const {
        float sqlog = sqrt_log_tbl[node->n_visits];
        float best = -1e30f;
        int best_i = 0;
        for (int i = 0; i < node->n_actions; i++) {
            const ActionNode &an = node->actions[i];
            if (an.n_visits == 0) return i;
            float score = an.acc_value * inv_tbl[an.n_visits]
                        + explore_coef * sqlog * rsqrt_tbl[an.n_visits];
            if (score > best) {
                best = score;
                best_i = i;
            }
        }
        return best_i;
    }

    // Progressive widening (alpha = 0.5): create a new chance child only when
    // sqrt(visits+1) > n_children, i.e. exactly when visits+1 > k*k (integer
    // compare). Otherwise revisit an existing child uniformly at random.
    StateNode *sample_or_create(ActionNode *a) {
        int n = a->n_visits + 1;
        int k = a->n_children;
        if (k == 0 || (int64_t)k * k < n) {
            Board spawned = spawn_tile(a->after_move, rng);
            ChildRef *ch = a->children;
            for (int i = 0; i < k; i++)
                if (ch[i].state == spawned) return ch[i].node;

            if (k == a->cap_children) {
                int newcap = k ? 2 * k : 4;
                ChildRef *nc = (ChildRef *)arena.alloc(newcap * sizeof(ChildRef));
                if (k) std::memcpy(nc, ch, k * sizeof(ChildRef));
                a->children = nc;
                a->cap_children = newcap;
            }
            StateNode *sn = new_state(spawned);
            sn->parent_action = a;
            a->children[k] = {spawned, sn};
            a->n_children = k + 1;
            return sn;
        }
        return a->children[rng.below((uint32_t)k)].node;
    }

    void iterate(StateNode *root) {
        StateNode *node = root;
        while (node->n_actions > 0) {
            int idx = select_idx(node);
            node = sample_or_create(&node->actions[idx]);
        }

        float val;
        if (node->n_actions < 0) { // fresh leaf: expand, then evaluate
            expand(node);
            val = (node->n_actions == 0) ? -1.0f : evaluate_board(node->state);
        } else {                   // terminal
            val = -1.0f;
        }

        node->n_visits++;
        ActionNode *pa = node->parent_action;
        while (pa) {
            pa->n_visits++;
            pa->acc_value += val;
            StateNode *ps = pa->parent_state;
            ps->n_visits++;
            pa = ps->parent_action;
        }
    }

public:
    Tree(uint64_t seed, float expc, int iters_per_move)
        : rng(seed), explore_coef(expc)
    {
        // Worst case: one StateNode + 4 ActionNodes + child-array growth per
        // iteration, ~320 bytes. Sized so alloc can never fail mid-search.
        arena.init((size_t)iters_per_move * 320 + (1 << 16));
    }

    // Fresh tree for `board`, run `iters` iterations, write per-action visit
    // counts and accumulated values (indexed by action id, zero if illegal).
    void search(Board board, int iters, int32_t out_visits[4], float out_acc[4]) {
        arena.reset();
        StateNode *root = new_state(board);
        for (int i = 0; i < iters; i++)
            iterate(root);

        for (int a = 0; a < 4; a++) {
            out_visits[a] = 0;
            out_acc[a] = 0.0f;
        }
        for (int i = 0; i < root->n_actions; i++) {
            const ActionNode &an = root->actions[i];
            out_visits[an.action_id] = an.n_visits;
            out_acc[an.action_id] = an.acc_value;
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
// 5) Root-parallel search: persistent worker pool
//
// Each worker owns one Tree and searches the same root position
// independently; the main thread searches its own share too, then sums
// per-action visits across trees to pick the move. Workers park on an
// atomic generation counter (spin; moves are back-to-back compute).
////////////////////////////////////////////////////////////////////////////////

struct alignas(64) WorkerResult {
    int32_t visits[4];
    float acc[4];
};

class ParallelMCTS {
    int n_trees;          // including the main thread's tree
    int iters_per_tree;
    std::vector<std::thread> threads;
    std::vector<WorkerResult> results;
    std::unique_ptr<Tree> main_tree;

    std::atomic<uint64_t> go_gen{0};
    std::atomic<int> done_count{0};
    std::atomic<bool> quit{false};
    Board shared_board = 0;

    void worker_main(int tid, uint64_t seed) {
        Tree tree(seed, expc_, iters_per_tree);
        uint64_t seen_gen = 0;
        while (true) {
            while (go_gen.load(std::memory_order_acquire) == seen_gen)
                std::this_thread::yield();
            seen_gen = go_gen.load(std::memory_order_acquire);
            if (quit.load(std::memory_order_acquire)) break;
            tree.search(shared_board, iters_per_tree,
                        results[tid].visits, results[tid].acc);
            done_count.fetch_add(1, std::memory_order_release);
        }
    }

    float expc_;

public:
    ParallelMCTS(int total_iters, int n_trees_, float expc)
        : n_trees(n_trees_), expc_(expc)
    {
        iters_per_tree = (total_iters + n_trees - 1) / n_trees;
        init_ucb_tables(iters_per_tree);

        std::random_device rd;
        auto seed = [&rd]() {
            return ((uint64_t)rd() << 32) ^ rd();
        };
        main_tree.reset(new Tree(seed(), expc_, iters_per_tree));
        results.resize(n_trees);
        for (int t = 1; t < n_trees; t++)
            threads.emplace_back(&ParallelMCTS::worker_main, this, t, seed());
    }

    ~ParallelMCTS() {
        quit.store(true, std::memory_order_release);
        go_gen.fetch_add(1, std::memory_order_release);
        for (auto &t : threads) t.join();
    }

    int trees() const { return n_trees; }
    int iters() const { return iters_per_tree; }

    // Runs the search on all trees; returns best action id or -1.
    int search_best_action(Board board) {
        shared_board = board;
        done_count.store(0, std::memory_order_relaxed);
        go_gen.fetch_add(1, std::memory_order_release);

        main_tree->search(board, iters_per_tree,
                          results[0].visits, results[0].acc);

        while (done_count.load(std::memory_order_acquire) < n_trees - 1)
            std::this_thread::yield();

        int64_t total_visits[4] = {0, 0, 0, 0};
        for (int t = 0; t < n_trees; t++)
            for (int a = 0; a < 4; a++)
                total_visits[a] += results[t].visits[a];

        int best_a = -1;
        int64_t best_v = 0;
        for (int a = 0; a < 4; a++) {
            if (total_visits[a] > best_v) {
                best_v = total_visits[a];
                best_a = a;
            }
        }
        return best_a;
    }

    void print_root_stats() const {
        for (int a = 0; a < 4; a++) {
            int64_t v = 0;
            double acc = 0.0;
            for (int t = 0; t < n_trees; t++) {
                v += results[t].visits[a];
                acc += results[t].acc[a];
            }
            if (v == 0) continue;
            std::cout << "  " << action_names[a]
                      << ": visits=" << v
                      << " Q=" << (acc / (double)v) << "\n";
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
// 6) Utility
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

static int env_int(const char *name, int fallback) {
    const char *v = std::getenv(name);
    return v ? std::atoi(v) : fallback;
}

////////////////////////////////////////////////////////////////////////////////
// 7) Self-play data generation
//
// MCTS_DUMP=<prefix> switches the binary into self-play mode: each thread
// plays complete games with its OWN single tree (no per-move root
// parallelism — one game per thread maximizes positions/s) and appends
// fixed 32-byte records to <prefix>.tNN.bin. Value targets are derived
// later in Python from the per-move rewards, so the dump stays raw.
////////////////////////////////////////////////////////////////////////////////

#pragma pack(push, 1)
struct SelfplayRecord {
    uint64_t board;      // bitboard BEFORE the chosen move
    uint32_t visits[4];  // root visit counts by action id -> policy target
    uint32_t reward;     // merge score of the move actually played
    uint8_t  chosen;     // action id played (argmax visits)
    uint8_t  done;       // 1 if the game ended after this move's spawn
    uint16_t pad;
};
#pragma pack(pop)
static_assert(sizeof(SelfplayRecord) == 32, "record must be 32 bytes");

static std::atomic<int> g_games_done{0};

static void selfplay_thread(int tid, const char *prefix, int n_games,
                            int iters, float expc, uint64_t seed) {
    char path[512];
    std::snprintf(path, sizeof path, "%s.t%02d.bin", prefix, tid);
    FILE *f = std::fopen(path, "wb");
    if (!f) { std::cerr << "cannot open " << path << "\n"; return; }

    Tree tree(seed, expc, iters);
    Rng game_rng(seed ^ 0xD1B54A32D192ED03ULL);
    std::vector<SelfplayRecord> recs;
    recs.reserve(4096);

    for (int g = 0; g < n_games; g++) {
        Board state = 0;
        state = spawn_tile(state, game_rng);
        state = spawn_tile(state, game_rng);
        recs.clear();

        while (can_move(state)) {
            int32_t visits[4];
            float acc[4];
            tree.search(state, iters, visits, acc);

            int best = -1;
            int32_t best_v = 0;
            for (int a = 0; a < 4; a++)
                if (visits[a] > best_v) { best_v = visits[a]; best = a; }
            if (best < 0) break;

            MoveResult m = do_move(state, best);
            SelfplayRecord r;
            r.board = state;
            for (int a = 0; a < 4; a++) r.visits[a] = (uint32_t)visits[a];
            r.reward = (uint32_t)m.score;
            r.chosen = (uint8_t)best;
            r.done = 0;
            r.pad = 0;
            recs.push_back(r);

            state = spawn_tile(m.board, game_rng);
        }

        if (!recs.empty()) {
            recs.back().done = 1;
            std::fwrite(recs.data(), sizeof(SelfplayRecord), recs.size(), f);
            std::fflush(f);
        }
        int total = g_games_done.fetch_add(1) + 1;
        if (total % 25 == 0)
            std::cerr << "games: " << total << "\r" << std::flush;
    }
    std::fclose(f);
}

static int selfplay_main(const char *prefix) {
    int n_games = env_int("MCTS_GAMES", 100);
    int iters = env_int("MCTS_ITERS", 10000); // per single tree here
    int hw = (int)std::thread::hardware_concurrency();
    int n_threads = env_int("MCTS_THREADS", std::max(1, hw / 2));
    n_threads = std::min(n_threads, n_games);
    float expc = 1.5f;

    init_ucb_tables(iters); // shared, read-only after init

    std::cerr << "Self-play: " << n_games << " games, " << iters
              << " iters/move (single tree), " << n_threads
              << " threads -> " << prefix << ".tNN.bin\n";

    auto t0 = std::chrono::steady_clock::now();
    std::random_device rd;
    std::vector<std::thread> threads;
    int per = n_games / n_threads, extra = n_games % n_threads;
    for (int t = 0; t < n_threads; t++) {
        uint64_t seed = ((uint64_t)rd() << 32) ^ rd();
        threads.emplace_back(selfplay_thread, t, prefix,
                             per + (t < extra ? 1 : 0), iters, expc, seed);
    }
    for (auto &th : threads) th.join();
    double el = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cerr << "\ndone in " << el << "s\n";
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// 8) Main
////////////////////////////////////////////////////////////////////////////////

int main() {
    if (const char *prefix = std::getenv("MCTS_DUMP")) {
        init_tables();
        return selfplay_main(prefix);
    }
    init_tables();

    float explore_coef = 1.5f;
    // Default: 3000 iters x 16 trees. Measured ~3k moves/s on a 9950X3D with
    // playing strength matching the old 10000-iteration single-tree search
    // (15-game comparison; 625-iter trees were noticeably weaker, so don't
    // shrink iters/tree below ~2-3k).
    int n_iter = env_int("MCTS_ITERS", 48000);
    int hw = (int)std::thread::hardware_concurrency();
    int n_threads = env_int("MCTS_THREADS", std::max(1, hw / 2)); // physical cores
    int print_every = 50;

    ParallelMCTS mcts(n_iter, n_threads, explore_coef);
    std::cout << "Trees: " << mcts.trees()
              << "  Iters/tree: " << mcts.iters()
              << "  (total " << (int64_t)mcts.trees() * mcts.iters() << "/move)\n\n";

    std::random_device rd;
    Rng game_rng(((uint64_t)rd() << 32) ^ rd());
    Board state = 0;
    state = spawn_tile(state, game_rng);
    state = spawn_tile(state, game_rng);

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

        int best_a = mcts.search_best_action(state);
        if (best_a < 0) {
            std::cout << "No moves left at step " << step << ".\n";
            break;
        }

        MoveResult m = do_move(state, best_a);
        total_merge_score += m.score;
        state = spawn_tile(m.board, game_rng);
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
    std::cout << "Time: " << elapsed << "s"
              << "  (" << (elapsed > 0 ? (double)step / elapsed : 0) << " moves/s)\n";

    return 0;
}
