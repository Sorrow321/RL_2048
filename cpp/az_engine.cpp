/****************************************************
 * az_engine.cpp — AlphaZero MCTS engine for 2048 (pybind11 module)
 *
 * Design: C++ owns the game and the trees; PyTorch stays in Python.
 * The bridge is batched: a Runner steps hundreds of concurrent games,
 * each pausing when its search reaches a leaf that needs a network
 * evaluation. Python collects ALL pending leaves as one numpy batch,
 * runs a single GPU forward, and feeds priors+values back; the trees
 * resume. Per Python round-trip: one forward for the whole fleet.
 *
 *   runner = az_engine.Runner(total_games, n_parallel, sims, ...)
 *   while (boards := runner.pending()).size:
 *       p, v = net(encode(boards))          # one batch
 *       runner.feed(softmax(p), v)
 *   examples, results = runner.get_examples(), runner.get_results()
 *
 * Search details:
 *   - Decision nodes: PUCT (Q + c_puct * P * sqrt(N) / (1+n)), priors
 *     from the net masked to legal moves and renormalized.
 *   - Chance nodes: spawn sampled from the true 90/10 distribution,
 *     deduplicated, progressive widening (create only while k^2 < n).
 *   - Value: net's scalar (target: log1p(future merge score)/12, so
 *     death = 0 naturally); leaf value only is backed up, Q ~ [0, 1.2].
 *   - Terminal leaves back up 0.0 without touching the net.
 *   - Root priors get Dirichlet noise during self-play (eps=0 disables
 *     for evaluation play). Moves are chosen by max root visits.
 *   - Trees are arena-allocated and reset per move (chance branching
 *     makes subtree reuse retain ~1% of visits — measured, not worth it).
 *
 * The game engine (bitboard, tables, transpose, spawn) is the one from
 * pure_cpp/run_2048_mcts.cpp, verified there against reference
 * implementations on millions of random boards.
 ****************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>
#include <cstdint>
#include <cstring>
#include <mutex>
#if defined(__BMI2__)
#include <immintrin.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace py = pybind11;

////////////////////////////////////////////////////////////////////////////////
// 1) Bitboard engine (same core as pure_cpp/run_2048_mcts.cpp)
////////////////////////////////////////////////////////////////////////////////

using Board = uint64_t;
using Row = uint16_t;

struct RowEntry {
    Row      left;
    Row      right;
    uint32_t score_left;
    uint32_t score_right;
};
static RowEntry row_tbl[65536];

static inline int nibble(Board b, int pos) {
    return (b >> (pos * 4)) & 0xF;
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
    for (int r = 0; r < 65536; r++) {
        Row row = (Row)r;
        int c[4] = {row & 0xF, (row >> 4) & 0xF, (row >> 8) & 0xF, (row >> 12) & 0xF};
        int a[4] = {0, 0, 0, 0};
        int pos = 0;
        for (int i = 0; i < 4; i++)
            if (c[i] != 0) a[pos++] = c[i];
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
    }
}

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

// action: 0=down, 1=up, 2=right, 3=left (same ids as pure_cpp binary)
static MoveResult do_move(Board b, int action) {
    switch (action) {
        case 0: { Board t = transpose(b); MoveResult m = move_right(t);
                  return {transpose(m.board), m.score, m.changed}; }
        case 1: { Board t = transpose(b); MoveResult m = move_left(t);
                  return {transpose(m.board), m.score, m.changed}; }
        case 2: return move_right(b);
        case 3: return move_left(b);
        default: return {b, 0, false};
    }
}

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

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    inline uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    inline uint32_t below(uint32_t n) {
        return (uint32_t)(((next() & 0xFFFFFFFFULL) * (uint64_t)n) >> 32);
    }
};

static inline Board spawn_tile(Board b, Rng &rng) {
    uint64_t t = b;
    t |= t >> 2;
    t |= t >> 1;
    uint64_t empty = ~t & 0x1111111111111111ULL;
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
    uint64_t val = ((r >> 32) < 429496730ULL) ? 2 : 1;
    return b | (bit * val);
}

////////////////////////////////////////////////////////////////////////////////
// 2) AlphaZero tree (arena-allocated, PUCT + sampled chance nodes)
////////////////////////////////////////////////////////////////////////////////

struct AZStateNode;

struct AZChildRef {
    Board state;
    AZStateNode *node;
};

struct AZActionNode {
    Board after_move;
    AZStateNode *parent_state;
    AZChildRef *children;
    float prior;
    float acc_value;
    int32_t n_visits;
    int32_t n_children;
    int32_t cap_children;
    uint8_t action_id;
};

struct AZStateNode {
    Board state;
    AZActionNode *parent_action;
    AZActionNode *actions;
    int32_t n_visits;
    int16_t n_actions; // -1 unexpanded, 0 terminal, >0 expanded
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
        if (used + sz > cap) throw std::runtime_error("az_engine: arena overflow");
        void *p = buf.get() + used;
        used += sz;
        return p;
    }
    inline void reset() { used = 0; }
};

////////////////////////////////////////////////////////////////////////////////
// 3) Batched self-play runner
////////////////////////////////////////////////////////////////////////////////

struct GameResult {
    int64_t score;
    int32_t max_tile;
    int32_t steps;
};

struct MoveExample {
    uint64_t board;
    int32_t visits[4];
    uint32_t reward;
    uint8_t chosen;
    uint8_t done;
};

struct Slot {
    Arena arena;
    Rng rng;
    std::mt19937 noise_rng;
    Board state = 0;
    int64_t score = 0;
    int32_t steps = 0;
    AZStateNode *root = nullptr;
    AZStateNode *pending = nullptr; // leaf awaiting NN priors+value
    int sims_done = 0;
    std::vector<MoveExample> recs;
    bool in_game = false;

    Slot() : rng(0) {}
};

class Runner {
    int64_t total_games;
    int n_parallel;
    int sims;
    float c_puct;
    float dir_alpha, dir_eps;

    std::vector<Slot> slots;
    std::vector<int> pending_slots; // slot ids matching the last pending() batch

    // shared accumulation (guarded by mu_)
    std::mutex mu_;
    int64_t games_started = 0;
    int64_t games_completed_ = 0;
    std::vector<MoveExample> examples;
    std::vector<GameResult> results;

    inline AZStateNode *new_state(Arena &arena, Board b) {
        AZStateNode *sn = (AZStateNode *)arena.alloc(sizeof(AZStateNode));
        sn->state = b;
        sn->parent_action = nullptr;
        sn->actions = nullptr;
        sn->n_visits = 0;
        sn->n_actions = -1;
        return sn;
    }

    // Fills the node's action array from the board (no priors yet).
    // Returns false if the position is terminal.
    bool expand_structure(Slot &s, AZStateNode *node) {
        MoveResult mr[4];
        compute_all_moves(node->state, mr);
        int n_legal = (int)mr[0].changed + mr[1].changed + mr[2].changed + mr[3].changed;
        if (n_legal == 0) {
            node->n_actions = 0;
            return false;
        }
        AZActionNode *arr = (AZActionNode *)s.arena.alloc(n_legal * sizeof(AZActionNode));
        int j = 0;
        for (int a = 0; a < 4; a++) {
            if (!mr[a].changed) continue;
            AZActionNode &an = arr[j++];
            an.after_move = mr[a].board;
            an.parent_state = node;
            an.children = nullptr;
            an.prior = 0.0f;
            an.acc_value = 0.0f;
            an.n_visits = 0;
            an.n_children = 0;
            an.cap_children = 0;
            an.action_id = (uint8_t)a;
        }
        node->actions = arr;
        node->n_actions = (int16_t)n_legal;
        return true;
    }

    inline int select_puct(const AZStateNode *node) const {
        float sqrt_n = std::sqrt((float)node->n_visits);
        float best = -1e30f;
        int best_i = 0;
        for (int i = 0; i < node->n_actions; i++) {
            const AZActionNode &an = node->actions[i];
            float q = an.n_visits ? an.acc_value / (float)an.n_visits : 0.0f;
            float u = c_puct * an.prior * sqrt_n / (1.0f + (float)an.n_visits);
            float score = q + u;
            if (score > best) {
                best = score;
                best_i = i;
            }
        }
        return best_i;
    }

    // Sample-first chance node: spawn from the true 90/10 distribution,
    // descend into the matching child; create it only while widening
    // allows (k^2 < n), else fall back to a uniform existing pick.
    AZStateNode *sample_child(Slot &s, AZActionNode *a) {
        int n = a->n_visits + 1;
        int k = a->n_children;
        Board spawned = spawn_tile(a->after_move, s.rng);
        AZChildRef *ch = a->children;
        for (int i = 0; i < k; i++)
            if (ch[i].state == spawned) return ch[i].node;
        if (k == 0 || (int64_t)k * k < n) {
            if (k == a->cap_children) {
                int newcap = k ? 2 * k : 4;
                AZChildRef *nc = (AZChildRef *)s.arena.alloc(newcap * sizeof(AZChildRef));
                if (k) std::memcpy(nc, ch, k * sizeof(AZChildRef));
                a->children = nc;
                a->cap_children = newcap;
            }
            AZStateNode *sn = new_state(s.arena, spawned);
            sn->parent_action = a;
            a->children[k] = {spawned, sn};
            a->n_children = k + 1;
            return sn;
        }
        return a->children[s.rng.below((uint32_t)k)].node;
    }

    static void backup(AZStateNode *node, float v) {
        node->n_visits++;
        AZActionNode *pa = node->parent_action;
        while (pa) {
            pa->n_visits++;
            pa->acc_value += v;
            AZStateNode *ps = pa->parent_state;
            ps->n_visits++;
            pa = ps->parent_action;
        }
    }

    void start_move_search(Slot &s) {
        s.arena.reset();
        s.root = new_state(s.arena, s.state);
        s.sims_done = 0;
    }

    void start_new_game_locked(Slot &s) {
        // caller holds mu_
        s.state = 0;
        s.state = spawn_tile(s.state, s.rng);
        s.state = spawn_tile(s.state, s.rng);
        s.score = 0;
        s.steps = 0;
        s.recs.clear();
        s.in_game = true;
        start_move_search(s);
    }

    void finish_move(Slot &s) {
        int32_t visits[4] = {0, 0, 0, 0};
        for (int i = 0; i < s.root->n_actions; i++) {
            const AZActionNode &an = s.root->actions[i];
            visits[an.action_id] = an.n_visits;
        }
        int best = 0;
        for (int a = 1; a < 4; a++)
            if (visits[a] > visits[best]) best = a;

        MoveResult m = do_move(s.state, best);
        MoveExample rec;
        rec.board = s.state;
        for (int a = 0; a < 4; a++) rec.visits[a] = visits[a];
        rec.reward = (uint32_t)m.score;
        rec.chosen = (uint8_t)best;
        rec.done = 0;
        s.recs.push_back(rec);

        s.score += m.score;
        s.steps++;
        s.state = spawn_tile(m.board, s.rng);

        if (can_move(s.state)) {
            start_move_search(s);
            return;
        }

        // game over: flush and maybe start the next game
        s.recs.back().done = 1;
        std::lock_guard<std::mutex> lock(mu_);
        examples.insert(examples.end(), s.recs.begin(), s.recs.end());
        results.push_back({s.score, 1 << max_tile_log(s.state), s.steps});
        games_completed_++;
        if (games_started < total_games) {
            games_started++;
            start_new_game_locked(s);
        } else {
            s.in_game = false;
        }
    }

    // Advance one slot until it needs an eval or runs out of games.
    void advance(Slot &s) {
        while (s.in_game) {
            if (s.pending) return;
            if (s.sims_done >= sims) {
                finish_move(s);
                continue;
            }
            AZStateNode *node = s.root;
            while (node->n_actions > 0) {
                int ai = select_puct(node);
                node = sample_child(s, &node->actions[ai]);
            }
            if (node->n_actions == 0 || !expand_structure(s, node)) {
                backup(node, 0.0f); // dead board: value 0, no net call
                s.sims_done++;
                continue;
            }
            s.pending = node;
            return;
        }
    }

    void apply_eval(Slot &s, const float *p4, float v) {
        AZStateNode *node = s.pending;
        int n = node->n_actions;
        float mass = 0.0f;
        for (int i = 0; i < n; i++)
            mass += p4[node->actions[i].action_id];
        if (mass > 1e-8f) {
            for (int i = 0; i < n; i++)
                node->actions[i].prior = p4[node->actions[i].action_id] / mass;
        } else {
            for (int i = 0; i < n; i++)
                node->actions[i].prior = 1.0f / (float)n;
        }
        if (node == s.root && dir_eps > 0.0f) {
            std::gamma_distribution<float> gamma(dir_alpha, 1.0f);
            float d[4], dsum = 0.0f;
            for (int i = 0; i < n; i++) {
                d[i] = gamma(s.noise_rng);
                dsum += d[i];
            }
            if (dsum > 1e-8f)
                for (int i = 0; i < n; i++)
                    node->actions[i].prior =
                        (1.0f - dir_eps) * node->actions[i].prior + dir_eps * d[i] / dsum;
        }
        backup(node, v);
        s.sims_done++;
        s.pending = nullptr;
    }

public:
    Runner(int64_t total_games_, int n_parallel_, int sims_,
           float c_puct_, float dirichlet_alpha, float dirichlet_eps,
           uint64_t seed)
        : total_games(total_games_), n_parallel(n_parallel_), sims(sims_),
          c_puct(c_puct_), dir_alpha(dirichlet_alpha), dir_eps(dirichlet_eps)
    {
        if (total_games < 1) throw std::invalid_argument("total_games must be >= 1");
        if (n_parallel < 1) throw std::invalid_argument("n_parallel must be >= 1");
        if (sims < 2) throw std::invalid_argument("sims must be >= 2");
        n_parallel = (int)std::min<int64_t>(n_parallel, total_games);
        slots.resize(n_parallel);
        Rng seeder(seed ? seed : 0x8badf00d12345678ULL);
        for (int i = 0; i < n_parallel; i++) {
            Slot &s = slots[i];
            // worst case one node + 4 actions + child growth per sim
            s.arena.init((size_t)(sims + 8) * 320 + 4096);
            s.rng = Rng(seeder.next());
            s.noise_rng.seed((uint32_t)seeder.next());
            games_started++;
            start_new_game_locked(s);
        }
    }

    // Advance all games; return boards of leaves needing evaluation.
    // Empty array <=> all requested games are finished.
    py::array_t<uint64_t> pending() {
        {
            py::gil_scoped_release release;
            int n = (int)slots.size();
#if defined(_OPENMP)
            #pragma omp parallel for schedule(dynamic, 1)
#endif
            for (int i = 0; i < n; i++)
                advance(slots[i]);
        }
        pending_slots.clear();
        for (int i = 0; i < (int)slots.size(); i++)
            if (slots[i].pending) pending_slots.push_back(i);

        py::array_t<uint64_t> out((py::ssize_t)pending_slots.size());
        uint64_t *ptr = out.mutable_data();
        for (size_t i = 0; i < pending_slots.size(); i++)
            ptr[i] = slots[pending_slots[i]].pending->state;
        return out;
    }

    // policy: (B,4) float32 softmax probs (full head; engine masks+renorms
    // over legal moves). value: (B,) float32.
    void feed(py::array_t<float, py::array::c_style | py::array::forcecast> policy,
              py::array_t<float, py::array::c_style | py::array::forcecast> value) {
        if (policy.ndim() != 2 || policy.shape(1) != 4 ||
            (size_t)policy.shape(0) != pending_slots.size())
            throw std::invalid_argument("policy must be (n_pending, 4)");
        if (value.ndim() != 1 || (size_t)value.shape(0) != pending_slots.size())
            throw std::invalid_argument("value must be (n_pending,)");
        const float *p = policy.data();
        const float *v = value.data();
        for (size_t i = 0; i < pending_slots.size(); i++)
            apply_eval(slots[pending_slots[i]], p + 4 * i, v[i]);
        pending_slots.clear();
    }

    // Move examples accumulated since the last call (finished games only):
    // dict of board (u64), visits (N,4 i32), reward (u32), chosen (u8),
    // done (u8). Cleared on return.
    py::dict get_examples() {
        std::lock_guard<std::mutex> lock(mu_);
        py::ssize_t n = (py::ssize_t)examples.size();
        py::array_t<uint64_t> board(n);
        py::array_t<int32_t> visits({n, (py::ssize_t)4});
        py::array_t<uint32_t> reward(n);
        py::array_t<uint8_t> chosen(n);
        py::array_t<uint8_t> done(n);
        for (py::ssize_t i = 0; i < n; i++) {
            const MoveExample &e = examples[i];
            board.mutable_data()[i] = e.board;
            std::memcpy(visits.mutable_data() + 4 * i, e.visits, 4 * sizeof(int32_t));
            reward.mutable_data()[i] = e.reward;
            chosen.mutable_data()[i] = e.chosen;
            done.mutable_data()[i] = e.done;
        }
        examples.clear();
        py::dict d;
        d["board"] = board;
        d["visits"] = visits;
        d["reward"] = reward;
        d["chosen"] = chosen;
        d["done"] = done;
        return d;
    }

    // (N,3) int64: score, max_tile, steps per finished game. Cleared on return.
    py::array_t<int64_t> get_results() {
        std::lock_guard<std::mutex> lock(mu_);
        py::ssize_t n = (py::ssize_t)results.size();
        py::array_t<int64_t> out({n, (py::ssize_t)3});
        for (py::ssize_t i = 0; i < n; i++) {
            out.mutable_data()[3 * i + 0] = results[i].score;
            out.mutable_data()[3 * i + 1] = results[i].max_tile;
            out.mutable_data()[3 * i + 2] = results[i].steps;
        }
        results.clear();
        return out;
    }

    int64_t games_completed() {
        std::lock_guard<std::mutex> lock(mu_);
        return games_completed_;
    }
};

////////////////////////////////////////////////////////////////////////////////
// 4) Small game utilities for Python (greedy eval, tests)
////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(az_engine, m) {
    m.doc() = "Batched AlphaZero MCTS engine for 2048 (bitboard, PUCT, "
              "sampled chance nodes). Action ids: 0=down 1=up 2=right 3=left.";
    init_tables();

    py::class_<Runner>(m, "Runner")
        .def(py::init<int64_t, int, int, float, float, float, uint64_t>(),
             py::arg("total_games"), py::arg("n_parallel"), py::arg("sims"),
             py::arg("c_puct") = 1.5f,
             py::arg("dirichlet_alpha") = 0.5f,
             py::arg("dirichlet_eps") = 0.25f,
             py::arg("seed") = 0)
        .def("pending", &Runner::pending)
        .def("feed", &Runner::feed, py::arg("policy"), py::arg("value"))
        .def("get_examples", &Runner::get_examples)
        .def("get_results", &Runner::get_results)
        .def_property_readonly("games_completed", &Runner::games_completed);

    m.def("do_move", [](uint64_t b, int action) {
        MoveResult r = do_move(b, action);
        return py::make_tuple(r.board, r.score, r.changed);
    }, py::arg("board"), py::arg("action"),
       "(board, action) -> (new_board, merge_score, changed)");

    m.def("legal_actions", [](uint64_t b) {
        MoveResult mr[4];
        compute_all_moves(b, mr);
        std::vector<int> out;
        for (int a = 0; a < 4; a++)
            if (mr[a].changed) out.push_back(a);
        return out;
    }, py::arg("board"));

    m.def("can_move", [](uint64_t b) { return can_move(b); }, py::arg("board"));
    m.def("max_tile", [](uint64_t b) { return 1 << max_tile_log(b); }, py::arg("board"));

    m.def("new_game", [](uint64_t seed) {
        Rng rng(seed);
        Board b = spawn_tile(spawn_tile(0, rng), rng);
        return py::make_tuple((uint64_t)b, (uint64_t)rng.s);
    }, py::arg("seed"), "-> (board, rng_state)");

    m.def("spawn", [](uint64_t b, uint64_t rng_state) {
        Rng rng(0);
        rng.s = rng_state;
        Board nb = spawn_tile(b, rng);
        return py::make_tuple((uint64_t)nb, (uint64_t)rng.s);
    }, py::arg("board"), py::arg("rng_state"), "-> (board, rng_state)");
}
