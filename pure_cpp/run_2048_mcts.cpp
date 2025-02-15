/****************************************************
 * run_2048_mcts_parallel.cpp
 * 
 * Demonstrates a single-file example of MCTS on 2048
 * with partial "tree parallelization" + "rollout
 * parallelization" using OpenMP.
 *
 * Compile: g++ -std=c++17 -O2 -fopenmp run_2048_mcts_parallel.cpp -o run_2048_mcts_parallel
 ****************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <tuple>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <memory>
#include <ctime>
#include <cassert>
#include <mutex>       // For std::mutex
#include <shared_mutex>  // C++17 shared mutex
#include <omp.h>       // OpenMP header

////////////////////////////////////////////////////////////////////////////////
// 1) Basic 2048 Mechanics
////////////////////////////////////////////////////////////////////////////////

class Game
{
private:
    // 4x4 board
    std::vector<std::vector<int>> state;
    long long score;

    std::random_device dev;
    std::mt19937 rng;

public:
    Game()
    : state(4, std::vector<int>(4, 0)),
      score(0),
      rng(dev())
    {
        try_spawn_new_tile();
        try_spawn_new_tile();
    }

    // Return board
    const std::vector<std::vector<int>>& get_state() const
    {
        return state;
    }

    long long get_current_score() const
    {
        return score;
    }

    // Reset board
    std::vector<std::vector<int>> reset()
    {
        score = 0;
        for(int i=0; i<4; i++){
            for(int j=0; j<4; j++){
                state[i][j] = 0;
            }
        }
        try_spawn_new_tile();
        try_spawn_new_tile();
        return state;
    }

    bool try_spawn_new_tile()
    {
        std::vector<std::pair<int,int>> empty_places;
        for(int i=0; i<4; i++){
            for(int j=0; j<4; j++){
                if(state[i][j]==0) {
                    empty_places.push_back({i,j});
                }
            }
        }
        if(empty_places.empty()) {
            return false;
        }
        std::uniform_int_distribution<> dist(0, (int)empty_places.size()-1);
        int rpos = dist(rng);
        auto [rr, cc] = empty_places[rpos];

        std::uniform_real_distribution<double> p_dist(0.0,1.0);
        double p = p_dist(rng);
        int val = (p<0.1)?4:2;
        state[rr][cc] = val;
        return true;
    }

    bool can_make_move(const std::vector<std::vector<int>>& st) const
    {
        for(int i=0; i<4; i++){
            for(int j=0; j<4; j++){
                if(st[i][j] == 0){
                    return true;
                }
                if(j>0 && st[i][j] == st[i][j-1]){
                    return true;
                }
                if(i>0 && st[i][j] == st[i-1][j]){
                    return true;
                }
            }
        }
        return false;
    }

    // Slide + merge logic
    std::pair<std::vector<std::vector<int>>, int>
    slide(const std::vector<std::vector<int>>& st, int action) const
    {
        // action: 0=down,1=up,2=right,3=left
        std::vector<std::vector<int>> res = st;
        int changes = 0;
        if(action==0) {
            // down
            for(int col=0; col<4; col++){
                int idx = 3;
                for(int row=3; row>=0; row--){
                    if(res[row][col]!=0){
                        if(idx!=row){
                            changes++;
                            res[idx][col] = res[row][col];
                            res[row][col] = 0;
                        }
                        idx--;
                    }
                }
            }
        }
        else if(action==1){
            // up
            for(int col=0; col<4; col++){
                int idx = 0;
                for(int row=0; row<4; row++){
                    if(res[row][col]!=0){
                        if(idx!=row){
                            changes++;
                            res[idx][col] = res[row][col];
                            res[row][col] = 0;
                        }
                        idx++;
                    }
                }
            }
        }
        else if(action==2){
            // right
            for(int row=0; row<4; row++){
                int idx = 3;
                for(int col=3; col>=0; col--){
                    if(res[row][col]!=0){
                        if(idx!=col){
                            changes++;
                            res[row][idx] = res[row][col];
                            res[row][col] = 0;
                        }
                        idx--;
                    }
                }
            }
        }
        else if(action==3){
            // left
            for(int row=0; row<4; row++){
                int idx = 0;
                for(int col=0; col<4; col++){
                    if(res[row][col]!=0){
                        if(idx!=col){
                            changes++;
                            res[row][idx] = res[row][col];
                            res[row][col] = 0;
                        }
                        idx++;
                    }
                }
            }
        }
        return {res, changes};
    }

    std::pair<std::vector<std::vector<int>>, long long>
    merge(const std::vector<std::vector<int>>& st, int action) const
    {
        std::vector<std::vector<int>> res = st;
        long long partial_reward = 0;
        if(action==0){
            // down
            for(int col=0; col<4; col++){
                int idx = 3; 
                int last_val = 0;
                for(int row=3; row>=0; row--){
                    if(res[row][col]!=0 && last_val==0){
                        last_val = res[row][col];
                    }
                    else if(res[row][col]!=0){
                        if(last_val == res[row][col]){
                            res[idx][col] = last_val * 2;
                            partial_reward += res[idx][col];
                            res[row][col] = 0;
                            last_val=0;
                            idx--;
                        } else {
                            last_val = res[row][col];
                            idx--;
                        }
                    }
                }
            }
        }
        else if(action==1){
            // up
            for(int col=0; col<4; col++){
                int idx = 0;
                int last_val = 0;
                for(int row=0; row<4; row++){
                    if(res[row][col]!=0 && last_val==0){
                        last_val = res[row][col];
                    }
                    else if(res[row][col]!=0){
                        if(last_val == res[row][col]){
                            res[idx][col] = last_val*2;
                            partial_reward += res[idx][col];
                            res[row][col] = 0;
                            last_val=0;
                            idx++;
                        } else {
                            last_val = res[row][col];
                            idx++;
                        }
                    }
                }
            }
        }
        else if(action==2){
            // right
            for(int row=0; row<4; row++){
                int idx = 3;
                int last_val = 0;
                for(int col=3; col>=0; col--){
                    if(res[row][col]!=0 && last_val==0){
                        last_val=res[row][col];
                    }
                    else if(res[row][col]!=0){
                        if(last_val == res[row][col]){
                            res[row][idx] = last_val*2;
                            partial_reward += res[row][idx];
                            res[row][col] = 0;
                            last_val=0;
                            idx--;
                        } else {
                            last_val = res[row][col];
                            idx--;
                        }
                    }
                }
            }
        }
        else if(action==3){
            // left
            for(int row=0; row<4; row++){
                int idx = 0;
                int last_val = 0;
                for(int col=0; col<4; col++){
                    if(res[row][col]!=0 && last_val==0){
                        last_val = res[row][col];
                    }
                    else if(res[row][col]!=0){
                        if(last_val == res[row][col]){
                            res[row][idx] = last_val*2;
                            partial_reward += res[row][idx];
                            res[row][col] = 0;
                            last_val=0;
                            idx++;
                        } else {
                            last_val = res[row][col];
                            idx++;
                        }
                    }
                }
            }
        }
        return {res, partial_reward};
    }

    // Return all next states from a given state+action, with probabilities
    std::tuple<
        std::vector<std::vector<std::vector<int>>>,
        std::vector<double>,
        std::vector<long long>,
        std::vector<bool>
    > simulate_action(const std::vector<std::vector<int>>& st, int action) const
    {
        auto [s1, c1] = slide(st, action);
        auto [s2, mr] = merge(s1, action);
        auto [s3, c2] = slide(s2, action);

        // find empties
        std::vector<std::pair<int,int>> empties;
        for(int i=0; i<4; i++){
            for(int j=0; j<4; j++){
                if(s3[i][j]==0){
                    empties.push_back({i,j});
                }
            }
        }

        std::vector<std::vector<std::vector<int>>> states;
        std::vector<double> probas;
        std::vector<long long> rewards;
        std::vector<bool> dones;

        if(empties.empty()){
            // single next state
            states.push_back(s3);
            probas.push_back(1.0);
            rewards.push_back(mr);
            bool stuck = !can_make_move(s3);
            dones.push_back(stuck);
            return {states, probas, rewards, dones};
        }

        for(auto &e : empties){
            auto st2 = s3;
            st2[e.first][e.second] = 2;
            states.push_back(st2);
            probas.push_back(0.9);
            rewards.push_back(mr);
            dones.push_back(!can_make_move(st2));

            st2[e.first][e.second] = 4;
            states.push_back(st2);
            probas.push_back(0.1);
            rewards.push_back(mr);
            dones.push_back(!can_make_move(st2));
        }
        double norm = empties.size();
        for(double &p: probas) {
            p /= norm;
        }
        return {states, probas, rewards, dones};
    }

    // Which moves are valid in a given board
    std::vector<int> get_possible_actions(const std::vector<std::vector<int>>& st) const
    {
        std::vector<int> acts;
        for(int a=0; a<4; a++){
            auto [s1, c1] = slide(st, a);
            auto [s2, mr] = merge(s1, a);
            auto [s3, c2] = slide(s2, a);
            if(c1+c2>0 || mr>0){
                acts.push_back(a);
            }
        }
        return acts;
    }
};

////////////////////////////////////////////////////////////////////////////////
// 2) Environment Wrapper: tile >=512 => done=1 => reward=+1
//    else if no moves => done=1 => reward=-1, else 0
////////////////////////////////////////////////////////////////////////////////

class GameEnv {
private:
    Game game;
public:
    GameEnv() { }

    std::vector<std::vector<int>> get_initial_state()
    {
        return game.reset();
    }

    std::vector<int> get_possible_actions(const std::vector<std::vector<int>>& st)
    {
        return game.get_possible_actions(st);
    }

    // main transition
    std::tuple<
        std::vector<std::vector<std::vector<int>>>,
        std::vector<double>,
        std::vector<double>,
        std::vector<bool>
    > make_transition(int action, const std::vector<std::vector<int>>& st)
    {
        auto [states, probas, merge_rewards, dones] = game.simulate_action(st, action);

        std::vector<double> final_rewards(states.size(), 0.0);
        for(size_t i=0; i<states.size(); i++){
            bool got_big = false;
            for(int r=0; r<4; r++){
                for(int c=0; c<4; c++){
                    if(states[i][r][c]>=4096){
                        got_big = true;
                        break;
                    }
                }
                if(got_big) break;
            }
            if(got_big){
                final_rewards[i] = 1.0;
                dones[i] = true;
            }
            else if(dones[i]){
                final_rewards[i] = -1.0;
            }
            else {
                final_rewards[i] = 0.0;
            }
        }
        return {states, probas, final_rewards, dones};
    }
};

////////////////////////////////////////////////////////////////////////////////
// 3) Data structures for MCTS
////////////////////////////////////////////////////////////////////////////////

struct ActionNode; // forward

struct StateNode {
    std::vector<std::vector<int>> state;
    int n_visits = 0;
    bool terminal = false;
    double terminal_reward = 0.0;
    int depth = 0;

    // Actions from this state
    std::vector<int> actions; 
    std::vector<std::shared_ptr<ActionNode>> children_actions;

    // pointer to parent action
    std::weak_ptr<ActionNode> parent_action;

    StateNode(const std::vector<std::vector<int>>& s)
    : state(s)
    {}
};

struct ActionNode {
    int n_visits = 0;
    double accumulated_value = 0.0;
    // from this action => set of next states
    std::vector<std::shared_ptr<StateNode>> children_states;
    std::vector<double> children_probas;

    // pointer back to parent state
    std::weak_ptr<StateNode> parent_state;
};

static bool states_equal(const std::vector<std::vector<int>>& a,
                         const std::vector<std::vector<int>>& b)
{
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            if(a[i][j]!=b[i][j]) return false;
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// 4) Parallel MCTS
////////////////////////////////////////////////////////////////////////////////

class ParallelMCTS {
private:
    GameEnv &env;
    double gamma;               // discount factor
    double explore_coef;        // UCB exploration
    std::shared_ptr<StateNode> root;

    // Use a shared_mutex so that multiple threads can safely read concurrently.
    mutable std::shared_mutex tree_mutex;

    // Global RNG (used only for seeding local RNGs).
    std::mt19937 rng;

public:
    ParallelMCTS(GameEnv &game_env, double expc=1.0, double df=0.99)
    : env(game_env),
      gamma(df),
      explore_coef(expc)
    {
        std::random_device rd;
        rng.seed(rd());
    }

    // Initialize the root node from an external state.
    void init_root(const std::vector<std::vector<int>> &st)
    {
        std::unique_lock<std::shared_mutex> lock(tree_mutex);
        root = std::make_shared<StateNode>(st);
        root->depth = 0;
        root->terminal = false;
        root->terminal_reward = 0.0;
        root->n_visits = 0;
    }

    // Expand a node if not already expanded.
    void expand_node(std::shared_ptr<StateNode> node)
    {
        if(node->terminal) return;
        if(!node->children_actions.empty()) return; // already expanded

        auto acts = env.get_possible_actions(node->state);
        if(acts.empty()){
            node->terminal = true;
            node->terminal_reward = -1.0;
            return;
        }
        node->actions = acts;
        for(auto a : acts){
            auto anode = std::make_shared<ActionNode>();
            anode->parent_state = node;

            // get next states
            auto [states, probas, rewards, dones] = env.make_transition(a, node->state);
            for(size_t i=0; i<states.size(); i++){
                auto sn = std::make_shared<StateNode>(states[i]);
                sn->terminal = dones[i];
                sn->terminal_reward = rewards[i];
                sn->depth = node->depth + 1;
                sn->parent_action = anode;
                anode->children_states.push_back(sn);
                anode->children_probas.push_back(probas[i]);
            }
            node->children_actions.push_back(anode);
        }
    }

    // UCB selection at a node.
    // Must be called while holding a shared (read) lock.
    int select_action_idx_locked(std::shared_ptr<const StateNode> node) const
    {
        double best_val = -1e9;
        int best_idx = 0;
        for (int i = 0; i < (int)node->children_actions.size(); i++){
            auto &actnode = node->children_actions[i];
            if(actnode->n_visits == 0){
                return i; // unvisited → infinite priority
            }
            double mean_val = actnode->accumulated_value / (double)actnode->n_visits;
            double bonus = explore_coef * std::sqrt(std::log(node->n_visits) / (double)actnode->n_visits);
            double score = mean_val + bonus;
            if(score > best_val){
                best_val = score;
                best_idx = i;
            }
        }
        return best_idx;
    }

    // Sample next state from an action node.
    std::shared_ptr<StateNode> sample_next_state(std::shared_ptr<ActionNode> a_node, std::mt19937 &local_rng)
    {
        std::discrete_distribution<int> dist(a_node->children_probas.begin(), a_node->children_probas.end());
        int idx = dist(local_rng);
        return a_node->children_states[idx];
    }

    // Single-thread selection + expansion.
    std::shared_ptr<StateNode> select_and_expand(std::mt19937 &local_rng)
    {
        std::shared_ptr<StateNode> node;
        { // acquire shared lock to safely read the root.
            std::shared_lock<std::shared_mutex> lock(tree_mutex);
            node = root;
        }
        while (true) {
            if(node->terminal)
                break;
            {
                // Acquire shared lock to safely read children_actions.
                std::shared_lock<std::shared_mutex> lock(tree_mutex);
                if(node->children_actions.empty())
                    break;
                // Use the locked version of UCB selection.
                int idx = select_action_idx_locked(node);
                auto action_node = node->children_actions[idx];
                // Sample next state from the chosen action.
                node = sample_next_state(action_node, local_rng);
            }
        }
        // Now, acquire an exclusive lock to expand the node.
        {
            std::unique_lock<std::shared_mutex> lock(tree_mutex);
            expand_node(node);
        }
        return node;
    }

    // --- ROLLOUT PARALLELIZATION ---
    // Run multiple random simulations (rollouts) from a given leaf and average their values.
    double parallel_rollout(std::shared_ptr<StateNode> leaf, int num_rollouts)
    {
        if(leaf->terminal){
            return leaf->terminal_reward; 
        }

        std::vector<double> results(num_rollouts, 0.0);

        #pragma omp parallel
        {
            std::mt19937 local_rng(std::random_device{}());
            #pragma omp for
            for (int i = 0; i < num_rollouts; i++){
                results[i] = single_rollout(*leaf, local_rng);
            }
        }

        double sum = 0.0;
        for(double v : results)
            sum += v;
        return sum / (double)num_rollouts;
    }

    // A single random rollout from a node until terminal.
    double single_rollout(const StateNode &start_node, std::mt19937 &local_rng)
    {
        if(start_node.terminal){
            return start_node.terminal_reward;
        }

        auto state = start_node.state;
        int depth_ = start_node.depth;
        bool done = start_node.terminal;
        double total_reward = 0.0;

        while (!done) {
            auto actions = env.get_possible_actions(state);
            if (actions.empty()){
                total_reward += std::pow(gamma, depth_) * (-1.0);
                break;
            }
            std::uniform_int_distribution<int> adist(0, (int)actions.size() - 1);
            int chosen_action = actions[adist(local_rng)];

            auto [states, probas, rewards, dones] = env.make_transition(chosen_action, state);
            std::discrete_distribution<int> ddist(probas.begin(), probas.end());
            int idx = ddist(local_rng);

            total_reward += std::pow(gamma, depth_) * rewards[idx];
            state = states[idx];
            done = dones[idx];
            depth_++;
        }
        return total_reward;
    }

    // Backpropagation: update visits and accumulated_value.
    void backprop(std::shared_ptr<StateNode> leaf, double value)
    {
        std::unique_lock<std::shared_mutex> lock(tree_mutex);
        auto node = leaf;
        while (true) {
            node->n_visits += 1;
            auto pa = node->parent_action.lock();
            if (!pa)
                break;  // reached the root
            pa->n_visits += 1;
            pa->accumulated_value += value;
            auto st_parent = pa->parent_state.lock();
            if (!st_parent)
                break;
            node = st_parent;
        }
    }

    // One iteration: selection, expansion, rollout, backpropagation.
    void do_one_iteration()
    {
        std::mt19937 local_rng(std::random_device{}());
        auto leaf = select_and_expand(local_rng);
        double val = parallel_rollout(leaf, /*num_rollouts=*/10);
        backprop(leaf, val);
    }

    // TREE PARALLELIZATION: run N iterations in parallel.
    void run_mcts(int n_iterations)
    {
        if (!root) return;
        #pragma omp parallel for
        for (int i = 0; i < n_iterations; i++){
            do_one_iteration();
        }
    }

    // Pick the best action from the root.
    int get_best_action()
    {
        std::shared_lock<std::shared_mutex> lock(tree_mutex);
        if (!root || root->children_actions.empty())
            return -1;
        double best_val = -1e9;
        int best_idx = -1;
        for (int i = 0; i < (int)root->children_actions.size(); i++){
            auto &an = root->children_actions[i];
            if (an->n_visits == 0)
                continue;
            double q = an->accumulated_value / (double)an->n_visits;
            if (q > best_val){
                best_val = q;
                best_idx = i;
            }
        }
        if (best_idx < 0)
            return -1;
        return root->actions[best_idx];
    }

    // Choose the next state matching the environment outcome to "re-root" the tree.
    void select_branch(int action, const std::vector<std::vector<int>> &new_state)
    {
        std::unique_lock<std::shared_mutex> lock(tree_mutex);
        if (!root)
            return;

        int aidx = -1;
        for (int i = 0; i < (int)root->actions.size(); i++){
            if (root->actions[i] == action) {
                aidx = i;
                break;
            }
        }
        if (aidx < 0) {
            root.reset();
            return;
        }

        auto anode = root->children_actions[aidx];
        int found = -1;
        for (int i = 0; i < (int)anode->children_states.size(); i++){
            if (states_equal(anode->children_states[i]->state, new_state)) {
                found = i;
                break;
            }
        }
        if (found < 0) {
            root.reset();
            return;
        }
        auto new_root = anode->children_states[found];
        new_root->parent_action.reset();
        root = new_root;
    }
};

////////////////////////////////////////////////////////////////////////////////
// 5) Utility function to print the board
////////////////////////////////////////////////////////////////////////////////

static void print_board(const std::vector<std::vector<int>>& board)
{
    for (auto &row : board){
        for (size_t i = 0; i < row.size(); i++){
            if (i > 0) std::cout << " ";
            std::cout << row[i];
        }
        std::cout << "\n";
    }
}

////////////////////////////////////////////////////////////////////////////////
// 6) Main demonstration
////////////////////////////////////////////////////////////////////////////////

int main()
{
    // Create environment
    GameEnv env;
    // Create parallel MCTS with chosen exploration coefficient and discount factor.
    ParallelMCTS mcts(env, /*explore_coef=*/0.01, /*discount=*/0.999);

    // Get initial board from the environment.
    auto state = env.get_initial_state();
    mcts.init_root(state);

    int step = 0;
    while (true) {
        std::cout << "Step " << step << "\n";
        print_board(state);

        // Run parallel MCTS for a given number of iterations (tune as desired).
        int n_iter = 2000;  
        mcts.run_mcts(n_iter);

        // Pick best action from the root.
        int best_a = mcts.get_best_action();
        if (best_a < 0) {
            std::cout << "No moves left. Stopping.\n";
            break;
        }

        // Apply environment step.
        auto [n_states, probas, rewards, dones] = env.make_transition(best_a, state);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<int> dd(probas.begin(), probas.end());
        int idx = dd(gen);

        state = n_states[idx];
        double rew = rewards[idx];
        bool done = dones[idx];

        std::cout << "Chosen action = " << best_a 
                  << ", reward = " << rew 
                  << ", done = " << done << "\n";

        if (done) {
            std::cout << "Game finished!\n";
            print_board(state);
            break;
        }

        // Re-root the tree based on the chosen action and resulting state.
        mcts.select_branch(best_a, state);
        step++;
    }

    return 0;
}