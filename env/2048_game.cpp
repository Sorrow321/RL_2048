#include <stdexcept>
#include <vector>
#include <random>
#include <iostream>
#include <tuple>
//#include <ncurses.h>

class Game
{
private:
std::vector<std::vector<int>> state;
std::random_device dev;
std::mt19937 rng;
long long score;

bool try_spawn_new_tile()
{
    std::vector<std::pair<int, int>> empty_places;
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            if (state[i][j] == 0) {
                empty_places.push_back({i, j});
            }
        }
    }

    if(empty_places.empty()) {
        return false;
    }

    std::uniform_int_distribution<> dist(0, empty_places.size() - 1);
    int random_pos = dist(rng);
    std::pair<int, int> pos = empty_places[random_pos];

    std::uniform_real_distribution<double> proba;
    
    int val_for_insertion;
    if (proba(rng) < 0.1) {
        val_for_insertion = 4;
    }else{
        val_for_insertion = 2;
    }
    state[pos.first][pos.second] = val_for_insertion;
    return true;
}

bool can_make_move(std::vector<std::vector<int>> state_to_check)
{
    bool found = false;
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            if(state_to_check[i][j] == 0) {
                found = true;
                break;
            }
            if(j > 0 && state_to_check[i][j] == state_to_check[i][j - 1]) {
                found = true;
                break;
            }
            if(i > 0 && state_to_check[i][j] == state_to_check[i - 1][j]) {
                found = true;
                break;
            }
        }
        if(found) {
            break;
        }
    }
    return found;
}

std::pair<std::vector<std::vector<int>>, int> slide(std::vector<std::vector<int>> state_to_slide, int direction) {
    int n_changes = 0;
    if (direction == 0) {
        for (int j = 0; j < 4; j++) {
            int idx = 3;
            for (int i = 3; i >= 0; i--) {
                if(state_to_slide[i][j] != 0) {
                    if(idx != i) {
                        n_changes++;
                        state_to_slide[idx][j] = state_to_slide[i][j];
                        state_to_slide[i][j] = 0;
                    }
                    idx--;
                }
            }
        }
    }else if(direction == 1) {
        for (int j = 0; j < 4; j++) {
            int idx = 0;
            for (int i = 0; i < 4; i++) {
                if(state_to_slide[i][j] != 0) {
                    if(idx != i) {
                        n_changes++;
                        state_to_slide[idx][j] = state_to_slide[i][j];
                        state_to_slide[i][j] = 0;
                    }
                    idx++;
                }
            }
        }
    }else if(direction == 2) {
        for (int i = 0; i < 4; i++) {
            int idx = 3;
            for (int j = 3; j >= 0; j--) {
                if(state_to_slide[i][j] != 0) {
                    if(idx != j) {
                        n_changes++;
                        state_to_slide[i][idx] = state_to_slide[i][j];
                        state_to_slide[i][j] = 0;
                    }
                    idx--;
                }
            }
        }
    }else if(direction == 3) {
        for (int i = 0; i < 4; i++) {
            int idx = 0;
            for (int j = 0; j < 4; j++) {
                if(state_to_slide[i][j] != 0) {
                    if(idx != j) {
                        n_changes++;
                        state_to_slide[i][idx] = state_to_slide[i][j];
                        state_to_slide[i][j] = 0;
                    }
                    idx++;
                }
            }
        }
    }
    return {state_to_slide, n_changes};
}

std::pair<std::vector<std::vector<int>>, long long> merge(std::vector<std::vector<int>> state_to_merge, int direction)
{
    long long partial_reward = 0;
    if (direction == 0) {
        for(int j = 0; j < 4; j++) {
            int idx = 3;
            int last_val = 0;
            for(int i = 3; i >= 0; i--) {
                if(state_to_merge[i][j] != 0 && last_val == 0) {
                    last_val = state_to_merge[i][j];
                }else if(state_to_merge[i][j] != 0) {
                    if(last_val == state_to_merge[i][j]) {
                        state_to_merge[idx][j] = last_val * 2;
                        state_to_merge[i][j] = 0;
                        partial_reward += state_to_merge[idx][j];
                        last_val = 0;
                        idx--;
                    }else{
                        last_val = state_to_merge[i][j];
                        idx--;
                    }
                }
            }
        }
    }else if(direction == 1) {
        for(int j = 0; j < 4; j++) {
            int idx = 0;
            int last_val = 0;
            for(int i = 0; i < 4; i++) {
                if(state_to_merge[i][j] != 0 && last_val == 0) {
                    last_val = state_to_merge[i][j];
                }else if(state_to_merge[i][j] != 0) {
                    if(last_val == state_to_merge[i][j]) {
                        state_to_merge[idx][j] = last_val * 2;
                        state_to_merge[i][j] = 0;
                        partial_reward += state_to_merge[idx][j];
                        last_val = 0;
                        idx++;
                    }else{
                        last_val = state_to_merge[i][j];
                        idx++;
                    }
                }
            }
        }
    }else if(direction == 2) {
        for(int i = 0; i < 4; i++) {
            int idx = 3;
            int last_val = 0;
            for(int j = 3; j >= 0; j--) {
                if(state_to_merge[i][j] != 0 && last_val == 0) {
                    last_val = state_to_merge[i][j];
                }else if(state_to_merge[i][j] != 0) {
                    if(last_val == state_to_merge[i][j]) {
                        state_to_merge[i][idx] = last_val * 2;
                        state_to_merge[i][j] = 0;
                        partial_reward += state_to_merge[i][idx];
                        last_val = 0;
                        idx--;
                    }else{
                        last_val = state_to_merge[i][j];
                        idx--;
                    }
                }
            }
        }
    }else if(direction == 3) {
        for(int i = 0; i < 4; i++) {
            int idx = 0;
            int last_val = 0;
            for(int j = 0; j < 4; j++) {
                if(state_to_merge[i][j] != 0 && last_val == 0) {
                    last_val = state_to_merge[i][j];
                }else if(state_to_merge[i][j] != 0) {
                    if(last_val == state_to_merge[i][j]) {
                        state_to_merge[i][idx] = last_val * 2;
                        state_to_merge[i][j] = 0;
                        partial_reward += state_to_merge[i][idx];
                        last_val = 0;
                        idx++;
                    }else{
                        last_val = state_to_merge[i][j];
                        idx++;
                    }
                }
            }
        }
    }
    return {state_to_merge, partial_reward};
}

public:
Game() : state(4, std::vector<int>(4, 0)), dev(), rng(dev()), score(0)
{
    try_spawn_new_tile();
    try_spawn_new_tile();
}

std::tuple<
    std::vector<std::vector<std::vector<int>>>,
    std::vector<double>,
    std::vector<long long>,
    std::vector<bool>
> simulate_action(std::vector<std::vector<int>> state_for_action, int action)
{
    /*
        action:
                0 - down
                1 - up
                2 - right
                3 - left
    */
    if(action < 0 or action > 3) {
        throw std::invalid_argument("Bad action idx");
    }

    long long part_reward;
    int changes = 0;

    auto [s, n_changes] = slide(state_for_action, action);
    changes += n_changes;
    state_for_action = s;

    std::tie(s, part_reward) = merge(state_for_action, action);
    state_for_action = s;

    std::tie(s, n_changes) = slide(state_for_action, action);
    changes += n_changes;
    state_for_action = s;

    score += part_reward;

    auto state_before_spawning = state_for_action;
    return get_possible_positions(state_before_spawning, part_reward);
}


std::vector<int> get_possible_actions(std::vector<std::vector<int>> state_before_spawning)
{
    std::vector<int> possible_actions;
    for(int action_idx = 0; action_idx < 4; action_idx++) {
        auto init_state = state_before_spawning;
        int changes = 0;
        long long part_reward;

        auto [s, n_changes] = slide(init_state, action_idx);
        changes += n_changes;
        init_state = s;

        std::tie(s, part_reward) = merge(init_state, action_idx);
        init_state = s;

        std::tie(s, n_changes) = slide(init_state, action_idx);
        changes += n_changes;
        init_state = s;

        if(changes > 0 || part_reward > 0) {
            possible_actions.push_back(action_idx);
        }
    }
    return possible_actions;
}

std::tuple<
    std::vector<std::vector<std::vector<int>>>,
    std::vector<double>,
    std::vector<long long>,
    std::vector<bool>
> get_possible_positions(
    std::vector<std::vector<int>> state_before_spawning,
    long long reward
)
{
    std::vector<std::pair<int, int>> empty_places;
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            if (state_before_spawning[i][j] == 0) {
                empty_places.push_back({i, j});
            }
        }
    }

    std::vector<std::vector<std::vector<int>>> state_candidates;
    std::vector<long long> rewards;
    std::vector<bool> dones;
    std::vector<double> probas; 
    auto state_copy = state_before_spawning;
    
    for(size_t t = 0; t < empty_places.size(); t++) {
        auto [i, j] = empty_places[t];
        
        state_copy[i][j] = 2;
        state_candidates.push_back(state_copy);
        probas.push_back(0.9);
        rewards.push_back(reward);
        dones.push_back(!can_make_move(state_copy));

        state_copy[i][j] = 4;
        state_candidates.push_back(state_copy);
        probas.push_back(0.1);
        rewards.push_back(reward);
        dones.push_back(!can_make_move(state_copy));

        state_copy[i][j] = 0;
    }
    for(size_t t = 0; t < probas.size(); t++) {
        probas[t] /= (probas.size() / 2);
    }
    return std::make_tuple(state_candidates, probas, rewards, dones);
}

/*
void print_state()
{
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            if(state[i][j] != 0) {
                printw("%d\t", state[i][j]);
            }else{
                printw("#\t");
            }
            
        }
        printw("\n");
    }
}
*/


const std::vector<std::vector<int>>& get_state()
{
    return state;
}

int get_current_score()
{
    return score;
}

const std::vector<std::vector<int>>& reset()
{
    score = 0;
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            state[i][j] = 0;
        }
    }
    try_spawn_new_tile();
    try_spawn_new_tile();
    return state;
}

};


/*
int main() {
    Game game;
    initscr();
    raw();
    keypad(stdscr, TRUE);  // Enable special keys
    noecho();  // Don't echo the typed characters

    game.print_state(); // Use ncurses functions inside this method
    refresh();

    int ch;
    int action;
    while (true) {
        ch = getch();
        switch (ch) {
            case KEY_LEFT:
                action = 3;
                break;
            case KEY_RIGHT:
                action = 2;
                break;
            case KEY_UP:
                action = 1;
                break;
            case KEY_DOWN:
                action = 0;
                break;
            case 'q':
                endwin();  // End ncurses mode
                return 0;
        }
        auto [state, part_reward, done] = game.action(action);
        clear();  // Clear the screen
        move(0, 0);  // Move the cursor to the top-left
        game.print_state();  // Ensure that ncurses functions are used inside
        refresh();
        if (done) {
            printw("Game over\n");
            break;
        }
    }
    getch(); // Wait for a key press before ending ncurses mode
    endwin();  // End ncurses mode
    return 0;
}
*/