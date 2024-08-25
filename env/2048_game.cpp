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

bool can_make_move()
{
    bool found = false;
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            if(state[i][j] == 0) {
                found = true;
                break;
            }
            if(j > 0 && state[i][j] == state[i][j - 1]) {
                found = true;
                break;
            }
            if(i > 0 && state[i][j] == state[i - 1][j]) {
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

int slide(int direction) {
    int n_changes = 0;
    if (direction == 0) {
        for (int j = 0; j < 4; j++) {
            int idx = 3;
            for (int i = 3; i >= 0; i--) {
                if(state[i][j] != 0) {
                    if(idx != i) {
                        n_changes++;
                        state[idx][j] = state[i][j];
                        state[i][j] = 0;
                    }
                    idx--;
                }
            }
        }
    }else if(direction == 1) {
        for (int j = 0; j < 4; j++) {
            int idx = 0;
            for (int i = 0; i < 4; i++) {
                if(state[i][j] != 0) {
                    if(idx != i) {
                        n_changes++;
                        state[idx][j] = state[i][j];
                        state[i][j] = 0;
                    }
                    idx++;
                }
            }
        }
    }else if(direction == 2) {
        for (int i = 0; i < 4; i++) {
            int idx = 3;
            for (int j = 3; j >= 0; j--) {
                if(state[i][j] != 0) {
                    if(idx != j) {
                        n_changes++;
                        state[i][idx] = state[i][j];
                        state[i][j] = 0;
                    }
                    idx--;
                }
            }
        }
    }else if(direction == 3) {
        for (int i = 0; i < 4; i++) {
            int idx = 0;
            for (int j = 0; j < 4; j++) {
                if(state[i][j] != 0) {
                    if(idx != j) {
                        n_changes++;
                        state[i][idx] = state[i][j];
                        state[i][j] = 0;
                    }
                    idx++;
                }
            }
        }
    }
    return n_changes;
}

long long merge(int direction)
{
    long long partial_reward = 0;
    if (direction == 0) {
        for(int j = 0; j < 4; j++) {
            int idx = 3;
            int last_val = 0;
            for(int i = 3; i >= 0; i--) {
                if(state[i][j] != 0 && last_val == 0) {
                    last_val = state[i][j];
                }else if(state[i][j] != 0) {
                    if(last_val == state[i][j]) {
                        state[idx][j] = last_val * 2;
                        state[i][j] = 0;
                        partial_reward += state[idx][j];
                        last_val = 0;
                        idx--;
                    }else{
                        last_val = state[i][j];
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
                if(state[i][j] != 0 && last_val == 0) {
                    last_val = state[i][j];
                }else if(state[i][j] != 0) {
                    if(last_val == state[i][j]) {
                        state[idx][j] = last_val * 2;
                        state[i][j] = 0;
                        partial_reward += state[idx][j];
                        last_val = 0;
                        idx++;
                    }else{
                        last_val = state[i][j];
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
                if(state[i][j] != 0 && last_val == 0) {
                    last_val = state[i][j];
                }else if(state[i][j] != 0) {
                    if(last_val == state[i][j]) {
                        state[i][idx] = last_val * 2;
                        state[i][j] = 0;
                        partial_reward += state[i][idx];
                        last_val = 0;
                        idx--;
                    }else{
                        last_val = state[i][j];
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
                if(state[i][j] != 0 && last_val == 0) {
                    last_val = state[i][j];
                }else if(state[i][j] != 0) {
                    if(last_val == state[i][j]) {
                        state[i][idx] = last_val * 2;
                        state[i][j] = 0;
                        partial_reward += state[i][idx];
                        last_val = 0;
                        idx++;
                    }else{
                        last_val = state[i][j];
                        idx++;
                    }
                }
            }
        }
    }
    return partial_reward;
}

public:
Game() : state(4, std::vector<int>(4, 0)), dev(), rng(dev()), score(0)
{
    try_spawn_new_tile();
    try_spawn_new_tile();
}

std::tuple<
    std::vector<std::vector<int>>,
    std::pair<std::vector<std::vector<std::vector<int>>>, std::vector<double>>,
    long long,
    bool
> action(int action)
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
    if(action == 0) {
        changes += slide(0);
        part_reward = merge(0);
        changes += slide(0);
    }else if(action == 1) {
        changes += slide(1);
        part_reward = merge(1);
        changes += slide(1);
    }else if(action == 2) {
        changes += slide(2);
        part_reward = merge(2);
        changes += slide(2);
    }else if(action == 3) {
        changes += slide(3);
        part_reward = merge(3);
        changes += slide(3);
    }
    score += part_reward;

    auto state_before_spawning = state;
    if(can_make_move() && changes == 0 && part_reward == 0) {
        bool done = !can_make_move();
        return std::make_tuple(state, get_possible_positions(state_before_spawning), -10, done);
    }
    
    if(changes > 0 || part_reward > 0) {
        try_spawn_new_tile();
    }
    bool done = !can_make_move();
    if(done == true) {
        part_reward = -100;
    }
    return std::make_tuple(state, get_possible_positions(state_before_spawning), part_reward, done);
}

std::pair<std::vector<std::vector<std::vector<int>>>, std::vector<double>> get_possible_positions(
    std::vector<std::vector<int>> state_before_spawning
)
{
    std::vector<std::pair<int, int>> empty_places;
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            if (state[i][j] == 0) {
                empty_places.push_back({i, j});
            }
        }
    }

    std::vector<std::vector<std::vector<int>>> state_candidates;
    std::vector<double> probas; 
    auto state_copy = state_before_spawning;
    
    for(size_t t = 0; t < empty_places.size(); t++) {
        auto [i, j] = empty_places[t];
        
        state_copy[i][j] = 2;
        state_candidates.push_back(state_copy);
        probas.push_back(0.9);
        state_copy[i][j] = 4;
        state_candidates.push_back(state_copy);
        probas.push_back(0.1);
        state_copy[i][j] = 0;
    }
    for(size_t t = 0; t < probas.size(); t++) {
        probas[t] /= (probas.size() / 2);
    }
    return {state_candidates, probas};
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