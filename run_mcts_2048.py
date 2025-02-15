import numpy as np
from MCTS import StochasticMCTS
from pyenv_2048 import Game2048

def print_board(board):
    """Helper function to visualize the board in text form."""
    for row in board:
        print("\t".join(str(x) for x in row))

def run_mcts_2048_simulation():
    # 1) Initialize the 2048 game and MCTS
    mcts = StochasticMCTS(game=Game2048(), exploration_coef=0.1, discount_factor=0.99)
    current_state = mcts.game.get_initial_state()

    # 2) Main simulation loop
    step = 0
    while True:
        print('-' * 20)
        print(f"Step {step}")
        print_board(current_state)  # Visualize the board

        # 2a) Get the MCTS-selected best action from the current state
        best_action = mcts.get_best_action(
            state=current_state,
            n_tree_iterations=1000,       # Increase or tune these
            n_rollout_simulations=1    # for better results
        )
        if best_action is None:
            print("No moves left. Stopping.")
            break

        # 2b) Apply the best action in the environment
        #    This game is stochastic, so we get multiple possible next states.
        next_states, next_probas, rewards, dones = mcts.game.make_transition(best_action, current_state)

        # 2c) Randomly pick which next state occurs, according to next_probas
        idx = np.random.choice(len(next_states), p=next_probas)
        current_state = next_states[idx]
        reward = rewards[idx]
        done = dones[idx]

        # 2d) Check if we have reached tile 16 or no moves
        if done:
            print(f"Reached tile 16 (reward={reward}) or no more moves possible. Stopping.")
            print("Final board:")
            print_board(current_state)
            print('Total time:', mcts.game.total_time)
            break
        
        # select the branch for the next step
        mcts.select_branch(best_action, current_state)
        step += 1


if __name__ == "__main__":
    run_mcts_2048_simulation()
