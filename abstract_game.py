from abc import ABC, abstractmethod


class OnePlayerGame(ABC):
    @abstractmethod
    def get_initial_state(self):
        """
            Returns:
                Tuple (state, player_idx) - current state of the game, ID of the player that moves next
        """
        pass

    @abstractmethod
    def get_possible_actions(self, state):
        """
            Given the player to move and the state, returns the list of possible actions
            Params:
                state - state of the game, depends on the implementation
            Returns:
                List[int] - list of possible action IDs
        """
        pass

    @abstractmethod
    def make_transition(self, action_idx, state):
        """
            Samples the next state from transition function p(s' | s, a). 
            Params:
                action_idx - action to do from the list of possible actions
                state - the position of the game in which we make the move
            Returns:
                List[Tuple] [(next_state, reward, done), proba] - list of tuples of descriptions of the next states along with their probabilities. The description is also a tuple of 3 values: the next state, reward, flag done telling you if the game is finished.
        """
        pass
