from __future__ import annotations
import numpy as np
import warnings
from abstract_game import OnePlayerGame
from dataclasses import dataclass, field
from typing import List, Any, Optional


@dataclass(eq=False)
class StateNode:
    state: Any
    n_visits: int = 0
    actions: List[int] = field(default_factory=list)
    children_nodes_actions: List[ActionNode] = field(default_factory=list)
    parent_action: Optional[ActionNode] = None
    terminal: bool = False
    terminal_reward: float = 0.0
    depth: int = 0

@dataclass(eq=False)
class ActionNode:
    n_visits: int = 0
    accumulated_value: float = 0.0
    children_states_nodes: List[StateNode] = field(default_factory=list)
    children_states_probas: List[float] = field(default_factory=list)
    parent_state: Optional[StateNode] = None

class StochasticMCTS:
    def __init__(self, game: OnePlayerGame, exploration_coef: float = 1.0, discount_factor: float = 0.999):
        if exploration_coef < 0:
            raise ValueError("exploration_coef must be greater than 0")
        if np.isclose(exploration_coef, 0):
            warnings.warn('exploration_coef is set to 0')
        self.game = game
        self.gamma = discount_factor
        self.exploration_coef = exploration_coef
        self.root = None

    def select_child_action_ucb(self, node):
        if node.n_visits <= 0:
            raise AssertionError('Parent node n_visits must be > 0.')
        action_scores = np.zeros(len(node.children_nodes_actions), dtype=np.float32)
        for child_idx, child in enumerate(node.children_nodes_actions):
            if child.n_visits != 0:
                state_action_value = child.accumulated_value / child.n_visits
                exploration_bonus = (
                    self.exploration_coef
                    * np.sqrt(np.log(node.n_visits) / child.n_visits)
                )
                action_scores[child_idx] = state_action_value + exploration_bonus
            else:
                # not visited yet => infinite priority for exploration
                action_scores[child_idx] = np.inf

        best_indices = np.where(action_scores == action_scores.max())[0]
        action_node_idx = np.random.choice(best_indices)
        return action_node_idx

    def run(self, state, n_tree_iterations, n_rollout_simulations=10):
        #if self.root is None:
        self.root = StateNode(state=state)
        for _ in range(n_tree_iterations):
            # forward pass (select)
            node = self.root
            while len(node.children_nodes_actions) != 0:
                # select the action according to UCB
                action_node = node.children_nodes_actions[self.select_child_action_ucb(node)]
                # select the next state randomly according to its probabilities
                node = np.random.choice(action_node.children_states_nodes, p=action_node.children_states_probas)

            # expansion
            if not node.terminal:
                children_nodes = []
                actions = []
                possible_actions = self.game.get_possible_actions(node.state)
                for action_idx in possible_actions:
                    states, probas, rewards, dones = self.game.make_transition(action_idx, node.state)

                    action_node = ActionNode(parent_state=node)

                    children_state_nodes = []
                    children_state_probas = []
                    for state, proba, reward, done in zip(states, probas, rewards, dones):
                        state_node = StateNode(
                            state=state,
                            parent_action=action_node,
                            terminal=done,
                            terminal_reward=reward if done else 0.0,
                            depth=node.depth + 1
                        )
                        children_state_nodes.append(state_node)
                        children_state_probas.append(proba)
                    
                    action_node.children_states_nodes = children_state_nodes
                    action_node.children_states_probas = children_state_probas

                    children_nodes.append(action_node)
                    actions.append(action_idx)
                node.children_nodes_actions = children_nodes
                node.actions = actions

                # selecting random action uniformly
                random_action_node = np.random.choice(node.children_nodes_actions)
                # selecting random state according to transition probas
                node_to_rollout = np.random.choice(random_action_node.children_states_nodes, p=random_action_node.children_states_probas)
            else:
                node_to_rollout = node

            # rollout
            if not node_to_rollout.terminal:
                rollout_reward = self.rollout(node_to_rollout, n_rollout_simulations)
            else:
                rollout_reward = node_to_rollout.terminal_reward

            # backprop
            node = node_to_rollout
            while node.parent_action is not None:
                node.n_visits += 1
                action_node = node.parent_action
                action_node.accumulated_value += rollout_reward
                action_node.n_visits += 1
                node = action_node.parent_state
            node.n_visits += 1  # increase n_visits for root which doesn't have parent_action

        return self.root

    def rollout(self, node: StateNode, n_rollout_simulations: int):
        total_reward = 0
        for _ in range(n_rollout_simulations):
            game_finished = False
            state = node.state
            total_depth = node.depth
            #how_deep = 0
            while not game_finished:
                possible_actions = self.game.get_possible_actions(state)
                if not possible_actions:
                    print('TTTT')
                    for x in state:
                        print(x)
                    raise AssertionError("Rollout: got no possible action in non-terminal state")
                # randomly select the action
                random_action = np.random.choice(possible_actions)
                # get possible transitions 
                states, probas, rewards, dones = self.game.make_transition(random_action, state)
                # sample the next state according to probas
                idx = np.random.choice(len(states), p=probas)
                state, reward, done = states[idx], rewards[idx], dones[idx]
                game_finished = done
                total_depth += 1
                #how_deep += 1
                #if how_deep > 20:
                #    break
            total_reward += (self.gamma ** total_depth) * reward
        avg_reward = total_reward / n_rollout_simulations
        return avg_reward

    def get_best_action(self, state, n_tree_iterations=10, n_rollout_simulations=5):
        root = self.run(state, n_tree_iterations, n_rollout_simulations)

        # pick best child
        scores = np.array([
            child.accumulated_value / child.n_visits
            for child in root.children_nodes_actions
        ])
        print('Scores: ', scores)
        print('Action: ', root.actions)
        best_idx = scores.argmax()
        return root.actions[best_idx]

    def select_branch(self, action, new_state):
        if self.root is None:
            raise AssertionError("Can't select a branch of non-existing tree. Build it first via run command.")
        action_idx = -1
        for i in range(len(self.root.actions)):
            if self.root.actions[i] == action:
                action_idx = i
                break
        if action_idx == -1:
            raise AssertionError(f"The action {action} is not found in the root node.")
        action_node = self.root.children_nodes_actions[action_idx]
        state_idx = -1
        for i in range(len(action_node.children_states_nodes)):
            if action_node.children_states_nodes[i].state == new_state:
                state_idx = i
        if state_idx == -1:
            print(f'State {new_state} was not found from state {self.root.state} and action {action}. Reseting to root to None.')
            self.root = None
        else:
            self.root = action_node.children_states_nodes[state_idx]
