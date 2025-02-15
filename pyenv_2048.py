from abstract_game import OnePlayerGame
import env.game as game
import time

class Game2048(OnePlayerGame):
    def __init__(self):
        self.env_game = game.Game()
        self.total_time = 0

    def get_initial_state(self):
        return self.env_game.reset()

    def get_possible_actions(self, state):
        return self.env_game.get_possible_actions(state)

    def make_transition(self, action_idx, state):
        t0 = time.time()
        states, probas, rewards, dones = self.env_game.simulate_action(state, action_idx)
        s_new = []
        p_new = []
        r_new = []
        d_new = []
        for s, p, r, d in zip(states, probas, rewards, dones):
            found_16 = False
            for t in s:
                for z in t:
                    if z >= 512:
                        found_16 = True
                        break
                if found_16:
                    break
            if found_16:
                r = 1
                d = True
            else:
                r = 0
                #d = False
            if d and not found_16:
                r = -1
            s_new.append(s)
            p_new.append(p)
            r_new.append(r)
            d_new.append(d)
        self.total_time += (time.time() - t0)
        return s_new, p_new, r_new, d_new
