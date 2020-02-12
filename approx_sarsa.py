import numpy as np
import matplotlib.pyplot as plt
from grid_world import grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from mc_exploring_starts import max_dict
from td0_prediction import random_action
from sarsa import GAMMA, ALPHA, ACTIONS

SA2IDX = {}
IDX = 0

class Model:
    def __init__(self):
        self.theta = np.random.randn(25) / np.sqrt(25)
        # if we use SA2IDX, a one-hot encoding for every (s,a) pair
        # in reality we wouldn't want to do this b/c we have just
        # as many params as before
        # print "D:", IDX
        # self.theta = np.random.randn(IDX) / np.sqrt(IDX)

    def sa2x(self, s, a):
        # NOTE: using just (r, c, r*c, u, d, l, r, 1) is not expressive enough
        return np.array([
            s[0] - 1 if a == 'U' else 0,
            s[1] - 1.5 if a == 'U' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'U' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'U' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'U' else 0,
            1 if a == 'U' else 0,
            s[0] - 1 if a == 'D' else 0,
            s[1] - 1.5 if a == 'D' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'D' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'D' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'D' else 0,
            1 if a == 'D' else 0,
            s[0] - 1 if a == 'L' else 0,
            s[1] - 1.5 if a == 'L' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'L' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'L' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'L' else 0,
            1 if a == 'L' else 0,
            s[0] - 1 if a == 'R' else 0,
            s[1] - 1.5 if a == 'R' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'R' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'R' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'R' else 0,
            1 if a == 'R' else 0,
            1
        ])
        # if we use SA2IDX, a one-hot encoding for every (s,a) pair
        # in reality we wouldn't want to do this b/c we have just
        # as many params as before
        # x = np.zeros(len(self.theta))
        # idx = SA2IDX[s][a]
        # x[idx] = 1
        # return x
    def predict(self, s, a):
        x = self.sa2x(s, a)
        return self.theta.dot(x)

    def grad(self, s, a):
        return self.sa2x(s, a)

def getQs(model, s):
    #we need Q9s,a) to choose an action
    Qs = {}
    for a in ACTIONS:
        q_sa = model.predict(s,a)
        Qs[a] = q_sa
    return Qs



if __name__ == '__main__':

    grid = negative_grid(step_cost= -0.1)

    # rewards
    print('rewards')
    print_values(grid.rewards, grid)

    #no policy initialization

    states = grid.all_states()
    for s in states:
        SA2IDX[s] = {}
        for a in ACTIONS:
            SA2IDX[s][a] = IDX
            IDX += 1

    #initialize model
    model = Model()

    #repeat until convergence
    # we have two so LR end epsilon can decrease at different rates
    t = 1.0
    t2 = 1.0
    deltas = []
    for i in range(20000):
        if i % 100 == 0:
            t += 10e-3
            t2 += 0.01
        if i % 1000 == 0:
            print('i: ', i)
        alpha = ALPHA/ t2

        #we play instead of generating an episode
        s = (2,0)
        grid.set_state(s)

        #get Q(s) to choose first action
        Qs = getQs(model,s)

        # the first (s, r) tuple is the state we start in and 0
        # (since we don't get a reward) for simply starting the game
        # the last (s, r) tuple is the terminal state and the final reward
        # the value for the terminal state is by definition 0, so we don't
        # care about updating it.
        a = max_dict(Qs)[0]
        a = random_action(a, eps = 0.5/t)
        biggest_change = 0
        while not grid.game_over():
            r = grid.move(a)
            s2 = grid.current_state()

            #need next action since Q(s,a) depends on Q(s',a')
            old_theta = model.theta.copy()
            if grid.is_terminal(s2):
                model.theta += alpha*(r - model.predict(s, a))*model.grad(s, a)
            else:
                #not terminal
                Qs2 = getQs(model,s2)
                a2 = max_dict(Qs2)[0]
                a2 = random_action(a2, eps=0.5/t) #epsilon greedy

                model.theta += alpha*(r+ GAMMA * model.predict(s2, a2) - model.predict(s, a))*model.grad(s, a)

                #next state becomes current state
                s = s2
                a = a2

            biggest_change = max(biggest_change, np.abs(model.theta - old_theta).sum())
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    # determine policy from Q* and find V* from Q*
    policy = {}
    V = {}
    Q = {}
    for s in grid.actions.keys():
        Qs = getQs(model, s)
        Q[s] = Qs
        a, max_q = max_dict(Qs)
        policy[s] = a
        V[s] = max_q

    print('values')
    print_values(V, grid)

    # print('policy')
    # print_policy(policy, grid)








