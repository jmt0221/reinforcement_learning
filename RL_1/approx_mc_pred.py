import numpy as np
import matplotlib.pyplot as plt
from grid_world import grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

#try to get same results as MC
from monte_carlo_windy import random_action, play_game, SMALL_ENOUGH, GAMMA, ALL_POSSIBLE_ACTIONS

LEARNING_RATE = 0.1

if __name__ == '__main__':

    grid = grid()

    print('rewards')
    print_values(grid.rewards, grid)

    # state -> action
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'L',
        (2, 2): 'U',
        (2, 3): 'L',

    }

    theta = np.random.randn(4) / 2
    def s2x(s):
        '''
        :param s: states
        :return: x = [row, col, row*col, 1] --> 1 is for bias
        '''
        return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1] -3, 1])

    deltas = []
    t= 1.0
    for i in range(20000):
        if i % 100 == 0:
            t += 0.01
        alpha = LEARNING_RATE/t
        #generate an episode using pi
        biggest_change = 0
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_and_returns:
            #first visit MC
            if s not in seen_states:
                old_theta = theta.copy()
                x = s2x(s)
                V_hat = theta.dot(x)
                #deriv V_hat wrt to theta = x
                theta += alpha * (G - V_hat) * x
                biggest_change = max(biggest_change, np.abs(old_theta - theta).sum())
                seen_states.add(s)

        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    # initialize V(s)
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = theta.dot(s2x(s))
        else:  # terminal state or state we cant get to
            V[s] = 0

    print('values')
    print_values(V, grid)

    print('policy')
    print_policy(policy, grid)

