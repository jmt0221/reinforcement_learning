import numpy as np
import matplotlib.pyplot as plt
from grid_world import grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 1e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U','D','L','R')


#this is determinist
#all p(s',r|s,a) = 1 or 0



if __name__ == '__main__':

    #we use the negative grid so we can make the agent as efficient as possible
    grid = negative_grid()


    #print rewards
    print('rewards')
    print_values(grid.rewards, grid)


    #state -> action
    #well randomly choose an action and update as we learn
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    #initial policy
    print('initial policy')
    print_policy(policy,grid)

    #initialize V(s)
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0



    #repeat until convergence - break when policy doesnt change
    while True:

        #policy evaluation step
        while True:
            biggest_change = 0
            for s in states:
                old_v = V[s]

                if s in policy:
                    a = policy[s]
                    grid.set_state(s)
                    r = grid.move(a)
                    V[s] = r + GAMMA * V[grid.current_state()]
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))
            if biggest_change < SMALL_ENOUGH:
                break
        #policy improvement step
        is_policy_convereged = True
        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None
                best_value = float('-inf')
                #loop through all possible actions to find the best current action
                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA * V[grid.current_state()]
                    if v > best_value:
                        best_value = v
                        new_a = a
                policy[s] = new_a
                if new_a != old_a:
                    is_policy_convereged = False
        if is_policy_convereged:
            break
    print('Values')
    print_values(V,grid)
    print_policy(policy,grid)









