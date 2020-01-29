import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_policy, print_values

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U','D','L','R')

#we not have some randomness due to the "wind"
#you go your desired way with a propability of 0.5
#you go direction a' != a randomly with probability of 0.5/3


if __name__ == '__main__':

    #we set the step cost higher, so it will want to end the game as fast as possible
    #the agent might even go to the negative reward to end the game sooner
    grid = negative_grid(step_cost=-1.0)

    # print rewards
    print('rewards')
    print_values(grid.rewards, grid)

    #state -> action
    #well randomly choose and action and update
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    #initial policy
    print('policy')
    print_policy(policy,grid)

    #initialize v
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            #terminal state
            V[s] = 0

    #repeat until convergence - break when policy doesnt change
    while True:
        #policy evaluation
        while True:
            biggest_change = 0
            for s in states:
                old_v = V[s]

                new_v = 0
                if s in policy:
                    for a in ALL_POSSIBLE_ACTIONS:
                        if a == policy[s]:
                            p = 0.5
                        else:
                            p = 0.5/3
                        grid.set_state(s)
                        r = grid.move(a)
                        new_v += p*(r + GAMMA*V[grid.current_state()])
                    V[s] = new_v
                    biggest_change = max(biggest_change,np.abs(old_v - V[s]))
            if biggest_change < SMALL_ENOUGH:
                break
        #policy improvement step
        is_policy_converged = True
        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None
                best_value = float('-inf')
                #now loop through all possible actions to find the best current actions
                for a in ALL_POSSIBLE_ACTIONS: #chosen action
                    v = 0
                    #need a second loop
                    for a2 in ALL_POSSIBLE_ACTIONS: #resulting action
                        if a == a2:
                            p = 0.5
                        else:
                            p = 0.5/3
                        grid.set_state(s)
                        r = grid.move(a2)
                        v += p*(r + GAMMA*V[grid.current_state()])
                    if v > best_value:
                        best_value = v
                        new_a = a
                policy[s] = new_a
                if new_a != old_a:
                    is_policy_converged = False
        if is_policy_converged:
            break
    print('Values')
    print_values(V, grid)
    print('policy')
    print_policy(policy, grid)
