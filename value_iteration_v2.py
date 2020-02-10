import numpy as np
from grid_world import grid

thresh = 1e-4
GAMMA = 0.9
ACTIONS = ('U', 'D', 'L', 'R')


if __name__ == '__main__':

    #instantiate the grid
    grid = grid()


    #randomly instantiate
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ACTIONS)

    #initialize V(s) randomly between 0 and 1
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            #terminal state so we set Value to 0
            V[s] = 0

    #repeat until convergence
    #V[s] = max[a]{sum[s',r] {p(s',r|s,a)[r + GAMMA * V[s']]}}
    while True:
        max_change = 0
        for s in states:
            old_vs = V[s]

            #V[s] only has policy if not a terminal state
            if s in policy:
                new_v = float('-inf')

                #find max[a]
                for a in ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA * V[grid.current_state()]
                    if v > new_v:
                        new_v = v
                V[s] = new_v
                biggest_change = max(max_change, np.abs(old_vs - V[s]))

        #when the value function converges break out of the loop
        if max_change < thresh:
            break
    #find a policy that leads to optimal value function
    for s in policy.keys():
        best_act = None
        best_value = float('-inf')
        for a in ACTIONS:
            grid.set_state(s)
            r = grid.move(a)
            v = r + GAMMA * V[grid.current_state()]
            if v > best_value:
                 best_value = v
                 best_act = a
        policy[s] = best_act



