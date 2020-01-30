import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_policy, print_values

GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


#this script implements the Monte Carlo Exploring-starts method
#for finding optimal policy

def play_game(grid, policy):
    '''
    reset game to start at random position
    we need to do this because given our current deterministic policy
    we would never end up at certain states, but we still want to measure them
    :param grid: the grid class object
    :param policy: dictionary containing policies
    :return: a list of states and corresponding returns
    '''

    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    a = np.random.choice(ALL_POSSIBLE_ACTIONS) #first action is uniformly random

    #be aware of timing
    #each triple is s(t), a(t), r(t)
    #but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)
    states_actions_rewards = [(s,a,0)]
    while True:
        old_s = grid.current_state()
        r = grid.move(a)
        s = grid.current_state()
        if old_s == s:
            #hack so we dont stay in the same place
            states_actions_rewards.append((s,None,-100))
            break
        elif grid.game_over():
            states_actions_rewards.append((s,None,r))
            break
        else:
            a = policy[s]
            states_actions_rewards.append((s,a,r))
            print(s)

    # calculate the returns by working backwards from the terminal state
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        #the value of the terminal state is 0 so we ingore first state
        #and we ignore the last G which is meaningless since it doesnt correspond
        if first:
            first = False
        else:
            states_actions_returns.append((s,a,G))
        G = r + GAMMA*G
    states_actions_returns.reverse() # we want it to be in order of state visited
    return states_actions_returns

def max_dict(d):
    '''
    :param d: dictionary
    :return: returns argmax (key) and max (value)
    '''
    max_key = None
    max_val = float('-inf')
    for k,v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val

if __name__ == '__main__':
    grid = negative_grid(step_cost= -0.1)

    #rewards
    print('rewards')
    print_values(grid.rewards,grid)

    #state -> action
    #initialize a random policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    #initialize Q(s,a) and returns
    Q = {}
    returns = {} # dictionary of state:list of returns recieved
    states = grid.all_states()
    for s in states:
        if s in grid.actions: # not terminal
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0 # needs to be initialized so we can argmax
                returns[(s,a)] = []
        else:
            #terminal state or state we cant get to
            pass
    #repeart until convergence
    deltas = []
    for t in range(2000):
        if t % 10 == 0:
            print(t)
        #generate an episoe using pi
        biggest_change = 0
        state_action_returns = play_game(grid,policy)
        seen_state_action_pairs = set()
        for s, a, G in state_action_returns:
            #check if we have already seen
            #called 'first-visit' MC policy evaluation
            sa = (s,a)
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
        #only record deltas for debugging purposes
        deltas.append(biggest_change)

        #update policy
        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]

    plt.plot(deltas)
    plt.show()

    print('final policy')
    print_policy(policy,grid)

    #find V
    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    print('final values')
    print_values(V,grid)


