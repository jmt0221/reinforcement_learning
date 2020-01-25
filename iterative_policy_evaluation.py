import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid

SMALL_ENOUGH = 10e-4 #threshold for convergence

def print_values(V,g):
	'''
	takes in value dictionary V and grid G
	'''
	for i in range(g.width):
		print("--------------------------")
		for j in range(g.height):
			v = V.get((i,j),0)
			if v >= 0:
				print(' %.2f|' % v,end="")
			else:
				print('%.2f|' % v,end="") # the - sign takes up an extra space
		print("")
def print_policy(P,g):
	for i in range(g.width):
		print("--------------------------")
		for j in range(g.height):
			a = P.get((i,j), ' ')
			print("  %s  |" % a,end="")
		print("")





if __name__ == '__main__':
	#iterative policy evaluation 
	#given a policy lets find its value function V(s)
	# we do this for both uniform random policy and fixed policy
	#Note:
	# there are 2 sources of randomnesss
	# p(a|a) - deciding what action to take given the state
	# p(s',r|s,a) - the next state and reward given your action=state pair
	#we are only modeling p(a|s) = uniform
	#how would the code change if p(s',r|s,a) is not deterministic

	grid = standard_grid()

	#states will be positions (i,j)
	states = grid.all_states()


	### uniformly random actions ###
	#initalize V(s)  = 0
	V = {}
	for s in states:
		V[s] = 0
	gamma = 1.0

	#repeat until convergence
	while True:
		biggest_change = 0
		for s in states:
			old_v = V[s]


			#V(s) only has value if its not a terminal state
			if s in grid.actions:

				new_v = 0 # we will accumulate the answer
				p_a = 1.0/len(grid.actions[s]) #each actions has equal prob since uniform
				for a in grid.actions[s]:
					grid.set_state(s)
					r = grid.move(a)
					new_v += p_a * (r + gamma*V[grid.current_state()])
				V[s] = new_v
				biggest_change = max(biggest_change, np.abs(old_v - V[s]))
		if biggest_change < SMALL_ENOUGH:
			break
	print("Values for uniformly random actions:")
	print_values(V,grid)
	print('\n\n')


	### fixed policy ###
	policy = {
			(2,0): 'U',
			(1,0): 'U',
			(0,0): 'R',
			(0,1): 'R',
			(0,2): 'R',
			(1,2): 'R',
			(2,1): 'R',
			(2,2): 'R',
			(2,3): 'U'
		}
	print_policy(policy,grid)

	#initialize V(s) = 0
	V ={}
	for s in states: 
		V[s] = 0

	#lets see how V(S) changes as we get further away from the reward
	gamma = 0.9 # discount factor

	#repeat until convergence
	while True:
		biggest_change = 0
		for s in states:
			old_v = V[s]


			#V(s) only has value if its not a terminal state
			if s in grid.actions:

				new_v = 0 # we will accumulate the answer
				p_a = 1.0/len(grid.actions[s]) #each actions has equal prob since uniform
				for a in grid.actions[s]:
					grid.set_state(s)
					r = grid.move(a)
					new_v += p_a * (r + gamma*V[grid.current_state()])
				V[s] = new_v
				biggest_change = max(biggest_change, np.abs(old_v - V[s]))
		if biggest_change < SMALL_ENOUGH:
			break
	print("Values for fixed policy:")
	print_values(V,grid)































