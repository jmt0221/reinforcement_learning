import numpy as np

#class Agent():
	#pass




class Grid: #Environment
	def __init__(self,width,height,start):
		self.width = width
		self.height = height
		self.i = start[0]
		self.j = start[1]

	def set(self,rewards, actions):
		'''
		rewards should be a dict of:  (i,j):r --> (row,col):reward
		actions should be a dict of:  (i,j):A --> (row,col):list of possible actions
		'''
		self.rewards = rewards
		self.actions = actions

	def set_state(self,s):
		#used to reset the start
		self.i = s[0]
		self.j = s[1]

	def current_state(self):
		return (self.i,self.j)

	def is_terminal(self,s):
		return s not in self.actions

	def move(self,action):
		#check if a legal move first
		if action in self.actions[self.i,self.j]:
			if action == 'U':
				self.i -= 1
			elif action == 'D':
				self.i += 1
			elif action == 'R':
				self.j += 1
			elif action == 'L':
				self.j -= 1
		#return reward if any
		return self.rewards.get((self.i,self.j),0)

	def undo_move(self, action):
		#opposite of what the above move function does
		if action == 'U':
			self.i += 1
		elif action == 'D':
			self.i -= 1
		elif action == 'R':
			self.j -= 1
		elif action == 'L':
			self.j += 1
	#raise an exception if we arrive somewhere we shouldnt
		assert(self.current_state() in self.all_states())

	def game_over(self):
		return (self.i, self.j) not in self.actions

	def all_states(self):
		'''
		possibly buggy but simple way to get all the states
		either a position that has possible next actions
		or a position that yeilds a reward
		'''
		return set(list(self.actions.keys()) + list(self.rewards.keys()))


def standard_grid():
	'''
	define a grid that describes the reward for arriving at each state
	and possible actions at each state
	x means you can go there
	s means start position
	number means reward at that state
	the grid looks like this:
	. . .  1
	. x . -1
	s . .  .
	'''
	g = Grid(3,4,(2,0))
	rewards = {(0,3):1, (1,3):-1}
	actions = {
				(0,0): ('D','R'),
				(0,1): ('L','R'),
				(0,2): ('L','D','R'),
				(1,0): ('U','D'),
				(1,2): ('U','D','R'),
				(2,0): ('U','R'),
				(2,1): ('L','R'),
				(2,2): ('L','R','U'),
				(2,3): ('L','U')
			}

	g.set(rewards,actions)
	return g

def negative_grid(step_cost=-0.1):
	'''
	add a step cost to penalize how much it moves
	'''
	g = standard_grid()
	g.rewards.update({
					(0,0):step_cost,
					(0,1):step_cost,
					(0,2):step_cost,
					(1,0):step_cost,
					(1,2):step_cost,
					(2,0):step_cost,
					(2,1):step_cost,
					(2,2):step_cost,
					(2,3):step_cost
				})
	return g


def play_game(agent,env):
	pass















