from __future__ import print_function, division
from builtins import range, input


import numpy as np
import matplotlib.pyplot as plt


LENGTH = 3

class Agent:

	def __init__(self, eps=0.1, alpha = 0.5):
		self.eps = eps # prob of choosing random action
		self.alpha = alpha # learning rate
		self.verbose = False # if we want to print information
		self.state_history = []


	def setV(self,V):
		self.V = V

	def set_verbose(self, v):
		self.verbose = v

	def set_symbol(self,sym):
		self.sym = sym

	def reset_history(self):
		self.state_history = []

	def take_action(self, env):
		# we will use the epsilon greedy strategy to choose an action
		r = np.random.rand()
		best_state = None
		if r < self.eps:
			#take random action
			if self.verbose:
				print("Taking Random Action")
			possible_moves = []

			for i in range(LENGTH):
				for j in range(LENGTH):
					if env.is_empty(i,j):
						possible_moves.append((i,j))
			idx = np.random.choice(len(possible_moves))
			next_move = possible_moves[idx]

		else:
			#choose the best action based on current values of states
			#loop through all possible moves, get their values 
			#keep track of best value
			pos2value = {} # for debugging
			next_move = None
			best_value = -1
			for i in range(LENGTH):
				for j in range(LENGTH):
					if env.is_empty(i,j):
						#what is the state if we made this move?
						env.board[i,j] = self.sym
						state = env.get_state()
						env.board[i,j] = 0 # dont forget to change it back
						pos2value[(i,j)] = self.V[state]
						if self.V[state] > best_value:
							best_value = self.V[state]
							best_state = state
							next_move = (i,j)

			#if verbose is true, draw the board w/ the values
			if self.verbose:
				print("Taking a greedy action")
				for i in range(LENGTH):
					print("------------------")
					for j in range(LENGTH):
						if env.is_empty(i,j):
							#print the value
							print(" %.2f|" % pos2value[(i,j)], end="")
						else:
							print("  ", end="")
							if env.board[i,j] == env.x:
								print("x  |", end="")
							elif env.board[i,j] == env.o:
								print("o  |", end="")
							else:
								print("   |", end="")
					print("")
				print("------------------")
		#make the move
		env.board[next_move[0], next_move[1]] = self.sym

	def update_state_history(self,s):
		# cannot put this in take_action, because take_action only happens
    	# once every other iteration for each player
    	# state history needs to be updated every iteration
    	# s = env.get_state() # don't want to do this twice so pass it i
		self.state_history.append(s)

	def update(self,env):
    	#we want to BACKTRACK over the states so that:
    	# V(prev_state) = V(prev_state) + alpha*(target-self.V[prev])
    	#where V(prev_state) = reward if its the most current state
    	# NOTE: we ONLY do this at the end of an episode
    	# not so for all the algorithms we will study
		reward = env.reward(self.sym)
		target = reward
		for prev in reversed(self.state_history):
			value = self.V[prev] + self.alpha*(target-self.V[prev])
			self.V[prev] = value
			target = value
		self.reset_history()



#this class represents a tic-tac-toe game
# is a CS101 type of project
class Environment:
	def __init__(self):
		self.board = np.zeros((LENGTH,LENGTH))
		self.x = -1 # this represents an x on the board(player1)
		self.o = 1 # this represents ax o on the board(player2)
		self.winner = None
		self.ended = False
		self.num_states = 3**(LENGTH*LENGTH)

	def is_empty(self, i ,j):
		#checks to see if boards location is empty
		return self.board[i,j] == 0

	def reward(self, sym):
		#no reward until game is over
		if not self.game_over():
			return 0

		return 1 if self.winner == sym else 0

	def get_state(self):
		#returns the current state, represented as an int, from 0 to [S-1]
		#where S is the set of all possible states (in our case 3^9), some states
		# are impossible but we ignore that detail
		#this is like finding an interger represented by a base-3 number
		k = 0
		h = 0
		for i in range(LENGTH):
			for j in range(LENGTH):
				if self.board[i,j] == 0:
					v = 0
				elif self.board[i,j] == self.x:
					v = 1
				elif self.board[i,j] == self.o:
					v = 2
				h += (3**k) * v
				k +=1
		return h

	def game_over(self, force_recalculate= False):
		#returns true if game over (player wins or theres a draw) o/w False
		#also sets 'winner' instance variable and 'ended' instance variable
		if not force_recalculate and self.ended:
			return self.ended

		#check rows
		for i in range(LENGTH):
			for player in (self.x, self.o):
				if self.board[i].sum() == player*LENGTH:
					self.winner = player
					self.ended = True
					return True

		#check columns
		for i in range(LENGTH):
			for player in (self.x, self.o):
				if self.board[:,i].sum() == player*LENGTH:
					self.winner = player
					self.ended = True
					return True
		
		#check diagonal 
		for j in range(LENGTH):
			for player in (self.x, self.o):
				#left top corner to right bottom corner
				if self.board.trace() == player*LENGTH:
					self.winner = player
					self.ended = True
					return True
				#right top corner to left bottom corner 
				if np.fliplr(self.board).trace() == player*LENGTH:
					self.winner = player
					self.ended = True
					return True

		#check if draw
		if np.all((self.board ==0) == False):
			self.winner = None
			self.ended = True
			return True

		self. winner = None
		return False

	def is_draw(self):
		return self.ended and self.winner is None

	def draw_board(self):
		for i in range(LENGTH):
			print("-------------")
			for j in range(LENGTH):
				print("  ",end="")
				if self.board[i,j] == self.x:
					print("x ", end="")
				elif self.board[i,j] == self.o:
					print("o ", end="")
				else:
					print("  ", end="")
			print("")
		print("-------------")

class Human:
	def __init__(self):
		pass

	def set_symbol(self,sym):
		self.sym = sym

	def take_action(self, env):
		while True:
			move = input("Enter coordinates i,j for your next move (i,j = 0,1,2): ")
			try:
				i,j = move.split(',')
			except:
					print('Please format as: 1,2')
			if i:
				i = int(i)
				j = int(j)
				if env.is_empty(i,j):
					env.board[i,j] = self.sym
					break
	def update(self,env):
		pass

	def update_state_history(self,s):
		pass


	#recrusive function that will return all possible staets (as ints) and 
	#who the corresponding winner is for those states
	#(i,j) refers to the next cell on the board to permute (need to try(-1,0,1))
	#impossible games are ignored, i.e 3xs and 3o's simultaneously
	#since that will never happen in a game

def get_state_hash_and_winner(env,i=0,j=0):
	results = []

	for v in (0,env.x,env.o):
		env.board[i,j] = v #if empty board it should already be 0:
		if j == 2:
			#j goes back to 0, increase i, unless i =2 then we are done
			if i ==2:
				#the board is full, recollect results and return
				state = env.get_state()
				ended = env.game_over(force_recalculate=True)
				winner = env.winner
				results.append((state,winner,ended))
			else:
				results += get_state_hash_and_winner(env, i+1, 0)
		else:
			results += get_state_hash_and_winner(env,i,j+1)
	return results

def initialV_x(env, state_winner_triples):
	'''
	initialize state values as follows
	if x wins, V(s) = 1
	if x loses or draws, V(s) = 0
	otherwirse V(s) = 0.5
	'''
	V = np.zeros(env.num_states)
	for state, winner, ended in state_winner_triples:
		if ended:
			if winner == env.x:
				v = 1
			else:
				v = 0 
		else:
			v = 0.5
		V[state] = v
	return V

def initalV_o(env,state_winner_triples):
	'''
	This is (almost) the opposite of inital V for player x since
	everywhere x wins (1) , o loses (0)
	but a draw is still a 0 for o
	'''
	V = np.zeros(env.num_states)
	for state, winner, ended in state_winner_triples:
		if ended:
			if winner == env.o:
				v = 1
			else:
				v = 0
		else:
			v = 0.5
		V[state] = v
	return V

def play_game(p1,p2,env,draw=False):
	#put draw =1 to draw map for first player and 2 for second, o/w None
	#loops until the game is over
	current_player = None
	while not env.game_over():
		#alternate players, p1 starts
		if current_player == p1:
			current_player = p2
		else:
			current_player = p1

		#draw the board before the user who wants to see it makes the move
		if draw:
			if draw == 1 and current_player == p1:
				env.draw_board()
			if draw == 2 and current_player == p2:
				env.draw_board()

		#current player makes a move
		current_player.take_action(env)

		#update state histories
		state = env.get_state()
		p1.update_state_history(state)
		p2.update_state_history(state)

	if draw:
		env.draw_board()

	#do the value function update
	p1.update(env)
	p2.update(env)



if __name__ == '__main__':
	#train the agent
	p1 = Agent()
	p2 = Agent()
	print('Agents created')

	#set inital V for p1 and p2
	env = Environment()
	print('environment created')
	state_winner_triples = get_state_hash_and_winner(env)
	print('state hashes created')

	Vx = initialV_x(env, state_winner_triples)
	p1.setV(Vx)
	Vo = initalV_o(env,state_winner_triples)
	p2.setV(Vo)


	#give each player their symbol
	p1.set_symbol(env.x)
	p2.set_symbol(env.o)

	print("Training Started")
	#Train the agents against eachother
	T = 10000
	for t in range(T):
		if t % 200 == 0:
			print(t)
		play_game(p1,p2,Environment())


	#now time for human to play against agent
	human = Human()
	human.set_symbol(env.o)
	while True:
		p1.set_verbose(True)
		#can swap p1 and human for who goes first
		play_game(p1,human,Environment(),draw=2)

		answer = input("play again? [Y/n]: ")
		if answer.lower()[0] == 'n':
			break


























