import numpy as np
import matplotlib.pyplot as plt
from comparing_epsilons import Bandit
from optimistic_initial_value_epsilon import run_experiment as run_experiment_oiv
from UCB1 import run_experiment as run_experiment_ucb

class BayesianBandit:
	def __init__(self,m):
		self.m = m
		#parameters for mu - prior is N(0,1)
		self.m0 = 0
		self.lambda0 = 1
		self.sum_x = 0 # for convenience
		self.tau = 1

	def pull(self):
		return np.random.randn() + self.m

	def sample(self):
		return np.random.randn() / np.sqrt(self.lambda0) + self.m0

	def update(self,x):
		#assume tau is 1
		self.lambda0 +=1
		self.sum_x += x
		self.m0 = self.tau * self.sum_x / self.lambda0


def run_experiment_decaying_epsilon(m1,m2,m3,N):
	bandits = [Bandit(m1),Bandit(m2), Bandit(m3)]
	data = np.empty(N)

	for i in range(N):
		p = np.random.random()
		if p < 1.0/(i+1):
			j = np.random.choice(3)
		else:
			j = np.argmax([b.mean for b in bandits])
		x = bandits[j].pull()
		bandits[j].update(x)

		data[i] = x
	cumulative_average = np.cumsum(data) / (np.arange(N) + 1)


	#plot moving average
	plt.plot(cumulative_average)
	plt.plot(np.ones(N)*m1)
	plt.plot(np.ones(N)*m2)
	plt.plot(np.ones(N)*m3)
	plt.xscale('log')
	plt.show()

	return cumulative_average

if __name__ == '__main__':
	eps = run_experiment_decaying_epsilon(1.0,2.0,3.0,100000)
	oiv = run_experiment_oiv(1.0,2.0,3.0,0.0,100000)
	ucb = run_experiment_ucb(1.0,2.0,3.0,100000)
	bayes = run_experiment_oiv(1.0,2.0,3.0,0,100000)


	#log scale plots
	plt.plot(eps, label = 'decaying epsilon')
	plt.plot(oiv, label = 'optimistic')
	plt.plot(ucb, label = 'UCB')
	plt.plot(bayes, label = 'Bayes')
	plt.legend()
	plt.xscale('log')
	plt.show()

	#linear scale plots
	plt.plot(eps, label = 'decaying epsilon')
	plt.plot(oiv, label = 'optimistic')
	plt.plot(ucb, label = 'UCB')
	plt.plot(bayes, label = 'Bayes')
	plt.legend()
	plt.show()



