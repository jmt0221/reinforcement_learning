import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

class FeatureTransformer:
    def __init__(self, env, n_components=500):
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # Used to converte a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        featurizer = FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
                ])
        featurizer.fit_transform(scaler.transform(observation_examples))

        self.scaler = scaler
        self.featurizer = featurizer


    def transform(self, observations):
        # print "observations:", observations
        scaled = self.scaler.transform(observations)
        # assert(len(scaled.shape) == 2)
        return self.featurizer.transform(scaled)


class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()]), [0])
            self.models.append(model)

    def predict(self,s):
        X = self.feature_transformer.transform([s])
        assert (len(X.shape) == 2)
        return np.array([mod.predict(X)[0] for mod in self.models])

    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        #we dont technically need to do epsilon greedy because of OIV
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one(model, env,eps, gamma):
    '''
    :return: list of states_rewards and total rewards
    '''
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        #update model
        G = reward + gamma*np.max(model.predict(observation)[0])
        model.update(prev_observation, action, G)

        totalreward += reward
        iters += 1

    return totalreward

def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    #both X and Y will be the shape of (num_tiles,num_tiles)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X,Y]))


    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title('Cost-To-Go Function')
    fig.colorbar(surf)
    plt.show()

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for i in range(N):
        running_avg[i] = totalrewards[max(0,i-100):(i+1)].mean()
    plt.plot(running_avg)
    plt.title('Running Average')
    plt.show()

def main():
    env = gym.make('MountainCar-v0')
    print('Env Created')
    ft = FeatureTransformer(env)
    model = Model(env, ft, 'constant')
    gamma = 0.99

    N = 300
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        totalreward = play_one(model, env, eps, gamma)
        totalrewards[n] = totalreward
        print('episode:', n, 'total reward: ', totalreward)
    print('avg reward for the last 100 episodes: ', totalrewards[-100:].mean())
    print('total steps: ', -totalrewards.sum())

    plt.plot(totalrewards)
    plt.title('Rewards')
    plt.show()

    plot_running_avg(totalrewards)

    plot_cost_to_go(env,model)

if __name__ == '__main__':
    main()

