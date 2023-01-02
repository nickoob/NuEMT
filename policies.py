import numpy as np
from filter import get_filter


class Policy(object):

    def __init__(self, policy_params):
        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        # self.ac_high = policy_params['ac_high']
        # self.ac_low = policy_params['ac_low']
        self.weights = np.empty(0)
        self.h1_dim = 0
        self.h2_dim = 0
        self.fitness = 0.
        self.steps = 0
        self.observation_filter = get_filter(policy_params['ob_filter'], shape=(self.ob_dim,))
        self.update_filter = True

    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def get_observation_filter(self):
        return self.observation_filter


class NonLinearPolicy(Policy):

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.h1_dim = policy_params['h1_dim']
        self.h2_dim = policy_params['h2_dim']
        # self.weights = np.zeros((self.ob_dim + 1) *self.ac_dim, dtype=np.float64)
        self.weights = np.zeros((self.ob_dim + 1) * self.h1_dim + self.h1_dim * self.h2_dim + self.h2_dim * self.ac_dim,
                                dtype=np.float64)
        # self.weights = np.zeros((self.ob_dim + 1) * self.h1_dim + self.h1_dim * self.ac_dim, dtype=np.float64)

    def act(self, ob):
        # 2 hidden layers feedforward neural network
        fc1 = self.weights[:(self.ob_dim+1) * self.h1_dim].reshape(self.h1_dim, self.ob_dim+1)
        fc2 = self.weights[(self.ob_dim+1) * self.h1_dim: (self.ob_dim+1) * self.h1_dim + self.h1_dim * self.h2_dim].reshape(self.h2_dim, self.h1_dim)
        fc3 = self.weights[(self.ob_dim+1) * self.h1_dim + self.h1_dim * self.h2_dim:].reshape(self.ac_dim, self.h2_dim)

        ob = ob.flatten()
        ob = self.observation_filter(ob, update=self.update_filter)
        ob = np.append(ob, [1.0])
        x = np.tanh(np.matmul(fc1, ob))
        x = np.tanh(np.matmul(fc2, x))
        x = np.tanh(np.matmul(fc3, x))

        return x

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux

    def __str__(self):
        return 'Non-Linear Policy'
