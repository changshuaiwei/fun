import numpy as np
import random

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

from qmemory import qmem

class agentQNet(object):
    def __init__(self, name, dim_state, n_action, gamma=0.99, epsilon=1.0, epsdecay = 0.98, minepsilon=0.0, epdecayfunc=None,
                 memsize=10000, batch_size=20, size2train=1000, update_freq=600, r_shaping=False, priority_sampling=False, net_params=None):
        self.name = name
        self.dim_state = dim_state
        self.n_action= n_action
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsdecay = epsdecay
        self.memsize = memsize
        self.batch_size = batch_size
        self.size2train = size2train
        self.update_freq = update_freq
        self.D = qmem(maxsize=memsize)
        self.r_shaping= r_shaping
        self.priority_sampling= priority_sampling
        self.minepsilon=minepsilon
        self.epdecayfunc = epdecayfunc

        #set network hyperparameter
        self.net_params = None
        if net_params is None:
            self.net_params = {'hidden_layers': [ (40, 'relu'), (40, 'relu') ],
                               'loss': 'mse',
                               'optimizer': Adam(lr=0.00025)}
        else:
            self.net_params = net_params

        # print self.net_params
        # raw_input("Countinue...")

        #set two networks
        self.model = self._build_model()
        self.target_model = self._build_model()

        self.training_started = False
        self.counter = 0

    def _build_model(self):

        model = Sequential()
        idx = 0
        for params in self.net_params['hidden_layers']:
            units, activation_name = params[0], params[1]
            if idx == 0:
                model.add(Dense(units, input_dim=self.dim_state, activation= activation_name))
            else:
                model.add(Dense(units, activation= activation_name))
            idx +=1

        model.add(Dense(self.n_action, activation='linear'))
        model.compile(loss=self.net_params['loss'], optimizer=self.net_params['optimizer'])
        return model

    def _update_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_instance(self, state, action, reward, new_state, done):
        state = np.array(state)
        new_state = np.array(new_state)

        if self.r_shaping:
            reward = self.reward_shaping(state, reward, new_state)

        self.D.add_instance(state, action, reward, new_state, done)
        self.counter +=1

        if done and self.training_started:
            if self.epdecayfunc is None:
                self.epsilon *= self.epsdecay
            else:
                self.epsilon = self.epdecayfunc(self.epsilon)

            if self.epsilon < self.minepsilon:
                self.epsilon = self.minepsilon
        if self.counter % self.update_freq == 0:
            self._update_model()

    def reward_shaping(self, state, reward, new_state):
        dist = np.sqrt(np.sum(state[0:2]*state[0:2]))
        new_dist = np.sqrt(np.sum(new_state[0:2]*new_state[0:2]))
        #new_reward = reward + 1.0*(dist > new_dist) - 1.0*(new_dist >dist)
        #new_reward = reward + 1.0/(dist+0.01) - 1.0/(new_dist+0.01)
        new_reward = reward + 2.0*(dist - new_dist)
        return new_reward


    def act(self, state, testmode=False):
        state = np.array(state)
        if testmode:
            score = self.model.predict(state[np.newaxis, :]).flatten()
            return np.argmax(score)
        if (not self.training_started):
            return random.randrange(self.n_action)
        if random.random() < self.epsilon:
            return random.randrange(self.n_action)
        score = self.model.predict(state[np.newaxis, :]).flatten()
        # print score
        # raw_input("Continue..")
        return np.argmax(score)

    def _sample_X_y(self):
        state, action, reward, new_state, done = self.D.sampling(self.batch_size, importance=self.priority_sampling)

        Qhat = self.model.predict(state)
        # print Qhat
        new_Qhat = self.target_model.predict(new_state)
        # print new_Qhat
        Qhat[range(Qhat.shape[0]), action.astype(int)] = np.max(new_Qhat,axis=1)*self.gamma*(1 - done) + reward
        # print Qhat
        # raw_input("Continue...")

        X, y = state, Qhat
        return X, y

    def train_on_batch(self):
        if self.counter < self.size2train:
            return 0.0
        if not self.training_started:
            self.training_started = True
        X, y = self._sample_X_y()
        rst = self.model.fit(X, y, verbose=False, batch_size=self.batch_size, nb_epoch=1)
        return rst.history['loss'][-1]







