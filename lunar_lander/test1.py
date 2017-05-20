import random
import numpy as np

from keras.optimizers import Adam
from keras import backend as K

import gym

from qmemory import qmem
from qagents import agentQNet
from experiment import lab

seed = 16


env = gym.make('LunarLander-v2')

np.random.seed(seed)
random.seed(seed)
env.seed(seed)


net_param = {'hidden_layers': [ (50, 'relu'), (40, 'relu')],
             'loss': 'mse','optimizer': Adam(lr=0.0005)}

lander = agentQNet(name='DQN_0', dim_state=env.observation_space.shape[0], n_action=env.action_space.n, epsilon=0.3, epsdecay=0.99,
                   memsize=300000, batch_size=32, size2train=1000, gamma=0.99, update_freq=600, net_params=net_param)

lab_lander = lab(name='train', env=env, agent=lander,num_episodes=200)
lab_lander.run()

env2 = gym.make('LunarLander-v2')
env2.seed(seed)
lab_lander = lab(name='test', env=env2, agent=lander,num_episodes=100)
lab_lander.run(testmode=True)