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

def hubert_loss(y_true, y_pred):
    err = y_pred - y_true
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )

def epsdecay(epsilon):
    if epsilon>0.13:
        return epsilon*0.98
    if epsilon<0.09:
        return epsilon*0.95
    return epsilon*0.999


net_param = {'hidden_layers': [ (50, 'relu'), (40, 'relu')],
             'loss': hubert_loss,'optimizer': Adam(lr=0.0005)}

lander = agentQNet(name='DQN_3', dim_state=env.observation_space.shape[0], n_action=env.action_space.n, epsdecay=0.98, minepsilon=0.0, epdecayfunc=epsdecay,
                   memsize=300000, batch_size=32, size2train=1000, gamma=0.99, update_freq=600, priority_sampling=True, net_params=net_param)

lab_lander = lab(name='train', env=env, agent=lander,num_episodes=600)
lab_lander.run()

env2 = gym.make('LunarLander-v2')
env2.seed(seed)
lab_lander = lab(name='test', env=env2, agent=lander,num_episodes=100)
lab_lander.run(testmode=True)