import os
from gym import wrappers
import sys
import tarfile
import pickle


class lab(object):

    def __init__(self, name, env, agent, num_episodes=1000):

        self.name = name
        self.agent = agent
        self.num_episodes = num_episodes

        self.agentname = self.agent.name
        self.monitordir = './monitor/' + self.name + '_' + self.agentname
        self.resultdir = './result/' + self.name + '_' + self.agentname
        os.makedirs(self.resultdir)
        os.makedirs(self.monitordir)
        self.env = wrappers.Monitor(env, self.monitordir)
        # self.env = env

    def run(self, testmode=False, name='0'):

        losses =[]
        numstepses = []
        totrewards = []
        for idx in xrange(self.num_episodes):
            obs = self.env.reset()
            done = False
            loss = 0.0
            numsteps = 0
            totreward = 0.0

            while not done:
                numsteps += 1
                action = self.agent.act(obs, testmode)
                old_obs = obs

                obs, reward, done, _ = self.env.step(action)

                # print obs
                # print reward
                # print done
                # print action
                # raw_input("Countinue..")

                totreward += reward

                if not testmode:
                    self.agent.get_instance(old_obs, action, reward, obs, done)
                    loss += self.agent.train_on_batch()
            print "Episode {}: loss is {}, total reward is {}, and number of steps is {}".format(idx, loss, totreward, numsteps)
            print "epsilon is {}, total count is {}".format(self.agent.epsilon, self.agent.counter)
            losses.append(loss)
            numstepses.append(numsteps)
            totrewards.append(totreward)

        result = {'loss': losses, 'steps':numstepses, 'reward': totrewards}
        result_file = self.resultdir + '/' + name + '.pickle'
        self._pkl_result(result, result_file)

    def _pkl_result(self, result, filename):
        pickle_file = filename
        try:
            pickleData = open(pickle_file, 'wb')
            pickle.dump(result, pickleData, pickle.HIGHEST_PROTOCOL)
            pickleData.close()
            print 'result pickled'
        except Exception as e:
            print 'Unable to save data to', pickle_file, ':', e
            raise
