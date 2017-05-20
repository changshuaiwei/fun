import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import OrderedDict
import numpy as np
import pandas as pd

class LearningAgent_v2(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent_v2, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.trip_history = []
        self.debug = True
        self.gamma = 0.2 #upper bound of discount 
        #self.alpha = 0.5 #uppder bound of learning rate
        self.epsilon = 0.1 #lower bound of proportion of random steps
        self.reg = 0.001 # regularization param for regression
        self.lr = 0.1 # learning rate for regression
        self.clock_update = 0 # store number of updates
        self.init_params_scale = 1e-4 # scale of initial params setting
        
        self.max_memory = 400 # number of rows for state_action_experience
        self.batch_size = 20 # size of batch
        self.batch_step = 20 # extract a batch for each batch_step steps
        self.param_step = self.max_memory # how many step should update w
        
        self.state_feature = ['right_no', 'forward_no', 'left_no', 'next_right', 'next_forward', 'next_left']
        self.action_feature = ['right', 'forward', 'left']
        
        self.state = None
        
        self.num_action_space = np.concatenate( ( np.diag(np.ones(3)), np.zeros(3)[np.newaxis,:]))
        
        self.state_action_feature = self.state_feature + self.action_feature + [x + "_action_" + y for x in self.state_feature for y in self.action_feature]
        #self.state_action_df = pd.DataFrame(columns = (self.state_action_feature + ['Q_score']) )
        self.state_action_experience = np.zeros( (1, len(self.state_action_feature)) )
        self.Q_score_experience = np.zeros(1)
        
        self.ex_state = np.zeros( len(self.state_feature) )
        self.ex_state_action = np.zeros( len(self.state_action_feature) )
        self.ex_reward = 0
        
        self.params = {'b': np.random.randn(1), 'w': self.init_params_scale * np.random.randn(len(self.state_action_feature),1)}
        
        self.params_update = self.params
        
        self.reward_history = np.zeros(1)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.ex_reward = 0
        self.ex_state_action = np.zeros( len(self.state_action_feature) )
        self.ex_state = np.zeros( len(self.state_feature) )
        
        if(len(self.trip_history) <150 ) :
            print 'Current success rate is {}'.format( sum(self.trip_history)/(len(self.trip_history) + 0.000001) )
        else:
            print 'Success rate for recent 100 trials is {}'.format(sum(self.trip_history[-100:])/(len(self.trip_history[-100:]) + 0.000001))
        print 'Average reward for recent moves is {}'.format(np.mean(self.reward_history))
        if(self.reward_history.shape[0] > 1000) :
            self.reward_history = np.delete(self.reward_history, range(100))
            
        
            
    def numeric_state(self, inputs=None, deadline=0, next_waypoint=None):
        #print 'inputs is {}, deadline is {}, next_waypoint is {}'.format(str(inputs), str(deadline), str(next_waypoint))
        col_name = self.state_feature
        state = np.zeros(len(col_name))
        state += np.array( map(lambda x: x=='next_' + str(next_waypoint), col_name) )

        if inputs['light'] == 'red' and inputs['left'] == 'forward':
            #state += np.array( map(lambda x: x=='right_no', col_name) )
            state[0] = 1    
        if inputs['light'] == 'red': 
            state[1] = 1    
        if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
            state[2] = 1
        #state[len(col_name)-1] = deadline
        
        if False:
            print 'inputs is {}, deadline is {}, next_waypoint is {}\n'.format(str(inputs), str(deadline), str(next_waypoint))
            print zip(col_name,state)
            raw_input("Press Enter to continue...")
            
        return state
        
    def numeric_action(self, action=None):
        col_name = self.action_feature
        return np.array( map(lambda x: x==str(action), col_name) )
        
    def numeric_state_action(self, num_state=None , num_action=None ):
        return np.concatenate( (num_state, num_action, np.outer(num_state,num_action).flatten() ), axis = 0)
        
    def max_Q_param(self, num_state):
        X = np.apply_along_axis(lambda x: self.numeric_state_action(num_state, x), axis = 1, arr=self.num_action_space)
        score = X.dot(self.params['w']) + self.params['b']
        
        if False:
            print '\nX are\n {}\n, Params are\n {}\n'.format(str(X), str(self.params))
            raw_input("Press Enter to continue...")
        
        choose = np.argmax(score)
        opt_action = None
        if choose<3:
            opt_action = self.action_feature[choose]
        num_state_action = X[choose]
        max_Q_hat = score[choose]
        
        if False:
            print '\nScores are\n {}\n, opt action are\n {}\n'.format(str(score), str(opt_action))
            raw_input("Press Enter to continue...")
        
        return opt_action, max_Q_hat, num_state_action
        
    def gradient(self, X, y, reg=0.01):
        if False:
            print '\nX are\n {}\n and y are\n {}\n'.format(str(X), str(y))
            raw_input("Press Enter to continue...")
            
        w, b = self.params_update['w'], self.params_update['b']
        scores = X.dot(w) + b
        y = y.flatten()[:,np.newaxis]
        loss = np.mean((y-scores)**2) + 0.5 * reg * np.sum(w**2)

        if False:
            print '\ny are\n {}\n and scores are\n {}\n and loss is\n {}\n'.format(str(y), str(scores), str(loss) )
            raw_input("Press Enter to continue...")        
        
        d_w = np.mean((X*((scores-y)*2)),axis=0)[:,np.newaxis] + reg * w
        d_b = np.mean((scores-y)*2)
        
        return d_w, d_b, loss
        
    def sample_X_y(self, size=10):
        idx = np.random.randint(self.state_action_experience.shape[0],size=size)
        X = self.state_action_experience[idx ,:]
        y = self.Q_score_experience[idx]
        
        return X, y
    
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        #may need to take deadline into account?

        # TODO: Update state
        self.state_action_experience = np.concatenate( (self.state_action_experience , self.ex_state_action[np.newaxis,:]) )
        num_state = self.numeric_state(inputs=inputs, deadline=deadline, next_waypoint=self.next_waypoint)
        
        self.state = zip(self.state_feature, num_state)
        
        # TODO: Select action according to your policy
        
        
        action, max_Q_hat, num_state_action = self.max_Q_param(num_state)
        
        if(random.uniform(0,1) < self.epsilon):
            action = random.choice(Environment.valid_actions[:])
            num_action = self.numeric_action(action)
            num_state_action = self.numeric_state_action(num_state=num_state, num_action=num_action)
            if False:
                print "\n Use a random action, {}".format(str(action) )
                #debug
                raw_input("Press Enter to continue...")
            
        
        true_Q_score = self.ex_reward + self.gamma * max_Q_hat
        self.Q_score_experience = np.append(self.Q_score_experience, true_Q_score)
        self.clock_update += 1
        
        if False:
            print '\nShape of State Action expreience Matrix is {}\n'.format(self.state_action_experience.shape)
            print '\nShape of Q score experience is {}\n'.format(self.Q_score_experience.shape)
            raw_input("Press Enter to continue...")
            

        # TODO: Learn policy based on state, action, reward
        reward = self.env.act(self, action)
        
        self.ex_reward = reward
        self.ex_state_action = num_state_action
        self.reward_history = np.append(self.reward_history, reward)

        if reward>9:
            self.trip_history.append(1)
            #need to write down something here
            self.state_action_experience = np.concatenate( (self.state_action_experience , self.ex_state_action[np.newaxis,:]) )
            self.Q_score_experience = np.append(self.Q_score_experience, reward)
            self.clock_update += 1
        elif deadline == 0:
            self.trip_history.append(0)
            self.state_action_experience = np.concatenate( (self.state_action_experience , self.ex_state_action[np.newaxis,:]) )
            self.Q_score_experience = np.append(self.Q_score_experience, reward)
            self.clock_update += 1

        if(self.clock_update > self.max_memory + 2):
            self.state_action_experience = np.delete(self.state_action_experience, range(self.state_action_experience.shape[0] - self.max_memory), 0 )
            self.Q_score_experience = np.delete(self.Q_score_experience, range(len(self.Q_score_experience) - self.max_memory ) )
            
            if False:
                print '\nShape of State Action expreience Matrix is {}\n'.format(self.state_action_experience.shape)
                print '\nShape of Q score experience is {}\n'.format(self.Q_score_experience.shape)
                raw_input("Press Enter to continue...")
            
            if(self.clock_update % self.batch_step == 0 ):
                for i in xrange(2):
                    if False:
                        print '\nUpdated Parameters are {}\n'.format(str(self.params_update))
                        raw_input("Press Enter to continue...")
                    data_X, data_y = self.sample_X_y( size = self.batch_size )
                    d_w, d_b, loss = self.gradient(data_X, data_y, reg=self.reg)
                    if False:
                        print '\nGradiants are {} and {}\n'.format(str(d_w),str(d_b))
                        raw_input("Press Enter to continue...")
                        
                    if False:
                        print '\nloss is {}\n'.format(loss)
                        raw_input("Press Enter to continue...")
                        
                    self.params_update['w'] = self.params_update['w'] - self.lr * d_w
                    self.params_update['b'] = self.params_update['b'] - self.lr * d_b
                
            if self.clock_update % self.param_step == 0:
                self.params = self.params_update
                if True:
                    print '\nBias for regression is {}\n'.format(str(self.params['b']))
                    weight_df = pd.DataFrame(data=self.params['w'].T, columns = self.state_action_feature)
                    print '\nWeights for regression is\n{}\n'.format(weight_df.T)
                    #raw_input("Press Enter to continue...")

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]



class LearningAgent_v1(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent_v1, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.gamma = 0.2 #upper bound of discount 
        self.alpha = 0.1 #uppder bound of learning rate
        self.epsilon = 0.2 #lower bound of proportion of random steps
        #self.state = {'deadline': None, 'forward_ok': True, 'left_ok': True, 'right_ok': True, 'next_waypoint': None }
        self.state = {'forward_ok': True, 'left_ok': True, 'right_ok': True, 'next_waypoint': None }
        self.exreward = 0
        #self.exstate = {'deadline': None, 'forward_ok': True, 'left_ok': True, 'right_ok': True, 'next_waypoint': None }
        self.exstate = {'forward_ok': True, 'left_ok': True, 'right_ok': True, 'next_waypoint': None }
        self.exaction = None
        self.debug = False
        self.trip_history = []
        
        self.Q = OrderedDict()
        
        self.reward_history = np.zeros(1)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = {'forward_ok': True, 'left_ok': True, 'right_ok': True, 'next_waypoint': None }
        self.exreward = 0
        self.exstate = {'forward_ok': True, 'left_ok': True, 'right_ok': True, 'next_waypoint': None }
        self.exaction = None
        if(len(self.trip_history) <150 ) :
            print 'Current success rate is {}'.format( sum(self.trip_history)/(len(self.trip_history) + 0.000001) )
        else:
            print 'Success rate for recent 100 trials is {}'.format(sum(self.trip_history[-100:])/(len(self.trip_history[-100:]) + 0.000001))
        print 'Average reward for recent moves is {}'.format(np.mean(self.reward_history))
        if(self.reward_history.shape[0] > 1000) :
            self.reward_history = np.delete(self.reward_history, range(100))
            
        #print str(self.Q)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        #may need to take deadline into account?

        # TODO: Update state
        #self.state = {'inputs': inputs, 'deadline': deadline, 'next_waypoint':self.next_waypoint}
        #self.state = {'inputs': inputs, 'next_waypoint':self.next_waypoint}

        #epsilon = self.epsilon + (1-self.epsilon)/(t+1)*5
        #gamma = ( 1- 10/(t+10) ) * self.gamma
        #alpha = self.alpha/(t+1.0)
        gamma = self.gamma
        epsilon = self.epsilon
        alpha = self.alpha
        
        self.state['next_waypoint'] = self.next_waypoint
        #self.state['deadline'] = int(deadline>5) + int(deadline>25)

        self.state['right_ok'] = True
        if inputs['light'] == 'red' and inputs['left'] == 'forward':
            self.state['right_ok'] = False
            
        if inputs['light'] == 'red': 
            self.state['forward_ok']=False 
        else: 
            self.state['forward_ok']=True
            
        if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
            self.state['left_ok']=False
        else:
            self.state['left_ok']=True
        
        
        
        # TODO: Select action according to your policy
        #action = random.choice(Environment.valid_actions[1:])
        
        newkey = str(self.exstate.values()) + ':' + str(self.exaction)
        
        if(self.debug):
            print "\n New key is {}".format(newkey)
            #debug
            raw_input("Press Enter to continue...")
        
        tmp_Q = dict([ (x, self.Q[x]) for x in self.Q.keys() if str(self.state.values()) in x])
        
        #print tmp_Q
        
        if self.debug:
            print "\n Q value for future state is {}".format(str(tmp_Q))
            #debug
            raw_input("Press Enter to continue...")
        
        action = random.choice(Environment.valid_actions[:])
        tmp_max_Q = 0
        if( len(tmp_Q) == 0 ):
            tmp_max_Q = 0
            action = random.choice(Environment.valid_actions[:])
        else:
            #tmp_idx = max(tmp_Q)
            tmp_idx = max(tmp_Q.iterkeys(), key=(lambda key: tmp_Q[key]))
            tmp_max_Q = tmp_Q[tmp_idx]
            if( tmp_max_Q>0 or len(tmp_Q)==4 ):
                #print tmp_idx
                tmp_Q_split = tmp_idx.split(':')
                #print tmp_Q_split
                #print tmp_Q_split
                action = tmp_Q_split[1]
                if action=='None' :
                    action = None
            else:
               exist_actions = [x.split(':')[1] for x in tmp_Q.keys() ]
               all_actions = ['None', 'forward', 'left', 'right']
               remaining_actions = [x for x in all_actions if not (x in exist_actions)]
               if self.debug:
                   print "Remaining actions are {}".format(str(remaining_actions))
               action = random.choice(remaining_actions)
               tmp_max_Q = 0
               if action=='None' :
                    action = None
                    
        if self.debug:
            print "\n future optimum action is {}".format(str(action))
            #debug
            raw_input("Press Enter to continue...")
        
        if(random.uniform(0,1) < epsilon):
            action = random.choice(Environment.valid_actions[:])
            if self.debug:
                print "\n Instead use a random action, {}".format(str(action) )
                #debug
                raw_input("Press Enter to continue...")
            
        #print 'now ' + str(action)
        #random guess have success rate about ~0.20
        #action = random.choice(Environment.valid_actions[:])
           
        newval = self.exreward + gamma * tmp_max_Q
        
        if self.debug:
            print "\n current reward is {0}, gamma is {1}, and estimated max future Q is {2}".format(self.exreward, gamma, tmp_max_Q)
            #debug
            raw_input("Press Enter to continue...")
        
        if newkey in self.Q.keys():
            self.Q[newkey] = self.Q[newkey] * (1-alpha) + alpha * newval
        else:
            self.Q[newkey] = self.alpha * newval
        
        if self.debug:
            print "updated Q values {}".format(str(self.Q))
            #debug
            raw_input("Press Enter to continue...")

        #print t
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        if reward>9:
            self.trip_history.append(1)
            #need to write down something here
            newkey = str(self.state.values()) + ':' + str(action)
            newval = reward # + deadline
            if newkey in self.Q.keys():
                self.Q[newkey] = self.Q[newkey] * (1-alpha) + alpha * newval
            else:
                self.Q[newkey] = self.alpha * newval
        elif deadline == 0:
            self.trip_history.append(0)
            newkey = str(self.state.values()) + ':' + str(action)
            newval = reward
            if newkey in self.Q.keys():
                self.Q[newkey] = self.Q[newkey] * (1-alpha) + alpha * newval
            else:
                self.Q[newkey] = self.alpha * newval

        # TODO: Learn policy based on state, action, reward
        self.exreward = reward
        self.exstate = self.state
        self.exaction = action
        self.reward_history = np.append(self.reward_history, reward)
        
        #print "number of parameter is {0}, sum of Qfunction is {1}".format( len(self.Q.keys()), sum(self.Q.values()) )

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


class LearningAgent_v0(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent_v0, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.gamma = 0.2 #upper bound of discount 
        self.alpha = 0.1 #uppder bound of learning rate
        self.epsilon = 0.2 #lower bound of proportion of random steps
        #self.state = {'deadline': None, 'forward_ok': True, 'left_ok': True, 'right_ok': True, 'next_waypoint': None }
        self.state = {'light': 'green', 'oncoming': None, 'left': None, 'right': None, 'next_waypoint': None }
        self.exreward = 0
        #self.exstate = {'deadline': None, 'forward_ok': True, 'left_ok': True, 'right_ok': True, 'next_waypoint': None }
        self.exstate = {'light': 'green', 'oncoming': None, 'left': None, 'right': None, 'next_waypoint': None }
        self.exaction = None
        self.debug = False
        self.trip_history = []
        
        self.reward_history = np.zeros(1)
        
        self.Q = OrderedDict()

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = {'light': 'green', 'oncoming': None, 'left': None, 'right': None, 'next_waypoint': None }
        self.exreward = 0
        self.exstate = {'light': 'green', 'oncoming': None, 'left': None, 'right': None, 'next_waypoint': None }
        self.exaction = None
        
        if(len(self.trip_history) <150 ) :
            print 'Current success rate is {}'.format( sum(self.trip_history)/(len(self.trip_history) + 0.000001) )
        else:
            print 'Success rate for recent 100 trials is {}'.format(sum(self.trip_history[-100:])/(len(self.trip_history[-100:]) + 0.000001))
        print 'Average reward for recent moves is {}'.format(np.mean(self.reward_history))
        if(self.reward_history.shape[0] > 1000) :
            self.reward_history = np.delete(self.reward_history, range(100))
            
        #print "number of parameter is {0}, sum of Qfunction is {1}".format( len(self.Q.keys()), sum(self.Q.values()) )


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        #may need to take deadline into account?

        # TODO: Update state
        #self.state = {'inputs': inputs, 'deadline': deadline, 'next_waypoint':self.next_waypoint}
        #self.state = {'inputs': inputs, 'next_waypoint':self.next_waypoint}

        #epsilon = self.epsilon + (1-self.epsilon)/(t+1)*5
        #gamma = ( 1- 10/(t+10) ) * self.gamma
        #alpha = self.alpha/(t+1.0)
        gamma = self.gamma
        epsilon = self.epsilon
        alpha = self.alpha
        
        self.state['next_waypoint'] = self.next_waypoint
        #self.state['deadline'] = int(deadline>5) + int(deadline>25)

        for k in inputs.keys():
            self.state[k] = inputs[k]
        
        # TODO: Select action according to your policy
        #action = random.choice(Environment.valid_actions[1:])
        
        newkey = str(self.exstate.values()) + ':' + str(self.exaction)
        
        if(self.debug):
            print "\n New key is {}".format(newkey)
            #debug
            raw_input("Press Enter to continue...")
        
        tmp_Q = dict([ (x, self.Q[x]) for x in self.Q.keys() if str(self.state.values()) in x])
        
        #print tmp_Q
        
        if self.debug:
            print "\n Q value for future state is {}".format(str(tmp_Q))
            #debug
            raw_input("Press Enter to continue...")
        
        action = random.choice(Environment.valid_actions[:])
        tmp_max_Q = 0
        if( len(tmp_Q) == 0 ):
            tmp_max_Q = 0
            action = random.choice(Environment.valid_actions[:])
        else:
            #tmp_idx = max(tmp_Q)
            tmp_idx = max(tmp_Q.iterkeys(), key=(lambda key: tmp_Q[key]))
            tmp_max_Q = tmp_Q[tmp_idx]
            if( tmp_max_Q>0 or len(tmp_Q)==4 ):
                #print tmp_idx
                tmp_Q_split = tmp_idx.split(':')
                #print tmp_Q_split
                #print tmp_Q_split
                action = tmp_Q_split[1]
                if action=='None' :
                    action = None
            else:
               exist_actions = [x.split(':')[1] for x in tmp_Q.keys() ]
               all_actions = ['None', 'forward', 'left', 'right']
               remaining_actions = [x for x in all_actions if not (x in exist_actions)]
               if self.debug:
                   print "Remaining actions are {}".format(str(remaining_actions))
               action = random.choice(remaining_actions)
               tmp_max_Q = 0
               if action=='None' :
                    action = None
                    
        if self.debug:
            print "\n future optimum action is {}".format(str(action))
            #debug
            raw_input("Press Enter to continue...")
        
        if(random.uniform(0,1) < epsilon):
            action = random.choice(Environment.valid_actions[:])
            if self.debug:
                print "\n Instead use a random action, {}".format(str(action) )
                #debug
                raw_input("Press Enter to continue...")
            
        #print 'now ' + str(action)
        #random guess have success rate about ~0.20
        #action = random.choice(Environment.valid_actions[:])
           
        newval = self.exreward + gamma * tmp_max_Q
        
        if self.debug:
            print "\n current reward is {0}, gamma is {1}, and estimated max future Q is {2}".format(self.exreward, gamma, tmp_max_Q)
            #debug
            raw_input("Press Enter to continue...")
        
        if newkey in self.Q.keys():
            self.Q[newkey] = self.Q[newkey] * (1-alpha) + alpha * newval
        else:
            self.Q[newkey] = self.alpha * newval
        
        if self.debug:
            print "updated Q values {}".format(str(self.Q))
            #debug
            raw_input("Press Enter to continue...")

        #print t
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        if reward>9:
            self.trip_history.append(1)
            #need to write down something here
            newkey = str(self.state.values()) + ':' + str(action)
            newval = reward # + deadline
            if newkey in self.Q.keys():
                self.Q[newkey] = self.Q[newkey] * (1-alpha) + alpha * newval
            else:
                self.Q[newkey] = self.alpha * newval
        elif deadline == 0:
            self.trip_history.append(0)
            newkey = str(self.state.values()) + ':' + str(action)
            newval = reward
            if newkey in self.Q.keys():
                self.Q[newkey] = self.Q[newkey] * (1-alpha) + alpha * newval
            else:
                self.Q[newkey] = self.alpha * newval

        # TODO: Learn policy based on state, action, reward
        self.exreward = reward
        self.exstate = self.state
        self.exaction = action
        self.reward_history = np.append(self.reward_history, reward)
        
        #print "number of parameter is {0}, sum of Qfunction is {1}".format( len(self.Q.keys()), sum(self.Q.values()) )

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]



def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent_v2)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=500)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
