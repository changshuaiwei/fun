import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd

class LearningAgent2(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent2, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.trip_history = []
        self.debug = True
        self.gamma = 0.2 #upper bound of discount 
        self.alpha = 0.5 #uppder bound of learning rate
        self.epsilon = 0.2 #lower bound of proportion of random steps
        self.reg = 0.01 # regularization param for regression
        self.lr = 0.1 # learning rate for regression
        self.clock_update = 0 # store number of updates
        self.init_params_scale = 1e-4 # scale of initial params setting
        
        self.max_memory = 400 # number of rows for state_action_experience
        self.batch_size = 20 # size of batch
        self.batch_step = 20 # extract a batch for each batch_step steps
        self.param_step = self.max_memory # how many step should update w
        
        self.state_feature = ['right_no', 'forward_no', 'left_no', 'next_right', 'next_forward', 'next_left']
        self.action_feature = ['right', 'forward', 'left']
        
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
        loss = np.mean((y-scores)**2)

        if False:
            print '\ny are\n {}\n and scores are\n {}\n and loss is\n {}\n'.format(str(y), str(scores), str(loss) )
            raw_input("Press Enter to continue...")        
        
        d_w = np.mean((X*((scores-y)*2)),axis=0)[:,np.newaxis]
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
                if False:
                    print '\nParameters are {}\n'.format(str(self.params))
                    raw_input("Press Enter to continue...")

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent2)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=1000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
