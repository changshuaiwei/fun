import numpy as np
import random

class qmem(object):
    def __init__(self, maxsize=10000, margin=100, alpha=1):
        self.s = None
        self.a = None
        self.new_s = None
        self.r = None
        self.done = None
        self.v = None

        self.maxsize = maxsize
        self.margin = margin
        self.alpha= alpha #for importance sampling
        self.size = 0

    def add_instance(self, state, action, reward, new_state, done):
        if self.size == 0:
            self.s = state[np.newaxis, :]
            self.a = action
            self.r = reward
            self.new_s = new_state[np.newaxis, :]
            self.done = done
            self.v = 0
        else:
            self.s = np.r_[self.s, state[np.newaxis, :]]
            self.a = np.r_[self.a, action]
            self.r = np.r_[self.r, reward]
            self.new_s = np.r_[self.new_s, new_state[np.newaxis, :]]
            self.done = np.r_[self.done, done]
            self.v = np.r_[self.v, 0]

        self.size +=1

        # print self.s
        # print self.a
        # raw_input("Countinue...")

        if self.size>self.maxsize:
            self.del_instances(range(self.margin))

    def del_instances(self, idx):
        # idx is a nump vec of index
        self.s = np.delete(self.s, idx, axis=0)
        self.new_s = np.delete(self.new_s, idx, axis=0)
        self.a = np.delete(self.a, idx)
        self.r = np.delete(self.r, idx)
        self.done = np.delete(self.done, idx)
        self.v = np.delete(self.v, idx)
        self.size -= len(idx)

    def sampling(self, n, importance=False):
        idx = None
        if importance:
            samp_p = self._calculate_importance()
            idx = np.random.choice(self.size, size=n, p=samp_p)
        else:
            idx = np.random.randint(self.size, size=n)
        s = self.s[idx, :]
        new_s = self.new_s[idx, :]
        a = self.a[idx]
        r = self.r[idx]
        done = self.done[idx]
        self.v[idx] += 1

        return (s, a, r, new_s, done)

    def _calculate_importance(self):
        p = (1.0/ (self.v + 1.0))**self.alpha
        return p/np.sum(p)





