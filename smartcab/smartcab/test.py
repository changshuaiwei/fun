import numpy as np

def gradient(X, y, w, b, reg=0.01):
    scores = X.dot(w) + b
    y = y[:,np.newaxis]
    loss = np.mean((y-scores)**2)
    d_w = np.mean((X*((scores-y)*2)),axis=0)[:,np.newaxis]
    d_b = np.mean((scores-y)*2)
    
    return d_w, d_b, loss
            

X = np.random.rand(1000,10)
w_true = np.arange(10)[:,np.newaxis] + 2
#y = np.random.randn(1000)
y = X.dot(w_true)
y = y.flatten()
w = 0.1*np.random.rand(10,1)
b = np.random.randn(1)

for i in xrange(1000):
    d_w, d_b, loss = gradient(X, y, w, b, reg=0.01)
    print '\nloss is {}\n'.format(loss)
    #raw_input("Press Enter to continue...")
    w = w - 0.01 * d_w
    b = b - 0.01 * d_b
                        
                        
                    

