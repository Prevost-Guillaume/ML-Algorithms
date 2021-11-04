import numpy as np


class Perceptron:
    def __init__(self, loss, sigmoid, history=True):
        self.history = history
        self.reshape_necessary = False
        self.W = None
        self.sig = sigmoid
        self.loss = loss
        

    def fit(self, X, y, alpha=1e-3, iterations=1000):
        # Add column 1 to X
        X = np.array([[x for x in X[i]]+[1] for i in range(X.shape[0])])
        # Init theta to 0
        self.W = np.zeros((X.shape[1], 1))
        if len(y.shape) == 1:
            y = y.reshape((y.shape[0], 1))
            self.reshape_necessary = True
        J_history = []

        # Start gradient descent        
        for iter in range(iterations):
            A = self.sigmoid(self.get_Z(X, self.W))
            self.W = self.W-alpha*self.get_grad(X, A, y)

            if self.history:
                J_history.append(self.get_loss(A, y))
                
        return J_history
        
    def get_Z(self, X, W):
        return np.dot(X, W)
        
    def sigmoid(self, Z):
        return self.sig(Z)

    def get_grad(self, X, A, y):
        m = y.shape[0]
        t = np.transpose(X)
        return (1/m)*np.dot(t,(A-y))


    def predict(self, X):
        X = np.array([[x for x in X[i]]+[1] for i in range(X.shape[0])])
        p = self.predict_(X)
        if self.reshape_necessary == True:
            return p[:,0]
        else:
            return p

    def predict_(self, X):
        return self.sigmoid(self.get_Z(X, self.W))

    def get_loss(self, A, y):
        return self.loss(A, y)

    def get_coefficients(self):
        return self.W


class PerceptronClassifier(Perceptron):
    def __init__(self, history=True):
        self.history = history
        self.reshape_necessary = False
        self.W = None
        
        Perceptron.__init__(self, loss=lambda A,y : -(y * np.log(A) + (1 - y) * np.log(1 - A)).mean(),
                            sigmoid = lambda Z : 1/(1+np.exp(-Z)), history=history)
        

class PerceptronRegressor(Perceptron):
    def __init__(self, history=True):
        self.history = history
        self.reshape_necessary = False
        self.W = None
        
        Perceptron.__init__(self, loss=lambda A,y : ((A-y)**2).mean(),
                            sigmoid = lambda Z : Z, history=history)
        

