import numpy as np


class LinearRegression:
    def __init__(self, history=True):
        self.history = history

    def fit(self, X, y, alpha=0.01, iterations=1000):
        # Add column 1 to X
        X = np.array([[x for x in X[i]]+[1] for i in range(X.shape[0])])
        # Init theta to 0
        theta = np.zeros((X.shape[1],1))
        J_history = []

        # Start gradient descent        
        for iter in range(iterations):
            delta = self.calculGradient(X, y, theta)
            theta = theta - alpha*delta
            if self.history:
                J_history.append(self.calculCost(X, y, theta))
        self.theta = theta
        return J_history
        
    def predict(self, X):
        # Add column 1 to X
        X = np.array([[x for x in X[i]]+[1] for i in range(X.shape[0])])
        return np.dot(X,self.theta).reshape(X.shape[0])


    def calculGradient(self, X, y, theta):
        m = len(y)
        predictions = np.dot(X,theta).reshape(m)
#        delta = [sum([(predictions[i] - y[i])*X[i,xi] for i in range(m)])/m for xi in range(X.shape[1])]
        delta = np.dot((predictions - y), X)
        delta = np.array(delta).reshape(theta.shape)
        return delta

    def calculCost(self, X, y, theta):
        m = len(y)
        predictions = np.dot(X,theta).reshape(m)
        sqrErrors = np.square(predictions - y)
        J = sum(sqrErrors)/(2*m)
        return J

    def getCoefficients(self):
        return self.theta
