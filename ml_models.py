import numpy as np


### TO DO :
# Ridge
# SVM
# DecisionTree
# Logistic regression
# Neural networks
###


class Scaler:
    def __init__(self, type='Standard'):
        self.params = np.array([])
        self.type = type

    def fit(self, X):
        X = np.array(X)
        m = X.shape[0]
        p = []
        for k in range(X.shape[1]):
            moy = sum(X[:,k])/len(X[:,k])
            if self.type == 'Standard':
                p.append([moy, np.std(X[:,k])])
            elif self.type == 'Minmax':
                mini, maxi = min(X[:,k]), max(X[:,k])
                p.append([moy, maxi-mini])
        self.params = np.array(p)
        return np.array(p)

    def transform(self, X):
        if self.params.shape == (0,):
            return None
        X = np.array(X)
        a = []
        for k in range(X.shape[1]):
            a.append((X[:,k] - self.params[k,0]) / self.params[k,1])
        X = np.transpose(a)
        return X
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def inv_transform(self, X):
        X = np.array(X)
        for k in range(X.shape[1]):
            X[:,k] = (X[:,k]*self.params[k,1])+self.params[k,0]
        return np.array(X)
        

    



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

    













