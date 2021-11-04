import numpy as np


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
        
