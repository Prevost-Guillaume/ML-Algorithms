import numpy as np
import matplotlib.pyplot as plt
import random
from ml_models import Scaler, LinearRegression


def generateLinearData(n=100, n_dim=1, window=[-10, 10], random_coeff=5):
    X = []
    Y = []
    
    n_dim = n_dim+1
    coeffs_poly = [(2*random.random()-1) for i in range(n_dim)]
    
    for i in range(n):
        x = random.randrange(window[0], window[1])
        poly = sum([coeffs_poly[i]*x**i for i in range(n_dim)])
        y = poly + random_coeff*random.gauss(0,1)
        X.append([x])
        Y.append(y)
    return np.array(X),np.array(Y)
    






# Generate data 
X,y = generateLinearData(n=1000, n_dim=2, window=[-100, 100], random_coeff=5)
#plt.scatter(X,y)
#plt.show()

# Add polynomial feature
X = np.array([[x for x in X[i]]+[x*x for x in X[i]]+[x*x*x for x in X[i]] for i in range(X.shape[0])])

# train_test split
split = 0.8
n = int(split*X.shape[0])
x_train, x_test = X[:n], X[n:]
y_train, y_test = y[:n], y[n:]
print(x_train.shape)

# Scale X
scaler = Scaler(type='Standard')
x_train = scaler.fit_transform(x_train)

# Fit regressor
clf = LinearRegression(history=True)
costs = clf.fit(x_train, y_train, alpha=0.001, iterations=10000)

# Predict x_test
x_test = scaler.transform(x_test)
pred = clf.predict(x_test)

# Get initial data
x_test = scaler.inv_transform(x_test)

print('Coût final : ',costs[-1])



# Plot curve
plt.scatter([i[0] for i in x_test], pred, label='Prediction', s=7)
plt.scatter([i[0] for i in x_test], [i for i in y_test], label='Données test', s=7)
plt.legend()
plt.show()


# Plot cost evolution
plt.plot([i for i in range(len(costs))], costs, label='Loss')
plt.legend()
plt.show()














