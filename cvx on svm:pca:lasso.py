# primal formulation of lasso using admm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x = pd.read_csv('X.csv').iloc[:,1:]
y = pd.read_csv('y.csv').iloc[:,1]

RHO = 1
LAMBDA= 0.3
Xrhoinv = np.linalg.inv(x.T.dot(x) + RHO * np.eye(x.shape[1]))
XTY = x.T.dot(y)

def objfun(z):
    return 0.5 * np.linalg.norm(x.dot(z) - y) ** 2 + LAMBDA * np.abs(z).sum()

z = np.zeros((x.shape[1],))
u = np.zeros((x.shape[1],))

def soft_threshold(x, t):
    x = x.copy()
    x[np.where(np.abs(x) < t)] = 0
    x[np.where(x > 0)] -= t
    x[np.where(x < 0)] += t
    return x

def run_admm_iter(z, u):
    xstar = Xrhoinv.dot(XTY + RHO * (z - u))
    zstar = soft_threshold(xstar + u, LAMBDA / RHO)
    ustar = u + xstar - zstar
    return zstar, ustar

def run(z, u):
    state = [(z, objfun(z))]
    for i in range(20):
        z, u = run_admm_iter(z, u)
        state.append((z, objfun(z)))
    return state

state = run(z, u)
plt.plot([o for z, o in state])
plt.yscale('log')
plt.title('ADMM Lasso')
plt.xlabel('Iteration (k)')
plt.ylabel('log(obj)');
plt.show()




# dual formulation of lasso
import cvxpy as cp

v = cp.Variable(x.shape[0])
lambd = cp.Parameter(nonneg=True)

objective = cp.Minimize(cp.pnorm(v,1)**2 - v.T*y)
constraint = [cp.max(cp.abs(x.T*v))<=lambd]
prob = cp.Problem(objective,constraint)

lambd.value = 0.1
prob.solve()




# dual formulation of svm
a = cp.Variable(x.shape[0])
c = cp.Parameter(nonneg = True)
objective = cp.Maximize(sum(a)-0.5*sum((a*(x.T*y).T**2)))
constraint = [y.T*a == 0, a >= 0, a<= c]
prob = cp.Problem(objective,constraint)
c.value = 0.5
prob.solve()




# sparce pca
sigma = (x.T.dot(x))/x.shape[0]
p = cp.Variable((x.shape[1],x.shape[1]),symmetric = True)
k = cp.Parameter(nonneg = True)
obj = cp.Maximize(cp.trace(sigma*p))
cons = [cp.trace(p) == 1,
        p >= 0,
        cp.norm(p,1)<=k]
k.value = 35
prob = cp.Problem(obj,cons)
prob.solve()




# pca
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

bc = open("wdbc.data.txt","r")
data = []
y = []
for line in bc:
    sample = []
    for value in line.split(',')[2:12]:
        sample.append(float(value))
    if line.split(',')[1] == 'M':
        y.append(1)
    else: y.append(0)
    data.append(sample)   
    
labels = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
          'compactness', 'concavity', 'concave points', 'symmetry', 'fractal_dimension']

data = StandardScaler().fit_transform(data)

pca = PCA().fit(data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()       
# from the plot we can find 3 components can explain clearly
 
pca = PCA(3).fit(data)
loading = pca.components_.T * np.sqrt(pca.explained_variance_)

a = cp.Variable(data.shape[0])
c = cp.Parameter(nonneg = True)
objective = cp.Maximize(sum(a)-0.5*sum(a*((PCA(3).fit_transform(data).T*y).T**2)))
constraint = [y.T*a == 0, a >= 0, a<= c]
prob = cp.Problem(objective,constraint)
c.value = 0.05
prob.solve()



# spca
from sklearn.decomposition import SparsePCA

spca = SparsePCA(2)
spca.components_
# choose the same component as pca

a = cp.Variable(data.shape[0])
c = cp.Parameter(nonneg = True)
objective = cp.Maximize(sum(a)-0.5*sum(a*((spca.fit_transform(data).T*y).T**2)))
constraint = [y.T*a == 0, a >= 0, a<= c]
prob = cp.Problem(objective,constraint)
c.value = 0.05
prob.solve()

# optimal value for pca is smaller than spca, which mean spca performs better.



