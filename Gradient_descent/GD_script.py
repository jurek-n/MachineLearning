# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# %% [markdown]
# load data and plot data

# %%
raw_x1, raw_x2, raw_y = np.loadtxt("ex1data2.txt", delimiter=',', unpack=True)
raw_x = np.vstack((raw_x1, raw_x2)).T

#get_ipython().run_line_magic('matplotlib', 'inline')

fig1 = plt.figure(1)

plt.subplot(131)
plt.plot(raw_x1, raw_y/1000, 'ro')
plt.ylabel('Price of a House in thousands')
plt.xlabel('squre feet')

plt.subplot(132)
plt.plot(raw_x2, raw_y/1000, 'bo')
plt.ylabel('Price of a House in thousands')
plt.xlabel('how many rooms')

plt.subplot(133)
plt.plot(raw_x2, raw_x1, 'rx')

plt.show()


# %%
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d')
ax2.scatter(raw_x1, raw_x2, raw_y, 'bo')

ax2.set_title(" 2 variable data")
ax2.set_xlabel("how many bedrooms")
ax2.set_ylabel("how many sq ft")
ax2.set_zlabel("price of house")
ax2.view_init(30, 30)
plt.show()

# %% [markdown]
# Normalization min max and x-mean/std function

# %%
def featureNormalizeMinMax(X):
    X_norm = np.copy(X)
    for i in range(X_norm.shape[0]):
        for j in range(X_norm.shape[1]):
            X_norm[i,j] = (X_norm[i,j] - np.min(X[:,j]))/(np.max(X[:,j])- np.min(X[:,j]))
    return X_norm


# %%
def featureNormalize(X):
    X_norm = np.copy(X)
    print(X_norm.shape[1])
    for i in range(X_norm.shape[0]):
        for j in range(X_norm.shape[1]):
            X_norm[i,j] = (X_norm[i,j] - np.mean(X[:,j]))/np.std(X_norm[:,j])
    return X_norm

# %% [markdown]
# Normalization prediction function

# %%
def predictionNormalize(f1, f2, raw_data):
    pred = np.array([1, f1, f2], dtype=np.float64)
    for i in range(1, pred.size):
        pred[i] = (pred[i] - np.mean(raw_data[:,i]))/np.std(raw_data[:,i])
        
    return pred


# %%
def predictionNormalizeMinMax(f1, f2, raw_data):
    pred = np.array([1, f1, f2], dtype=np.float64)
    for i in range(1, pred.size):
        pred[i] = (pred[i] - np.min(raw_data[:,i]))/(np.max(raw_data[:,i])- np.min(raw_data[:,i]))
        
    return pred

# %% [markdown]
# Compute Cost function

# %%
def computeCost(x, y, theta):
    m = y.size
    h = x @ theta
    error = np.square(h - y)
    return error.sum() * (1/(2*m))

# %% [markdown]
# Mutivariate gradient descent function

# %%
def gradientDescentMulti(x, y, theta, alpha, iteration):
    
    m = y.size
    learning_rate = (alpha/m)
    temp = np.zeros_like(theta)
    J_history = np.zeros(iteration)
    theta_history = np.zeros((iteration, 3))

    for i in range(iteration):
        h = x @ theta
        #print("h = {}".format(h))
        
        for j in range(theta.size):
            error = (h - y)
            #print("error = {}".format(error))
            grad = error*x[:,j]
            #print("grad = {}".format(grad))
            sum_grad = np.sum(grad)
            #print("sum_grad = {}".format(sum_grad))
            temp[j] = learning_rate*sum_grad

        theta = theta - temp
        #print("theta = {}".format(theta))
        J_history[i] = computeCost(x, y, theta)
        theta_history = theta
        
    return theta, J_history, theta_history

# %% [markdown]
# Normalization and addtion of ones column

# %%
x_norm = featureNormalize(raw_x)
x_norm_min_max = featureNormalizeMinMax(raw_x)


# %%
ones_c = np.ones((x_norm.shape[0], 1), dtype=np.float64)
raw_x = np.hstack((ones_c, raw_x))
x_norm = np.hstack((ones_c, x_norm))
x_norm_min_max = np.hstack((ones_c, x_norm_min_max))

# %% [markdown]
# Gradient descent for every normalization 

# %%
theta = np.zeros(3,)
alpha = 0.01
num_iter = 15000

theta_raw, J_history_raw, theta_history_raw = gradientDescentMulti(raw_x, raw_y, theta, alpha/10000000, num_iter)
theta_norm, J_history_norm, theta_history_norm = gradientDescentMulti(x_norm, raw_y, theta, alpha, num_iter)
theta_norm_min_max, J_history_norm_min_max, theta_history_norm_min_max = gradientDescentMulti(x_norm_min_max, raw_y, theta, alpha, num_iter)

print(" theta raw")
print(theta_raw)
print(" theta norm")
print(theta_norm)
print(" theta norm min max")
print(theta_norm_min_max)

# %% [markdown]
# Cost function over iteration for every normalization 

# %%
n = np.arange(num_iter)
plt.figure(3)
plt.subplot(131, title="no norm", ylabel="cost", xlabel="n of iter")
plt.plot(n, J_history_raw)
plt.subplot(132, title="norm", xlabel="n of iter")
plt.plot(n, J_history_norm)
plt.subplot(133, title="min max norm", xlabel="n of iter")
plt.plot(n, J_history_norm_min_max)

# %% [markdown]
# Selection of a real dataset

# %%
training_set = 0
print("real data: sq ft: {}, bedrooms: {}, price: {}".format(raw_x[training_set,1],
                                                             raw_x[training_set,2],
                                                             raw_y[training_set]))

# %% [markdown]
# Prediction in every normalization

# %%
sq_ft = 2104
nr_beds = 7
pred_raw = np.array([1, sq_ft, nr_beds])
pred_norm = predictionNormalize(sq_ft, nr_beds, raw_x)
pred_norm_min_max = predictionNormalizeMinMax(sq_ft, nr_beds, raw_x)
print("No normalization")
print(" House price for {} sq ft and {} bedrooms = {}".format(sq_ft, nr_beds, pred_raw @ theta_raw))
print("Normalization")
print(" House price for {} sq ft and {} bedrooms = {}".format(sq_ft, nr_beds, pred_norm @ theta_norm))
print("Min Max normalization")
print(" House price for {} sq ft and {} bedrooms = {}".format(sq_ft, nr_beds, pred_norm_min_max @ theta_norm_min_max))

# %% [markdown]
# Ploting of a hypothesis surface
# 
# theta norm

# %%
one = np.ones(1,)
x1 = np.linspace(0, 5000, num=100)
x2 = np.linspace(0, 5, num=100)

h_surface = np.zeros((x1.size, x2.size))
temp = np.zeros(3, dtype=np.float64)
for i in range(x1.size):
    for j in range(x2.size):
        temp = predictionNormalize(x1[i], x2[j], raw_x)
        h_surface[i][j] = temp @ theta_norm
        temp = np.zeros(3,)
        

x1, x2 = np.meshgrid(x1, x2)
plt.figure(4)
ax4 = plt.axes(projection='3d')
surf = ax4.plot_surface(x1, x2, h_surface.T)
data = ax4.scatter(raw_x[:,1], raw_x[:,2], raw_y, 'bo')


ax4.set_xlabel("how many bedrooms")
ax4.set_ylabel("how many sq ft")
ax4.set_zlabel("price of house")

plt.show()

# %% [markdown]
# theta min max

# %%
one = np.ones(1,)
x1 = np.linspace(0, 5000, num=100)
x2 = np.linspace(0, 5, num=100)
h_surface = np.zeros((x1.size, x2.size))
temp = np.zeros(3, dtype=np.float64)
for i in range(x1.size):
    for j in range(x2.size):
        temp = predictionNormalizeMinMax(x1[i], x2[j], raw_x)
        #print(temp)
        #print(theta)
        h_surface[i][j] = temp @ theta_norm_min_max
        temp = np.zeros(3,)
        
x1, x2 = np.meshgrid(x1, x2)
plt.figure(5)
ax5 = plt.axes(projection='3d')
surf = ax5.plot_surface(x1, x2, h_surface.T)
data = ax5.scatter(raw_x[:,1], raw_x[:,2], raw_y, 'bo')

ax4.set_xlabel("how many bedrooms")
ax4.set_ylabel("how many sq ft")
ax4.set_zlabel("price of house")

plt.show()

# %% [markdown]
# theta raw

# %%
one = np.ones(1,)
x1 = np.linspace(0, 5000, num=100)
x2 = np.linspace(0, 5, num=100)
h_surface = np.zeros((x1.size, x2.size))
for i in range(x1.size):
    for j in range(x2.size):
        temp = np.vstack((1, x1[i], x2[j]))
        h_surface[i][j] = temp.T @ theta_raw
        

x1, x2 = np.meshgrid(x1, x2)
plt.figure(6)
ax6 = plt.axes(projection='3d')
surf = ax6.plot_surface(x1, x2, h_surface.T)
data = ax6.scatter(raw_x[:,1], raw_x[:,2], raw_y, 'bo')

ax4.set_xlabel("how many bedrooms")
ax4.set_ylabel("how many sq ft")
ax4.set_zlabel("price of house")

plt.show()


# %%


