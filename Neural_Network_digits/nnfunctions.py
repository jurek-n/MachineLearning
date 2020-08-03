import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.optimize as opt
from decimal import Decimal

def displayData(x, example_width=20):
    m = x.shape[0]
    data_sets = np.zeros((m, example_width, example_width), dtype=np.float64)
    for row_i in range(m):
        data_sets[row_i] = np.reshape(x[row_i], (example_width, example_width)).T

    m = float(m)
    canvas = np.zeros((example_width * 10, example_width * 10))
    k = 0
    for i in range(10):
        for j in range(10):
            pos_x = i*20
            pos_y = j*20
            canvas[pos_x:pos_x+example_width, pos_y:pos_y+example_width] = data_sets[k]
            k += 1

    plt.imshow(canvas, cmap='Greys')


def sigmoid(z):
    g = np.zeros(np.size(z))
    g = 1/(1 + np.exp(-z))
    return g


def sigmoidGradient(z):
    g = np.zeros(z.shape)
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


def unrollThetas(nn_params, input_layer_size, hidden_layer_size, num_labels):
    split_index = (input_layer_size + 1) * hidden_layer_size
    theta_1 = np.reshape(nn_params[0:split_index], (hidden_layer_size, input_layer_size + 1)).copy()
    theta_2 = np.reshape(nn_params[split_index:], (num_labels, hidden_layer_size + 1)).copy()

    return theta_1, theta_2


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lam):
    theta_1, theta_2 = unrollThetas(nn_params, input_layer_size, hidden_layer_size, num_labels)
    #print("t_1.shape = {}".format(theta_1.shape))
    #print("t_2.shape = {}".format(theta_2.shape))
    m = x.shape[0]
    #print(m)
    Y = np.zeros((y.shape[0], num_labels))
    ones = np.ones((x.shape[0], 1))

    for i in range(y.shape[0]):
        Y[i, y[i]] = 1
    #for i in range(y.shape[0]):
    #    if y[i] == 10:
    #        index = 9
    #    else:
    #        index = y[i] - 1
    #        
    #    Y[i][index] = 1
    
    #print("x.shape = {}".format(x.shape))
    a1 = np.hstack((ones, x))
    #print("a1.shape = {}".format(a1.shape))
    z2 = a1 @ theta_1.T
    #print("z2.shape = {}".format(z2.shape))
    a2 = np.hstack((ones, sigmoid(z2)))
    #print("a2.shape = {}".format(a2.shape))
    z3 = a2 @ theta_2.T
    #print("z3.shape = {}".format(z3.shape))
    a3 = sigmoid(z3)
    #print("a3.shape = {}".format(a3.shape))

    theta_1nb = theta_1[:, 1:]
    theta_2nb = theta_2[:, 1:]
    sq_theta_1nb = np.square(theta_1nb)
    sq_theta_2nb = np.square(theta_2nb)
    p = (np.sum(sq_theta_1nb) + np.sum(sq_theta_2nb)) * lam/(2*m)
    cost = (Y * np.log(a3)) + ((1 - Y) * np.log(1 - a3))
    J = (-1 / m) * np.sum(cost) + p
    return J


def nnGrad(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lam):
    theta_1, theta_2 = unrollThetas(nn_params, input_layer_size, hidden_layer_size, num_labels)
    
    m = y.shape[0]
    theta_1_grad = np.zeros_like(theta_1)
    theta_2_grad = np.zeros_like(theta_2)
    Y = np.zeros((y.shape[0], num_labels))
    #print(y)
    #print(Y)
    for i in range(y.shape[0]):
        Y[i, y[i]] = 1
    #for i in range(y.shape[0]):
    #    if y[i] == 10:
    #        index = 9
    #    else:
    #        index = y[i] - 1
    #
    #    Y[i, index] = 1

    ones = np.ones((x.shape[0], 1))
    #print("x.shape = {}".format(x.shape))
    a1 = np.hstack((ones, x))
    #print("a1.shape = {}".format(a1.shape))
    z2 = a1 @ theta_1.T
    #print("z2.shape = {}".format(z2.shape))
    a2 = np.hstack((ones, sigmoid(z2)))
    #print("a2.shape = {}".format(a2.shape))
    z3 = a2 @ theta_2.T
    #print("z3.shape = {}".format(z3.shape))
    a3 = sigmoid(z3)
    #print("a3.shape = {}".format(a3.shape))

    theta_1nb = theta_1[:, 1:]
    theta_2nb = theta_2[:, 1:]

    d3 = a3 - Y
    d2 = (d3 @ theta_2nb) * sigmoidGradient(z2)
    D2 = d3.T @ a2
    D1 = d2.T @ a1

    theta_1_grad = (1 / m) * D1
    #print("theta_1_grad.shape = {}".format(theta_1_grad.shape))
    theta_2_grad = (1 / m) * D2
    #print("theta_2_grad.shape = {}".format(theta_2_grad.shape))
    theta_1_grad[:, 1:] += ((lam / m) * theta_1nb)
    theta_2_grad[:, 1:] += ((lam / m) * theta_2nb)

    grad = np.hstack((np.reshape(theta_1_grad, theta_1_grad.size),
                       np.reshape(theta_2_grad, theta_2_grad.size)))
    return grad


def randInitializeWeights(L_in, L_out):
    epsilion_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * epsilion_init
    return W


def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    W = np.reshape(np.sin(range(W.size)), W.shape) / 10
    return W


def computeNumericalGradient(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1 * 10**(-4)
    for p in range(theta.size):
        perturb.reshape(perturb.size)[p] = e
        loss1= J(theta - perturb)
        loss2= J(theta + perturb)
        numgrad.reshape(numgrad.size)[p] = (loss2 - loss1) / (2*e)
        perturb.reshape(perturb.size)[p] = 0
    
    return numgrad


def checkNNgradients(lam):
    i_layer_size = 3
    h_layer_size = 5
    num_labels = 3
    m = 5

    theta1 = debugInitializeWeights(h_layer_size, i_layer_size)
    theta2 = debugInitializeWeights(num_labels, h_layer_size)

    X = debugInitializeWeights(m, i_layer_size - 1)
    y = np.mod(range(m), num_labels)
    y = np.reshape(y, (y.shape[0], 1))

    nn_params = np.hstack((np.reshape(theta1, theta1.size),
                           np.reshape(theta2, theta2.size)))
    
    def decCostFunc(p):
        return nnCostFunction(p, i_layer_size, h_layer_size, num_labels, X, y, lam)
    

    grad = nnGrad(nn_params, i_layer_size, h_layer_size, num_labels, X, y, lam)
    numgrad = computeNumericalGradient(decCostFunc, nn_params)
    

    fmt = '{:25} {}'
    print(fmt.format('Numerical Gradient', 'Analytical Gradient'))
    for numerical, analytical in zip(numgrad, grad):
        print(fmt.format(numerical, analytical))

    print('The above two columns you get should be very similat. \n'\
          'Left Col.: Youe Numerical Gradient, Right>Col.:Analytical Gradient) ')

    diff = Decimal(np.linalg.norm(numgrad-grad))/Decimal(np.linalg.norm(numgrad+grad))
    print('If your backpropagation implementation is correct, then\n' \
          'the relative difference will be small (less than 1e-9)\n ' \
          '\nRelative Difference: {:.10E}'.format(diff))


def predict(theta1, theta2, x):
    x = x.T
    x = np.reshape(x, 400)
    x = np.hstack(((1, x)))
    p = np.zeros(1)
    h1 = sigmoid(x @ theta1.T)
    h1 = np.hstack((1, h1))
    h2 = sigmoid(h1 @ theta2.T)
    p = np.argmax(h2)
    return p 

