import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def model(x, theta):
    ret = sigmoid(np.dot(x, theta.T))
    return ret


def cost(x, y, theta):
    left = np.multiply(-y, np.log(model(x, theta)))
    right = np.multiply(1 - y, np.log(1 - model(x, theta)))
    return np.sum(left - right) / len(x)


def gradient(x, y, theta):
    grad = np.zeros(theta.shape)
    ret = model(x, theta)
    error = (ret - y).ravel()
    for i in range(len(theta.ravel())):
        term = np.multiply(error, x[:, i])
        grad[0, i] = np.sum(term) / len(x)
    return grad


def stopCriterion(stopType, value, threshold):
    if stopType == STOP_ITER:
        return value > threshold
    elif stopType == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold
    elif stopType == STOP_GRAD:
        return np.linalg.norm(value) < threshold


def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    x = data[:, 0:cols - 1]
    y = data[:, cols - 1:]

    return x, y


def descent(data, theta, batchSize, stopType, threshold, alpha):
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    x, y = shuffleData(data)
    costs = [cost(x, y, theta)]

    while True:
        grad = gradient(x[k:k + batchSize], y[k:k + batchSize], theta)
        k += batchSize
        if k >= data.shape[0]:
            k = 0
            x, y = shuffleData(data)
        theta = theta - alpha * grad
        costs.append(cost(x, y, theta))
        i += 1
        value = None
        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad

        if stopCriterion(stopType, value, threshold):
            break

    return theta, i - 1, costs, grad, time.time() - init_time


def runExpe(data, theta, batchSize, stopType, threshold, alpha):
    theta, iterTimes, costs, grad, dur = descent(data, theta, batchSize, stopType, threshold, alpha)
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += " data - learning rate: {0} - ".format(alpha)
    if batchSize == data.shape[0]:
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER:
        strStop = "{} iterations".format(threshold)
    elif stopType == STOP_COST:
        strStop = "costs change < {}".format(threshold)
    else:
        strStop = "gradient norm < {}".format(threshold)
    name += strStop
    print("***{}\nTheta: {} - iterTimes: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iterTimes, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    plt.show()
    return theta


data_path = 'D:' + os.sep + 'TestData' + os.sep + 'LogiReg_data.csv'

pdData = pd.read_csv(data_path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
pdData.insert(0, 'Ones', 1)
orig_data = pdData.as_matrix()

cols = orig_data.shape[1]  # 4
x = orig_data[:, 0:cols - 1]
y = orig_data[:, cols - 1:]

theta = np.zeros([1, 3])
runExpe(orig_data,theta,orig_data.shape[0],STOP_ITER,threshold=5000,alpha=0.000001)
runExpe(orig_data,theta,orig_data.shape[0],STOP_COST,threshold=0.000001,alpha=0.001)

# positive = pdData[pdData['Admitted'] == 1]
# negative = pdData[pdData['Admitted'] == 0]
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
# ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
# plt.show()
