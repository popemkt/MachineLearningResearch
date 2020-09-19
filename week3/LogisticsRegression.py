import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(a: np.ndarray):
    return 1 / (1 + np.exp(-a))


def generate_data(size: int):
    df = pd.read_csv("F:\Capstone\Playground\week3\data.txt", header=None)
    X = df.iloc[:, :-1].values
    y = (df.iloc[:, -1].values).reshape(-1,1)
    return X, y


def plot_data(X: np.ndarray, y: np.ndarray):
    pos, neg = (y == 1).reshape(100, 1), (y == 0).reshape(100, 1)
    plt.scatter(X[pos[:, 0], 0], X[pos[:, 0], 1], c="r", marker="+")
    plt.scatter(X[neg[:, 0], 0], X[neg[:, 0], 1], marker="o", s=10)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(["= 0", "= 1"], loc=0)


def plot_result(p_vectors: np.ndarray, X: np.ndarray):
    x_value = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
    y_value = -(p_vectors[0]+ p_vectors[1]*x_value)/p_vectors[2]
    plt.plot(x_value, y_value, "g")


def cost_function(m: int, y_pred: np.ndarray, y: np.ndarray):
    cost = -(1/m)*(np.log(y_pred).T.dot(y)+np.log(1-y_pred).T.dot(1-y))
    return cost[0]


def add_x0(X: np.ndarray, m: int):
    # add x0 = 1 to all training samples
    X0 = np.ones((m, 1))
    new_X = np.concatenate((X0, X), axis=1)
    return new_X


def predict(p_vectors: np.ndarray, X: np.ndarray):
    # y = X.p
    return sigmoid(np.dot(X, p_vectors))


def gradient_descent(p_vectors: np.ndarray, learning_rate: float, m: int, y0: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    err = (y0 - y)
    x_T = x.T
    der = 1 / m * np.dot(x_T, err)
    updated_p = p_vectors - learning_rate * der
    return updated_p


def scale(x: np.ndarray):
    x_T = x.T
    n = x_T.shape[0]
    new_x = np.zeros(x_T.shape)
    new_x[0] = x_T[0]
    for i in range(1, n):
        cur_x = x_T[i]
        max_abs = abs(cur_x.max())
        min_abs = abs(cur_x.min())
        div = max_abs if max_abs > min_abs else min_abs
        div = div if div != 0 else 1
        new_x[i] = cur_x / div
    new_x = new_x.T
    return new_x


def logistics_regression(X0: np.ndarray, y: np.ndarray, m: int, p_vectors: np.ndarray, epochs: int, learning_rate: float):
    n = X0.shape[1] - 1
    if (p_vectors is None):
        p_vectors = np.zeros((n + 1, 1))
    y0 = predict(p_vectors, X0)
    loss = cost_function(m, y0, y)
    count = 0
    while count < epochs:
        p_vectors = gradient_descent(p_vectors, learning_rate, m, y0, y, X0)
        y0 = predict(p_vectors, X0)
        loss = cost_function(m, y0, y)
        count += 1
    return p_vectors


m = 100
degree = 3

X, y = generate_data(size=m)

X0 = add_x0(X/100, m)
# X0 = scale(X0)

p_vectors = logistics_regression(
    X0, y, m, p_vectors=None, epochs=100000, learning_rate=5)

plot_data(X0[:,1:], y)
plot_result(p_vectors=p_vectors, X=X0[:,1:])
plt.show()
