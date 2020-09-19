import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def generate_data(size: int):
    np.random.seed(0)
    x = np.array(sorted(2 - 3 * np.random.normal(0, 1, size)))
    y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, size)
    plt.scatter(x, y, s=10)
    # plt.show()
    return (x.reshape(-1, 1), y.reshape(-1, 1))
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def generate_data(size: int):
    np.random.seed(0)
    x = np.array(sorted(2 - 3 * np.random.normal(0, 1, size)))
    y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, size)
    plt.scatter(x, y, s=10)
    # plt.show()
    return (x.reshape(-1, 1), y.reshape(-1, 1))

def preprocess(X: np.ndarray, degree: int):
    # add features
    additional_features = np.empty([X.shape[0], degree - 1])
    for i in range(1,X.shape[0]+1):
        for j in range(2,degree+1):
            additional_features[i-1,j-2] = X[i-1, 1]**j
    X = np.concatenate((X, additional_features), axis=1)
    return X


def cost_function(m: int, y_pred: np.ndarray, y: np.ndarray):
    cost = 1/(2*m) * np.sum((y_pred-y)**2)
    return cost

def add_x0(X: np.ndarray, m: int):
    # add x0 = 1 to all training samples
    X0 = np.ones((m, 1))
    new_X = np.concatenate((X0, X), axis=1)
    return new_X


def predict(p_vectors: np.ndarray, X: np.ndarray):
    # y = X.p
    return np.dot(X, p_vectors)


def gradient_descent(p_vectors: np.ndarray, learning_rate: float, m: int, n: int,
                     y0: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    err = (y0 - y)
    x_T = x.T
    der = 1 / m * np.dot(x_T, err)
    updated_p = p_vectors - learning_rate * der
    return updated_p


def scale(x: np.ndarray):  # bien X chi con nam trong khoang -1 <= x <= 1
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


def polynomial_regression(X0: np.ndarray, y: np.ndarray, m: int, p_vectors: np.ndarray, epochs: int, learning_rate: float, degree: int):
    n = degree
    X0 = preprocess(X0,n)
    if (p_vectors is None):
        p_vectors = np.zeros((n + 1, 1))
    y0 = predict(p_vectors, X0)
    loss = cost_function(m, y0, y)
    count = 0
    while count < epochs:
        p_vectors = gradient_descent(p_vectors, learning_rate, m, n, y0, y, X0)
        y0 = predict(p_vectors, X0)
        loss = cost_function(m, y0, y)
        count += 1
    return (p_vectors, X0)


m = 100
degree = 3

data = generate_data(size=m)

X = data[0]
y = data[1]

print(X)

X0 = add_x0(X, m)
X0 = scale(X0)
print(X0)
print(X0[2,0])
p_vectors, X0_processed = polynomial_regression(
    X0, y, m, p_vectors=None, epochs=100000, learning_rate=0.01, degree=degree)

pred = predict(p_vectors, X0_processed)

print(pred)


plt.scatter(X, y)
plt.plot(X, pred, c='red')
plt.show()