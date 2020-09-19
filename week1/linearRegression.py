import numpy as np
import matplotlib.pyplot as plt

# number of examples
def cost_function(m: int, y_pred: np.ndarray, y: np.ndarray):
    cost = 1/(2*m) * np.sum((y_pred-y)**2)
    return cost

m = 5

def add_x0(X: np.ndarray, m: int):
    X0 = np.ones((m, 1))
    new_X = np.concatenate((X0, X), axis=1)
    return new_X

def predict(p_vectors: np.ndarray, X: np.ndarray):
    # y = X.p
    return np.dot(X, p_vectors)

def gradient_descent(p_vectors: np.ndarray, learning_rate: float, m: int, n: int,
    y0: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    # start
    err = (y0 - y)
    x_T = x.T
    der = 1 / m * np.dot(x_T, err) # cong thuc tinh dao ham cua ham mat mat (cost function)
    updated_p = p_vectors - learning_rate * der # cong thuc tinh he so cap nhật
    return updated_p 

# tien_luong_du_doan = theta_0 + theta_1*gio_cong

def scale(x: np.ndarray): # bien X chi con nam trong khoang -1 <= x <= 1
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

def linear_regression(X0: np.ndarray, y: np.ndarray, m: int, p_vectors: np.ndarray, epochs: int, learning_rate: float):
    n = X0.shape[1] - 1 # num of features
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
    return p_vectors

X = np.array([
    [150.0],
    [120.0],
    [110.0],
    [170.0],
    [180.0],
]).reshape(-1, 1)

random = np.random.rand(m) * 100000

y = np.dot(X, np.array([15000])) + random
y = y.reshape(-1, 1)
print(y)

X0 = add_x0(X, m)
X0 = scale(X0)
print(X0)
p_vectors = linear_regression(X0, y, m, p_vectors=None, epochs=100000, learning_rate=0.1)

pred = predict(p_vectors, X0)

plt.scatter(X, y)
plt.plot(X, pred, c='red')
plt.show()