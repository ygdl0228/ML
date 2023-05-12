# @Time    : 2023/5/11 21:48
# @Author  : ygd
# @FileName: linear_regression.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

np.random.seed(0)


def true_fun(x):
    return 1.5 * x + 0.2


def grad_fun(x, y, theta, alpha):
    error = np.dot(x, theta) - y
    grad = x.transpose().dot(error) / len(x)
    theta = theta - alpha * grad
    return theta, sum(error) / len(x)


n_samples = 100

x_samples = np.sort(np.random.rand(n_samples))
y = (true_fun(x_samples) + np.random.randn(n_samples) * 0.05).reshape(n_samples, 1)
x = np.array([[1, i] for i in x_samples])

plt.subplot(221)
plt.scatter(x_samples, y)

max_iter = 1000
alpha = 0.05
m, p = np.shape(x)
theta = np.ones((p, 1))
loss_list = []
for _ in range(max_iter):
    theta, error = grad_fun(x, y, theta, alpha)
    loss_list.append(error)
print(theta)
plt.subplot(222)
plt.plot(loss_list)

x_test = np.linspace(0, 1, 100)
plt.subplot(223)
plt.plot(x_test, true_fun(x_test), label='true')
plt.plot(x_test, x_test * theta[1][0] + theta[0][0], label='pre')
plt.scatter(x_samples, y)
plt.legend()
plt.show()

model = LinearRegression()
model.fit(x_samples[:, np.newaxis], y)
print(model.coef_)
print(model.intercept_)
x_test = np.linspace(0, 1, 100)
plt.plot(x_test, model.predict(x_test[:, np.newaxis]), label='true')
plt.plot(x_test, true_fun(x_test), label='pre')
plt.scatter(x_samples, y)
plt.legend()
plt.show()


def true_fun(x):
    return np.cos(1.5 * np.pi * x)


degrees = [1, 4, 15]

x = np.sort(np.random.rand(n_samples))
y = true_fun(x) + np.random.randn(n_samples) * 0.1
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])  # 使用pipline串联模型
    pipeline.fit(x[:, np.newaxis], y)

    # 使用交叉验证
    scores = cross_val_score(pipeline, x[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)
    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(x, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
plt.show()
