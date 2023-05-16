# @Time    : 2023/5/12 15:48
# @Author  : ygd
# @FileName: logistic_regression.py
# @Software: PyCharm

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression


def main():
    mnist = fetch_openml('mnist_784')
    x, y = mnist['data'], mnist['target']
    x_train = np.array(x[:60000], dtype=float)
    y_train = np.array(y[:60000], dtype=float)
    x_test = np.array(x[6000:], dtype=float)
    y_test = np.array(y[6000:], dtype=float)
    clf = LogisticRegression(penalty='l1', solver='saga', tol=0.1)
    print('开始训练')
    clf.fit(x_train, y_train)
    print("训练结束")
    score = clf.score(x_test, y_test)
    print(score)


if __name__ == "__main__":
    main()
