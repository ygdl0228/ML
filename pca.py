# @Time    : 2023/5/15 20:46
# @Author  : ygd
# @FileName: pca.py
# @Software: PyCharm


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')
x, y = make_blobs(n_samples=10000, n_features=3, centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
                  cluster_std=[0.2, 0.1, 0.2, 0.2], random_state=9)
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(x[:, 0], x[:, 1], x[:, 2], marker='o')

pca=PCA(n_components=2)
pca.fit(x)
x1=x
x_new=pca.transform(x1)
plt.scatter(x_new[:, 0], x_new[:, 1],marker='o')

pca = PCA(n_components=0.95)
pca.fit(x)
x2=x
x1_new=pca.transform(x2)
plt.scatter(x1_new[0], x1_new[1],marker='o')
















plt.show()