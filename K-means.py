# @Time    : 2023/5/15 20:24
# @Author  : ygd
# @FileName: K-means.py
# @Software: PyCharm

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')
x = np.random.rand(100, 2)
plt.scatter(x[:, 0], x[:, 1], marker='o')

kmeans = KMeans(n_clusters=2).fit(x)
label_pre = kmeans.labels_
plt.scatter(x[:, 0], x[:, 1], c=label_pre)
plt.show()
