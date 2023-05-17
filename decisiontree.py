# @Time    : 2023/5/16 20:14
# @Author  : ygd
# @FileName: decisiontree.py
# @Software: PyCharm

import seaborn as sns
from pandas import plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Species'] = data.target

antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864']
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
sns.despine(left=True)  # 删除上方和右方坐标轴上不需要的边框，这在matplotlib中是无法通过参数实现的
sns.violinplot(x='Species', y=df.columns[0], data=df, palette=antV, ax=axes[0, 0])
sns.violinplot(x='Species', y=df.columns[1], data=df, palette=antV, ax=axes[0, 1])
sns.violinplot(x='Species', y=df.columns[2], data=df, palette=antV, ax=axes[1, 0])
sns.violinplot(x='Species', y=df.columns[3], data=df, palette=antV, ax=axes[1, 1])
plt.show()
f, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
sns.despine(left=True)
sns.pointplot(x='Species', y=df.columns[0], data=df, color=antV[1], ax=axes[0, 0])
sns.pointplot(x='Species', y=df.columns[1], data=df, color=antV[1], ax=axes[0, 1])
sns.pointplot(x='Species', y=df.columns[2], data=df, color=antV[1], ax=axes[1, 0])
sns.pointplot(x='Species', y=df.columns[3], data=df, color=antV[1], ax=axes[1, 1])
plt.show()
plt.subplots(figsize=(8, 6))
plotting.andrews_curves(df, 'Species', colormap='cool')
plt.show()

target = np.unique(data.target)
target_name = np.unique(data.target_names)
targets = dict(zip(target, target_name))
df['Species'] = df['Species'].replace(target)

x = df.drop(columns='Species')
y = df['Species']
feature_name = x.columns
labels = y.unique()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
model = DecisionTreeClassifier(max_depth=3)
model.fit(x_train, y_train)

text_represrntation = tree.export_text(model)
print(text_represrntation)
plt.figure(facecolor='g')  #
a = tree.plot_tree(model,
                   feature_names=feature_name,
                   class_names=labels,
                   rounded=True,
                   filled=True,
                   fontsize=14)
plt.show()
