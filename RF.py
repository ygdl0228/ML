# @Time    : 2023/5/16 20:56
# @Author  : ygd
# @FileName: RF.py
# @Software: PyCharm

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

wine = load_wine()
x = pd.DataFrame(data=wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
base_model = DecisionTreeClassifier(max_depth=3, criterion='gini').fit(x_train, y_train)
base_score = accuracy_score(y_test, base_model.predict(x_test))
print(base_score)

bagging_model = BaggingClassifier(base_estimator=base_model, n_estimators=50)
bagging_model.fit(x_train, y_train)
bagging_score = accuracy_score(y_test, base_model.predict(x_test))
print(bagging_score)

rf_model = RandomForestClassifier(n_estimators=50)
rf_model.fit(x_train, y_train)
rf_score = accuracy_score(y_test, rf_model.predict(x_test))
print(rf_score)
