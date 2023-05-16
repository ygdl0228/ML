# @Time    : 2023/5/15 20:20
# @Author  : ygd
# @FileName: knn.py
# @Software: PyCharm

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')
digits = load_digits()
data = digits.data
target = digits.target

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
score = accuracy_score(y_test, knn.predict(x_test))
print(score)
