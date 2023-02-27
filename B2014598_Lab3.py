import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
dulieu = pd.read_csv("iris_data.csv")
data = pd.read_csv("housing_RT.csv", index_col=0)
""" print(data) """
# lay file iris truc tiep tu sklearn
iris_data = load_iris()

# test data
""" print(iris_data.data) """
""" print(iris_data.target) """

# Phan chia du lieu theo nghi thuc hold-out
X_train, x_test, y_train, y_test = train_test_split(  # hai tap X_train, Y_train dung de huan. Tap x_test, y_test dung de kiem tra
    iris_data.data, iris_data.target, test_size=1/3, random_state=5)

""" print(X_train)
print(y_train)
print(x_test)
print(y_test) """

# Xây dựng mô hình cây quyết định dựa trên chỉ số Gini với độ sâu của cây bằng 3, nút nhánh ít nhất có 5 phần tử.
clf_Gini = DecisionTreeClassifier(
    criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
clf_Gini.fit(X_train, y_train)

# Dự đoán nhãn cho các phần tử trong tập kiểm tra
y_pred = clf_Gini.predict(x_test)
""" print(y_pred) """

y2_pred = clf_Gini.predict([[4, 4, 3, 3]])
""" print(y2_pred)"""
""" 4,4,3,3 la [2] """

# Tính độ chính xác cho giá trị dự đoán của phần tử trong tập kiểm tra
print("accuracy_score is: ", accuracy_score(y_test, y_pred) * 100)
""" Do chinh xac la 94.0%"""

# Tính độ chính xác cho giá trị dự đoán thông qua ma trận con
# lay test va tap du doan de kiem tra
dochixac_matric = confusion_matrix(y_test, y_pred, labels=[2, 0, 1])
""" print(dochixac_matric) """


# ----------------------------------------------------------------------
# Sử dụng nghi thức k-fold để phân chia tập dữ liệu “iris” với k=15 với hàm Kfold

# Chia tap du lieu thanh 15
kf = KFold(n_splits=15)
for train_index, test_index in kf.split(dulieu):
    # kiem tra tap train va test
    """ print("Train: ", test_index, "Test: ", test_index) """
    X_train, X_test = dulieu.iloc[train_index, ], dulieu.iloc[test_index, ]
    y_train, y_test = dulieu.iloc[train_index, ], dulieu.iloc[test_index, ]
    # kiem tra X_test hold_out X_test K F
    """ print("X_test: ", X_test, "x_test: ", x_test) """

# ---------------------------------------------------------------------

X3_train, X3_test, Y3_train, Y3_test = train_test_split(
    data.iloc[:, 1: 5], data.iloc[:, 0], test_size=1/3, random_state=100)

regressor = DecisionTreeClassifier(random_state=0)
regressor.fit(X3_train, Y3_train)

y3_pred = regressor.predict(X3_test)

err = mean_absolute_error(Y3_test, y3_pred)
print(err)

a = np.sqrt(err)
print(a)
