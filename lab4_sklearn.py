from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sklearn
# module tạo và huấn luyện mô hình hồi quy tuyến tính
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# doc file housing
data = pd.read_csv("Housing_2019.csv", index_col=0)
data.iloc[2:4,]
x = data.iloc[:, [1, 2, 3, 4, 10]]  # chọn tất cả các hàng của cột 1,2,3,4,10
x.iloc[1:5,]
y = data.price

# hiển thị tương quan giữa diện tích và giá nhà
plt.scatter(data.lotsize, data.price)
""" plt.show() """

# huấn luyện mô hình bằng sklearn
# Xây dựng mô hình Linear Regression và fit dữ liệu
lm = linear_model.LinearRegression()
lm.fit(x[0:520], y[0:520])

print("#######################################################")
# Tất cả các thược tính của tập dữ liệu housing_2019.csv
# Chuyển danh sách các thuộc tính thành chuỗi
cols_str = ', '.join(data.columns.tolist())
# In ra danh sách các thuộc tính dưới dạng chuỗi
print("Cac thuoc tinh cua tap du lieu:", cols_str, ".")
# Các thuộc tính dùng để dự đoán giá nhà
cols_str2 = ', '.join(x.columns.tolist())
print("Cac thuoc tinh dung de du doan gia nha: ", cols_str2)
print("########################################################")

# In ra các giá trị theta
print("theta0:", lm.intercept_)  # theta0
print("theta1,5:", lm.coef_)  # theta1, theta2, theta3, theta4, theta5

print("########################################################")
# dự báo phần giá nhà với 20 phan tư cuối cùng
y = data.price
y_test = y[-20:]
x_test = x[-20:]
y_pred = lm.predict(x_test)

print("Thuc te:")
print(y_test)
print("Du doan:")
print(y_pred.reshape(-1, 1))

print("########################################################")

err = mean_squared_error(y_test, y_pred)

print("Binh phuong trung binh loi:", err)

rmse_error = np.sqrt(err)
# làm tròn phương sai
print("Phuong sai: ", round(rmse_error, 3))


print("########################################")
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=1/3, random_state=100)


tree = DecisionTreeRegressor(random_state=0)

bagging_tree = BaggingRegressor(
    base_estimator=tree, n_estimators=10, random_state=42)
bagging_tree.fit(X_train, Y_train)
y_pred = bagging_tree.predict(X_test)
err = mean_squared_error(Y_test, y_pred)
print(err)
