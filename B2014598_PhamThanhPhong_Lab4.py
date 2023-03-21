from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

""" X = np.array([1, 2, 4])
Y = np.array([2, 3, 6])

# Bieu dien toa do len mat phang toa do
plt.axis([0, 5, 0, 8])  # Hoanh 0->5, tung 0->8
plt.plot(X, Y, "ro", color="Blue")  # Cac diem co mau blue
plt.xlabel("Gia tri thuoc tinh x")
plt.ylabel("Gia tri thuoc tinh y") """
""" plt.show() """

# Tim hoi quy voi theta0 = 0, theta1 = 1, toc do hoc = 0.2 va so lan lap = 1


def LR1(X, Y, eta, lanlap, theta0, theta1):
    m = len(X)  # Gan m = do dai tap X
    for i in range(0, lanlap):  # i chay tu 0 den lanlap - 1
        """  print("Lan lap thu:", i) """
        for j in range(0, m):  # j chay tu 0 den do dai tap X - 1
            h_j = theta0 + theta1 * X[j]
            # Tinh theta0
            theta0 = theta0 + eta*(Y[j]-h_j)
            """  print("Phan tu", j, ",", "y=", Y[j], ",", "h =", h_j, ",",
                  "Gia tri cua theta0 =", round(theta0, 3)) """
            # Tinh theta1
            theta1 = theta1 + eta*(Y[j] - h_j) * X[j]
            """ print("Phan tu", i, ",", "Gia tri cua theta1 =", round(theta1, 3)) """
    return [round(theta0, 3), round(theta1, 3)]


""" theta = LR1(X, Y, 0.2, 2, 0, 1) """
""" print(theta) """
""" theta2 = LR1(X, Y, 0.1, 2, 0, 1) """
""" print(theta2) """

# Ve duong hoi quy
""" X1 = np.array([1, 6])
X2 = np.array([1, 6])
Y1 = theta[0] + theta[1] * X1
Y2 = theta2[0] + theta2[1] * X2 """

""" plt.axis([0, 7, 0, 10])
plt.plot(X, Y, "ro", color="blue")

plt.plot(X1, Y1, color="violet")
plt.plot(X2, Y2, color="green")

plt.xlabel("Gia tri cua thuoc tinh x")
plt.ylabel("Gia tri cua thuoc tinh y") """
""" plt.show() """


# Du bao
""" y1 = theta[0] + theta[1] * 0
y2 = theta[0] + theta[1] * 3
y3 = theta[0] + theta[1] * 5 """
""" 
print("Tap du bao y1:", y1)
print("Tap du bao y2:", y2)
print("Tap du bao y3:", y3)
 """

#  -------------------------------------------------------------------------------------

# Doc file du lieu
dt = pd.read_csv("Housing_2019.csv", index_col=0)
""" print(dt) """

dt_X = dt.iloc[:, [1, 2, 3, 4, 10]]
dt_Y = dt.price

""" print(dt_Y) """

# Hiển thị dữ liệu tương quan giữa diện tích (lotsize) và giá nhà (price)
plt2.scatter(dt.lotsize, dt.price)
""" plt2.show() """

# Huan luyen mo hinh

lm = linear_model.LinearRegression()
lm.fit(dt_X[0:520], dt_Y[0:520])

""" print("Theta0:", lm.intercept_)
print("Theta1-5:", lm.coef_) """

# Du bao gia nha cho 20 phan tu cuoi cung
y_test = dt_Y[-20:]
x_test = dt_X[-20:]
y_pred = lm.predict(x_test)

""" print(y_pred) """

# So sanh gia tri thuc te voi du bao
err = mean_squared_error(y_test, y_pred)

""" print(err) """

rmse = np.sqrt(err)

""" print(round(rmse, 3)) """

# ------------------------------------------------------------------------------------#
dt_X = dt.iloc[:, [1, 2, 4, 10]]
dt_Y = dt.price
bag_X_train, bag_X_test, bag_Y_train, bag_Y_test = train_test_split(
    dt_X, dt_Y, test_size=1/3, random_state=100)

len(bag_X_train)
""" print(len(bag_X_train)) """

tree = DecisionTreeRegressor(random_state=0)

baggingtree = BaggingRegressor(
    base_estimator=tree, n_estimators=10, random_state=42)
baggingtree.fit(bag_X_train, bag_Y_train)

bag_y_pred = baggingtree.predict(bag_X_test)

""" err_bag = mean_squared_error(bag_Y_test, bag_y_pred)

print(err_bag)
print(round(np.sqrt(err_bag), 3)) """

# Dự đoán giá nhà (bài toán hồi quy) với hồi quy tuyến tính

lm2 = linear_model.LinearRegression()

bagging_reg = BaggingRegressor(
    base_estimator=lm2, n_estimators=10, random_state=42)
bagging_reg.fit(bag_X_train, bag_Y_train)

bagging_reg_y_pred = bagging_reg.predict(bag_X_test)

err_bag = mean_squared_error(bag_Y_test, bag_y_pred)

print(err_bag)
print(round(np.sqrt(err_bag), 3))
