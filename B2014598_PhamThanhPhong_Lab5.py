from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
""" print(X) """

X = X.T

X1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
""" print(X1) """

Y = np.array([0, 0, 0, 1])
""" print(Y) """

colormap = np.array(["red", "green"])

""" plt.axis([0, 1.5, 0, 2])
plt.scatter(X[:, 0], X[:, 1], c=colormap[Y], s=150)
plt.xlabel("Gia trin thuoc tinh X1")
plt.ylabel("Gia tri thuoc tinh X2")
plt.show()
 """


def my_preception(X, Y, eta, lanlap):
    n = len(X[0,])
    m = len(X[:, 0])
    print("m:", m, "n: ", n)

    w0 = -2
    w = (0.5, 0.5)
    print("W0: ", w0)
    print("W", w)

    for i in range(0, lanlap):
        print("Lan Lap: ", i + 1)
        for j in range(0, m):
            gx = w0 + sum(X[i, ]*w)
            print("gx:", gx)
            if (gx > 0):
                output = 1
            else:
                output = 0

            w0 = w0+eta*(Y[i]-output)
            w = w + eta*(Y[i] - output) * X[i,]
            print("W0: ", w0)
            print("W: ", w)
    return (np.round(w0, 3), np.round(w, 3))


""" my_preception(X, Y, 0.15, 2) """

# --------------------------------------------------------------------------------------------->
data = pd.read_csv("data_per.csv")
""" print(data) """
X = data.iloc[:, 0:5]
Y = data.Y

""" print(X)
print(Y) """

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0)


net = Perceptron()
net.fit(X_train, y_train)
""" print(net)
print("Trong so thuoc tinh:", net.coef_)
print("Trong so w0:", net.intercept_)
print("So lan lap:", net.n_iter_) """
dubao = net.predict(X_test)
""" print(accuracy_score(y_test, dubao)*100, "%")
print(np.random.rand(5))
print(np.random.randn(5)) """


def my_preception2(X, Y, eta, lanlap):
    m, n = X.shape
    print("m:", m, "n: ", n)
    w0 = np.random.rand()
    w = np.random.rand(n)
    for i in range(0, lanlap):
        print("Lan Lap: ", i + 1)
        cost = 0
        for j in range(0, m):
            x = X[j]
            print("X:", x)
            z = np.dot(w, x) + w
            output = np.sign(z)

            w0 = w0+eta*(Y[j]-output)
            w = w + eta*(Y[j] - output) * x
            print("W0: ", w0)
            print("W: ", w)
            cost += abs(Y[j]-output)
        cost /= m
        if cost == 0:
            break
    return (np.round(w0, 3), np.round(w, 3))


""" my_preception2(X, Y, 0.15, 2) """


# --------------------------------------------------------------------------------------->

iris = load_iris()
X = iris.data
y = iris.target

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=50)

# Huấn luyện mô hình với các giá trị tham số khác nhau
max_iters = [5, 100, 1000]
etas = [0.002, 0.02, 0.2]
for max_iter in max_iters:
    for eta in etas:
        perceptron = Perceptron(max_iter=max_iter, eta0=eta)
        perceptron.fit(X_train, y_train)
        accuracy = perceptron.score(X_test, y_test)
        print("max_iter = {}, eta0 = {}: accuracy = {:.3f}%".format(
            max_iter, eta, accuracy*100))
