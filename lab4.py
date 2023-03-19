import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# hai mang x va y
X = np.array([1, 2, 4])
Y = np.array([2, 3, 6])

# truc x = 0 -> 5
# truc y = 0 -> 8
plt.axis([0, 5, 0, 8])
plt.plot(X, Y, "ro", color="blue")
plt.xlabel("Gia tri thuoc tinh X")
plt.ylabel("Gia tri thuoc tinh Y")
""" plt.show() """


# tim hoi quy
def LR1(X, Y, eta, lanlap, theta0, theta1):
    m = len(X)
    for k in range(0, lanlap):
        print("Lan lap: ", k)
        for i in range(0, m):
            h_i = theta0 + theta1*X[i]
            theta0 += eta*(Y[i] - h_i) * 1
            """  print("Phan tu ", i, "y = ",
                  Y[i], "h = ", h_i, "gia tri theta0 = ", round(theta0, 3)) """
            theta1 += eta*(Y[i] - h_i) * X[i]
            """ print("Phan tu ", i, "Gia tri theta1 = ", round(theta1, 3)) """
    return [round(theta0, 3), round(theta1, 3)]


theta = LR1(X, Y, 0.2, 2, 0, 1)
""" print(theta) """
theta2 = LR1(X, Y, 0.1, 2, 0, 1)
""" print(theta2) """


x1 = np.array([1, 6])
y1 = theta[0] + theta[1]*x1

x2 = np.array([1, 6])
y2 = theta2[0] + theta2[1]*x2

plt.axis([0, 7, 0, 10])
plt.plot(X, Y, "ro", color="blue")
plt.plot(x1, y1, color="violet")
plt.plot(x2, y2, color="green")

plt.xlabel("Gia tri thuoc tinh X")
plt.ylabel("Gia tri thuoc tinh Y")

""" plt.show() """

# du bao

y1 = theta[0] + theta[1]*0
y2 = theta[0] + theta[1]*3
y3 = theta[0] + theta[1]*5
""" 
print("X = 0 -> Y = ", y1)
print("X = 3 -> Y = ", y2)
print("X = 5 -> Y = ", y3) """


print("################################################################")
