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
plt.show()


# tim hoi quy
def LR1(X, Y, eta, lanlap, theta0, theta1):
    m = len(X)
    for k in range(0, )
