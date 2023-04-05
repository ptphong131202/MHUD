from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
x = np.array([4, 5, 10, 4, 3, 11, 14, 6, 10, 12])
y = np.array([21, 19, 24, 17, 16, 25, 24, 22, 21, 21])

""" plt.scatter(x, y)
plt.xlabel("Diem van")
plt.ylabel("Diem toan")
plt.show() """

data = list(zip(x, y))

inertials = []

""" for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertials.append(kmeans.inertia_) """

""" plt.plot(range(1, 11), inertials, marker='o')
plt.title("Eblow method")
plt.xlabel("Number of cluter")
plt.ylabel("Inertia")
plt.show()
 """

""" kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.title("Eblow method")
plt.xlabel("Number of cluter")
plt.ylabel("Inertia")
plt.show() """


linkage_data = linkage(data, method="ward", metric="euclidean")
dendrogram(linkage_data)

plt.show()
