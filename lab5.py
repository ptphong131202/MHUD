from sklearn.preprocessing import scale
import pandas as pd
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

""" data = list(zip(x, y)) """

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
plt.show() 


linkage_data = linkage(data, method="ward", metric="euclidean")
dendrogram(linkage_data)

plt.show() """


# ---------------------------------------------------------------------------------------->

url = 'https://raw.githubusercontent.com/ltdaovn/dataset/master/USArrests.csv'
df = pd.read_csv(url, index_col=0)

df = pd.DataFrame(scale(df), index=df.index, columns=df.columns)

# Tìm số lượng cụm tối ưu
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df)
    inertials.append(kmeans.inertia_)
plt.plot(range(1, 11), inertials, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# Áp dụng thuật toán k-means với số lượng cụm tối ưu
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df)

# Hiển thị kết quả phân cụm
df['cluster'] = kmeans.labels_
print(df.groupby('cluster').mean())
