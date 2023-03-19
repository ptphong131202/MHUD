
import pandas as pd
dt = pd.read_csv("winequality-white.csv", delimiter=";")

#-----------------------------------------------------------------------
# Co 12 thuoc thinh
# Cot nhan la quality
print("\nSo phan tu:", len(dt), "phan tu")    #4898
import numpy as np
print("\nCac nhan khac nhau:", np.unique(dt.quality))
# Cac nhan khac nhau: [3 4 5 6 7 8 9]
print(dt.quality.value_counts())
# 6    2198
# 5    1457
# 7     880
# 8     175
# 4     163
# 3      20
# 9       5
# Sử dụng nghi thức K-Fold
from sklearn.model_selection import KFold
kf = KFold(n_splits=50, shuffle=True, random_state=3000)
x = dt.iloc[:,0:11]
y = dt.quality

# Xa dung mo hinh
from sklearn.tree import DecisionTreeClassifier
cayquyetdinh = DecisionTreeClassifier(criterion = "entropy", random_state = 10, max_depth = 7, min_samples_leaf = 5)

# Phan chia tap du lieu && #Danh gia mo hinh
from sklearn.metrics import confusion_matrix
i = 0
for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index,], x.iloc[test_index,]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print("-------------------------------")
    print("X_test:", len(x_test))
    print("X_train:", len(x_train))
    cayquyetdinh.fit(x_train, y_train)

    # Du doan nhan
    y_pred = cayquyetdinh.predict(x_test)

    # Tinh do chinh xac tren tung phan lop
    ls = confusion_matrix(y_test, y_pred, labels=[3,4,5,6,7,8,9])
    i = i+1
    print("\nDo chinh xac tung phan lop lan lap", i, ":\n", ls)

# Do chinh xac tung phan lop:
# [[ 0  0  0  0  0  0  0]
#  [ 0  0  1  0  0  0  0]
#  [ 0  0 10 14  0  0  0]
#  [ 0  0 12 30  1  0  0]
#  [ 0  0  1 15  9  2  0]
#  [ 0  0  0  0  0  1  0]
#  [ 0  0  0  0  1  0  0]]


# Tính độ chính xác tổng thể cho mỗi lần lặp và độ chính xác tổng thể của trung bình 50 lần lặp
from sklearn.metrics import accuracy_score

i = 0
total = 0
for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index,], x.iloc[test_index,]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print("-------------------------------")
    cayquyetdinh.fit(x_train, y_train)

    # Du doan nhan
    y_pred = cayquyetdinh.predict(x_test)

    # Tinh do chinh xac tong the
    i = i+1
    acc = accuracy_score(y_test, y_pred)*100
    print("Do chinh xac tong the lap " , i , ":", acc, "%")
    if (i <= 50):
        total = total + acc

print("\nDo chinh xac trung binh cua 50 lan lap:", total/50, "%") #53.162634125815266 %

# so sánh hiệu quả phân lớp của giải thuật cây quyết định với nghi thức đánh giá k-fold với K=60
# KNN vs Bayer vs Tree
import pandas as pd
dt = pd.read_csv("winequality-white.csv", delimiter=";")

## Nghi thuc K-Fold
from sklearn.model_selection import KFold
kf = KFold(n_splits=60, shuffle=True, random_state=3000)
x = dt.iloc[:,0:11]
y = dt.quality

# Tree 
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = "entropy", random_state = 10, max_depth = 7, min_samples_leaf = 5)
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
# Bayer
from sklearn.naive_bayes import GaussianNB
bayer = GaussianNB()

# Huan luyen
from sklearn.metrics import confusion_matrix
i = 0
total_tree = 0
total_knn = 0
total_bayer = 0
for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index,], x.iloc[test_index,]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    i = i+1
    print("---------------------------------------------------------",i)
    tree.fit(x_train, y_train)
    knn.fit(x_train, y_train)
    bayer.fit(x_train, y_train)

    # Du doan nhan
    y_pred_tree = tree.predict(x_test)
    y_pred_knn = knn.predict(x_test)
    y_pred_bayer = bayer.predict(x_test)

    # Tinh do chinh xac tong the
    from sklearn.metrics import accuracy_score
    print("\nDo chinh xac tong the Tree:", accuracy_score(y_test, y_pred_tree)*100, "%")
    print("\nDo chinh xac tong the KNN:", accuracy_score(y_test, y_pred_knn)*100, "%")
    print("\nDo chinh xac tong the Bayes:", accuracy_score(y_test, y_pred_bayer)*100, "%")

    total_tree = total_tree + accuracy_score(y_test, y_pred_tree)*100
    total_knn = total_knn + accuracy_score(y_test, y_pred_knn)*100
    total_bayer = total_bayer + accuracy_score(y_test, y_pred_bayer)*100

print("\n\n---------------------------------------------------------------")
print("Do chinh xac trung binh Tree:     ", total_tree/i , "%")
print("\nDo chinh xac trung binh KNN (k=7):", total_knn/i , "%")
print("\nDo chinh xac trung binh Bayer:    ", total_bayer/i , "%")
print("---------------------------------------------------------------")
