# import thư viện GaussianNB dể huấn luyện mô hình theo bayes
from sklearn.naive_bayes import GaussianNB
# import thư viện KNeighborsClassifier để huấn luyện mô hình theo KNN
from sklearn.neighbors import KNeighborsClassifier
# import thư viện accuracy_score tính trung bình tổng thể
from sklearn.metrics import accuracy_score
# import thư viện confusion_matrix để hiển thị độ chính xác mô hình
from sklearn.metrics import confusion_matrix
# import thư viện DecisionTreeClassifier tạo cây quyết định
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold  # import thư viện K flod
from sklearn.model_selection import train_test_split
import numpy as np  # import thư viện numpy
import pandas as pd  # import thư vien pandas
# đọc file winequality-white.csv lưu vào biến data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.data'
names = ['date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged', 'severity', 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo', 'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf', 'leaf-mild',
         'stem', 'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external decay', 'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'fruit spots', 'seed', 'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots']
data = pd.read_csv(url, names=names)

# Xóa cột 'date' vì nó không có tác động đến kết quả dự đoán
data = data.drop(['date'], axis=1)

# Chuyển đổi các giá trị trong các cột sang kiểu số nguyên
data = data.apply(lambda x: pd.factorize(x)[0])

""" print(data) """
# Số phần tử Mảng
print("######################################################################################")

print("So phan tu:", len(data), "phan tu")
# các giá trị trong mảng
print("Cac loai phan tu trong nhan:", np.unique(
    data.roots))
# số lượng mỗi phần tử
print(data.roots.value_counts())

print("######################################################################################\n")

# Sử dụng nghi thức K-Fold
# Chia làm 10 phần rồi xáo trộn
kf = KFold(n_splits=10, shuffle=True, random_state=3000)
x = data.drop(['roots'], axis=1)
y = data['roots']


# Tree
tree = DecisionTreeClassifier(
    criterion="entropy", random_state=10, max_depth=7, min_samples_leaf=5)
# KNN
knn = KNeighborsClassifier(n_neighbors=5)
# Bayer
bayer = GaussianNB()

# Huan luyen
i = 0
total_tree = 0
total_knn = 0
total_bayer = 0

for train_index, test_index in kf.split(x):
    """ for i in range(0, 10): """
    """x_train, x_test, y_train, y_test = train_test_split("""
    """x, y, test_size=1/3, random_state=5) """
    x_train, x_test = x.iloc[train_index,], x.iloc[test_index,]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    i = i+1
    tree.fit(x_train, y_train)
    knn.fit(x_train, y_train)
    bayer.fit(x_train, y_train)
    print("---------------------------------------------------------")
    print("\nLần lập thứ ", i)

    # Du doan nhan
    y_pred_tree = tree.predict(x_test)
    y_pred_knn = knn.predict(x_test)
    y_pred_bayer = bayer.predict(x_test)

    # Tinh do chinh xac tong the
    print("\nDo chinh xac tong the Tree:",
          accuracy_score(y_test, y_pred_tree)*100, "%")
    print("\nDo chinh xac tong the KNN:",
          accuracy_score(y_test, y_pred_knn)*100, "%")
    print("\nDo chinh xac tong the Bayes:",
          accuracy_score(y_test, y_pred_bayer)*100, "%")

    total_tree += accuracy_score(y_test, y_pred_tree)*100
    total_knn += accuracy_score(y_test, y_pred_knn)*100
    total_bayer += accuracy_score(y_test, y_pred_bayer)*100

print("\n\n---------------------------------------------------------------")
print("Do chinh xac trung binh Tree:     ", total_tree/i, "%")
print("\nDo chinh xac trung binh KNN (k=5):", total_knn/i, "%")
print("\nDo chinh xac trung binh Bayer:    ", total_bayer/i, "%")
print("---------------------------------------------------------------")
