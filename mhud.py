import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# đọc file từ 'https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.data'
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.data'
names = ['date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged', 'severity', 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo', 'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf', 'leaf-mild',
         'stem', 'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external decay', 'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'fruit spots', 'seed', 'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots']
df = pd.read_csv(url, names=names)

print("\n\tTAP DU LIEU TRUOC KHI CHUYEN DOI\n")
print(df)

# Xóa cột 'date' vì nó không có tác động đến kết quả dự đoán
df = df.drop(['date'], axis=1)

# Chuyển đổi các giá trị trong các cột sang kiểu số nguyên
df = df.apply(lambda x: pd.factorize(x)[0])


print("\n\tTAP DU LIEU SAU KHI CHUYEN DOI\n")
print(df)

# y la nhan, x la cot con lai
X = df.drop(['roots'], axis=1)
y = df['roots']
print("\n< ------------------------------------------------------------------------------ >")
# Tinh nhãn
print("\n<-----NHAN----->\n")
print("array labels ", np.unique(df.roots))
print(df.roots.value_counts(), "\n")

print("\n< ------------------------------------------------------------------------------ >")

# Chia tập dữ liệu thành 2 phần 7 phần train 3 phần test với random state = 5
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3.0, random_state=5)

acc_score = 0  # độ chính xác tổng thể
max_random_state = 0  # max random_state
max_max_depth = 0  # độ sâu lớn nhất
max_min_samples_leaf = 0  # số nút lá
test_min_samples_leaf = 2  # số nút lá để kiểm tra

# random state trong khoảng 50 - 60 với bước nhảy = 1. Đảm bảo kết quả huấn luyện và kiểm tra ổn định.
for test_random_state in range(50, 61, 1):
    print("Test_Random_state ", test_random_state)
    for test_max_depth in range(1, 11, 1):  # độ sâu cây quyết định max = 10

        # xây dừng cây quyết định dựa trên độ lợi thông tin (entropy)
        tree = DecisionTreeClassifier(criterion="entropy", random_state=test_random_state,
                                      max_depth=test_max_depth, min_samples_leaf=test_min_samples_leaf)

        # huan luyen dựa trên tập train
        tree.fit(x_train, y_train)

        # du doan
        y_pred_tree = tree.predict(x_test)

        # do chinh xac tong the
        acc = accuracy_score(y_test, y_pred_tree)*100  # tree
        acc = round(acc, 3)  # làm tròn

        if (acc > acc_score):  # nếu độ chính xác tổng thể ở random_state - ở độ sâu > độ chính xác tổng thể
            max_random_state = test_random_state  # max random = tại random_state
            max_max_depth = test_max_depth  # độ sâu = tại độ sâu kiểm tra
            max_min_samples_leaf = test_min_samples_leaf  # số nút lá = số nút lá kiểm tra
            acc_score = acc  # gán lại độ chính xác tổng thể

        print("Max_depth:", test_max_depth,
              ". Min_samples_leaf:", test_min_samples_leaf,
              ". Do chinh xac tong the:", acc, "%")
    print("\n------------------------------------------------->")

# độ chính xác cao nhat trong tất cả các lần lập
print("do chinh xac tong the cao nhat voi cac gia tri la:")
print(" - Random state:", max_random_state, "\n - Max depth:", max_max_depth,
      "\n - Min samples leaf:", test_min_samples_leaf, "\n - Do chinh xac cao nhat la:", acc_score, "%")

print("\n< ------------------------------------------------------------------------------ >\n")

print("<-------Kiem tra voi 10 lan lap ------>")
# do chinh xac sau 10 lan lap
sum_knn = 0  # Tổng độ chính xác knn
sum_bayes = 0  # Tổng độ chính xác bayes
sum_tree = 0  # Tổng độ chính xác tree

for b in range(1, 11):  # kiểm tra với 10 lần lập
    print("LAN LAP ", b, "VOI random_state = ", str(b*5))

    # chia tập dữ liệu thành 2 phần 7 train 3 test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=1/3.0, random_state=b*5)

    print(" - Train:", len(x_train), "Test:", len(x_test))

    # <------------------------------------------------------------------------------------------------------>
    # TREE
    # tạo cây quyết định với độ lợi thông tin (entropy)
    cayquyetdinh = DecisionTreeClassifier(
        criterion="entropy", random_state=max_random_state, max_depth=max_max_depth, min_samples_leaf=test_min_samples_leaf)

    # huan luyen
    cayquyetdinh.fit(x_train, y_train)

    # du doan
    dudoan_tree = cayquyetdinh.predict(x_test)

    # độ chính xác tree
    ptr1 = accuracy_score(y_test, dudoan_tree)*100
    print(" - Do chính xac:", ptr1, "%")

    # độ chính xác qua ma trân lỗi
    print(" - Do chinh xac tree, lan lap thu ", b, "la:\n", confusion_matrix(y_test,
          dudoan_tree, labels=[0, 1, 2, 3]))

    # độ chính xác từng lớp
    print(" - Do chinh xac tung phan lop:\n",
          classification_report(y_test, dudoan_tree, zero_division=0))

    # <------------------------------------------------------------------------------------------------------>

    # BAYES
    # 1. xay dung mo hinh dua tren phan phoi xac suat tuan theo Gaussian
    model = GaussianNB()

    # huấn luyện
    model.fit(x_train, y_train)

    # 2. du doan
    dudoan_bayes = model.predict(x_test)

    # độ chính xác của bayes
    ptr2 = accuracy_score(y_test, dudoan_bayes)*100

    # <------------------------------------------------------------------------------------------------------>

    # KNN
    # 1. xay dung mo hinh k lang gieng knn voi 5 la gieng
    Mohinh_KNN = KNeighborsClassifier(n_neighbors=5)

    # huấn luyện
    Mohinh_KNN.fit(x_train, y_train)

    # 2. du doan
    dudoan_knn = Mohinh_KNN.predict(x_test)

    # 3. tinh do chinh xac tong the knn
    ptr3 = accuracy_score(y_test, dudoan_knn)*100

    # 4. do chinh xac tong the knn, bayes, tree
    sum_knn += ptr3
    sum_tree += ptr1
    sum_bayes += ptr2

    # hiện thị độ chính xác qua mỗi lần lập
    print("Do chinnh xac tong the tree:", round(ptr1, 3), "%")
    print("Do chinnh xac tong the bayes:", round(ptr2, 3), "%")
    print("Do chinnh xac tong the knn:", round(ptr3, 3), "%")
    print("\n------------------------------>\n")


# hiển thị độ chính xác trung bình(làm tròn 3 số) sau 10 lần lặp
print("Do chinh xac tong the trung binh cua knn sau 10 lan lap la:",
      round(sum_knn/10, 3), "%")
print("Do chinh xac tong the trung binh cua bayes sau 10 lan lap la:",
      round(sum_bayes/10, 3), "%")
print("Do chinh xac tong the trung binh cua tree sau 10 lan lap la:",
      round(sum_tree/10, 3), "%")
