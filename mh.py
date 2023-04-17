import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.tree import plot_tree

names = ["class", 'date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged',
         'severity', 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo',
         'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf', 'leaf-mild',
         'stem', 'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external decay',
         'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'fruit spots', 'seed', 'mold-growth',
         'seed-discolor', 'seed-size', 'shriveling', 'roots']
df = pd.read_csv("soybean-large.csv", names=names)

print("\n\tTAP DU LIEU TRUOC KHI CHUYEN DOI\n")
print(df)


df = df.drop(['date'], axis=1)
# Thay thế giá trị '?' bằng 0
df.replace('?', 0, inplace=True)


df['class'] = df['class'].map({'diaporthe-stem-canker': 0, 'charcoal-rot': 1,
                               'rhizoctonia-root-rot': 2, 'phytophthora-rot': 3,
                               'brown-stem-rot': 4, 'powdery-mildew': 5,
                               'downy-mildew': 6, 'brown-spot': 7,
                               'bacterial-blight': 8, 'bacterial-pustule': 9,
                               'purple-seed-stain': 10, 'anthracnose': 11,
                               'phyllosticta-leaf-spot': 12, 'alternarialeaf-spot': 13,
                               'frog-eye-leaf-spot': 14, 'diaporthe-pod-&-stem-blight': 15,
                               'cyst-nematode': 16, '2-4-d-injury': 17,
                               'herbicide-injury': 18})

print("\n\tTAP DU LIEU SAU KHI CHUYEN DOI\n")
print(df)
print("------------------------------------------------------------------------------------>\n")

# y la nhan, x la cot con lai
X = df.drop(['class'], axis=1)
y = df['class']


df.info()
# Tinh nhãn
print("\n<-----NHAN----->\n")
print("array labels ", np.unique(df['class']))
print(df['class'].value_counts(), "\n")

plt.pie(df['class'].value_counts(), labels=[0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], autopct='%1.1f%%')
plt.axis('equal')
plt.title("Đồ thị thể hiện phần trăm số lượng của các nhãn.")
plt.show()


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
dcxknn = []
dcxbayes = []
dcxtree = []

f1_scores = []
for b in range(1, 11):  # kiểm tra với 10 lần lập
    print("LAN LAP ", b, "VOI random_state = ", str(b*5))

    # chia tập dữ liệu thành 2 phần 7 train 3 test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=b*5)

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

    # f1_corce
    f1_scores.append(f1_score(y_test, dudoan_tree, average='macro'))

    # độ chính xác tree
    ptr1 = accuracy_score(y_test, dudoan_tree)*100
    print(" - Do chính xac:", ptr1, "%")
    dcxtree.append(round(ptr1, 3))

    # độ chính xác qua ma trân lỗi
    print(" - Do chinh xac tree, lan lap thu ", b, "la:\n", confusion_matrix(y_test,
          dudoan_tree, labels=[0, 1, 2, 3, 4, 5, 6, 7,
                               8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]))

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
    dcxbayes.append(round(ptr2, 3))

    # <------------------------------------------------------------------------------------------------------>

    # KNN
    # 1. xay dung mo hinh k lang gieng knn voi 5 la gieng
    Mohinh_KNN = KNeighborsClassifier(n_neighbors=7)

    # huấn luyện
    Mohinh_KNN.fit(x_train, y_train)

    # 2. du doan
    dudoan_knn = Mohinh_KNN.predict(x_test)

    # 3. tinh do chinh xac tong the knn
    ptr3 = accuracy_score(y_test, dudoan_knn)*100
    dcxknn.append(round(ptr3, 3))

    # 4. do chinh xac tong the knn, bayes, tree
    sum_knn += ptr3
    sum_tree += ptr1
    sum_bayes += ptr2

    # hiện thị độ chính xác qua mỗi lần lập
    print("Do chinnh xac tong the tree:", round(ptr1, 3), "%")
    print("Do chinnh xac tong the bayes:", round(ptr2, 3), "%")
    print("Do chinnh xac tong the knn:", round(ptr3, 3), "%")
    print("\n------------------------------>\n")

lanlap = np.arange(1, 11)

plt.figure(figsize=(20, 20))
plt.bar(lanlap, f1_scores, align="center", width=0.5)


plt.xlabel("Lần lập")
plt.ylabel("F1")
plt.title(
    "Đồ thị thể hiện F1 của thuật toán tree")
for x, y in zip(lanlap, f1_scores):
    plt.text(x+0.02, y + 0.01, "%.2f" % y, ha="center", va="bottom")

plt.show()


# Tạo mảng chứa các điểm trên trục x

# Vẽ biểu đồ đường cho từng thuật toán
plt.figure(figsize=(15, 10))
plt.plot(lanlap, dcxknn, label='KNN')
plt.plot(lanlap, dcxbayes, label='BAYES')
plt.plot(lanlap, dcxtree, label='TREE')
for i in range(len(lanlap)):
    plt.text(lanlap[i], dcxknn[i], f"{dcxknn[i]:.2f}%")
    plt.text(lanlap[i], dcxbayes[i], f"{dcxbayes[i]:.2f}%")
    plt.text(lanlap[i], dcxtree[i], f"{dcxtree[i]:.2f}%")
# Thiết lập tiêu đề và nhãn trục cho biểu đồ
plt.title('Đồ thị thị thể hiện độ chính xác của 3 thuật toán qua 10 lần lập.')
plt.xlabel('X')
plt.ylabel('Y')

# Hiển thị chú thích cho biểu đồ
plt.legend()

# Hiển thị biểu đồ
plt.show()


kq = []
# hiển thị độ chính xác trung bình(làm tròn 3 số) sau 10 lần lặp
print("Do chinh xac tong the trung binh cua knn sau 10 lan lap la:",
      round(sum_knn/10, 3), "%")
kq.append(round(sum_knn/10, 3))
print("Do chinh xac tong the trung binh cua bayes sau 10 lan lap la:",
      round(sum_bayes/10, 3), "%")
kq.append(round(sum_bayes/10, 3))
print("Do chinh xac tong the trung binh cua tree sau 10 lan lap la:",
      round(sum_tree/10, 3), "%")
kq.append(round(sum_tree/10, 3))


lables = np.array(['Knn', 'Bayes', 'Tree'])

in_lables = range(len(lables))
plt.figure(figsize=(10, 7))
plt.bar(in_lables, kq, align="center", width=0.5)
plt.xticks(in_lables, lables)


plt.xlabel("Thuat Toan")
plt.ylabel("Do chinh xac (%)")
plt.title(
    "Đồ thị thể hiện độ chính xác trung bình tổng thể của 3 thuật toán knn, bayes, tree")
for x, y in zip(in_lables, kq):
    plt.text(x+0.02, y + 0.05, "%.2f%%" % y, ha="center", va="bottom")

plt.show()
