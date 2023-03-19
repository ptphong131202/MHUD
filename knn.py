import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton
import pandas as pd
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
# đọc file winequality-white.csv lưu vào biến data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.data'
names = ['date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged', 'severity', 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo', 'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf', 'leaf-mild',
         'stem', 'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external decay', 'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'fruit spots', 'seed', 'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots']
data = pd.read_csv(url, names=names)

# Xóa cột 'date' vì nó không có tác động đến kết quả dự đoán
data = data.drop(['date'], axis=1)

# Chuyển đổi các giá trị trong các cột sang kiểu số nguyên
data = data.apply(lambda x: pd.factorize(x)[0])

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
a = []
b = []
c = []

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

    # Du doan nhan
    y_pred_tree = tree.predict(x_test)
    y_pred_knn = knn.predict(x_test)
    y_pred_bayer = bayer.predict(x_test)

    # Tinh do chinh xac tong the
    a.append(accuracy_score(y_test, y_pred_knn)*100)
    b.append(accuracy_score(y_test, y_pred_bayer)*100)
    c.append(accuracy_score(y_test, y_pred_tree)*100)

    d = confusion_matrix(y_test, y_pred_knn)
    e = confusion_matrix(y_test, y_pred_bayer)
    f = confusion_matrix(y_test, y_pred_tree)

    total_tree += accuracy_score(y_test, y_pred_tree)*100
    total_knn += accuracy_score(y_test, y_pred_knn)*100
    total_bayer += accuracy_score(y_test, y_pred_bayer)*100

a1 = np.array(a).reshape(-1, 1)
b2 = np.array(b).reshape(-1, 1)
c3 = np.array(c).reshape(-1, 1)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Tạo các đối tượng trên giao diện
        self.label_algorithm = QLabel('Select an algorithm:', self)
        self.combobox_algorithm = QComboBox(self)
        self.combobox_algorithm.addItem('KNN')
        self.combobox_algorithm.addItem('Naive Bayes')
        self.combobox_algorithm.addItem('Decision Tree')
        self.button_run = QPushButton('Run', self)
        self.label_result = QLabel('', self)

        # Đặt vị trí và kích thước cho các đối tượng trên giao diện
        self.label_algorithm.setGeometry(20, 20, 120, 30)
        self.combobox_algorithm.setGeometry(150, 20, 120, 30)
        self.button_run.setGeometry(150, 70, 120, 30)
        self.label_result.setGeometry(20, 80, 500, 600)

        # Kết nối sự kiện click của nút Run
        self.button_run.clicked.connect(self.run_algorithm)

        # Thiết lập kích thước cửa sổ và hiển thị giao diện
        self.setGeometry(100, 100, 600, 700)
        self.setWindowTitle('Classification Algorithms')
        self.show()

    def run_algorithm(self):
        # Lấy giá trị của thuật toán từ combobox
        algorithm = self.combobox_algorithm.currentText()

        # Chạy thuật toán và lấy kết quả
        result = self.run_algorithm_function(algorithm)

        # Hiển thị kết quả trên label
        self.label_result.setText(f'{algorithm} result:\n {result},\n {d}')

    def run_algorithm_function(self, algorithm):
        # Thực hiện thuật toán và trả về kết quả ở đây
        # Ví dụ: Trả về kết quả là "accuracy: 0.85"
        if algorithm == 'KNN':
            return a1
        elif algorithm == 'Naive Bayes':
            return b2
        elif algorithm == 'Decision Tree':
            return c3


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
