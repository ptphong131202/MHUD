from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import confusion_matrix


class ClassifierGUI(QMainWindow):

    def __init__(self):
        super().__init__()

        # Load data
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.data'
        names = ['date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged', 'severity', 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo', 'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf', 'leaf-mild',
                 'stem', 'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external decay', 'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'fruit spots', 'seed', 'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots']
        data = pd.read_csv(url, names=names)

        # Xóa cột 'date' vì nó không có tác động đến kết quả dự đoán
        data = data.drop(['date'], axis=1)

        # Chuyển đổi các giá trị trong các cột sang kiểu số nguyên
        data = data.apply(lambda x: pd.factorize(x)[0])

        # Chia làm 10 phần rồi xáo trộn
        self.X = data.drop(['roots'], axis=1)
        self.y = data['roots']

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2)

        # Create KNN classifier
        self.knn = KNeighborsClassifier()

        # Create Naive Bayes classifier
        self.nb = GaussianNB()

        # Create Decision Tree classifier
        self.dt = DecisionTreeClassifier()

        # Set up main window
        self.setWindowTitle("Classifier Results")
        self.resize(500, 500)
        self.knn_button = QPushButton("KNN", self)
        self.knn_button.move(10, 10)
        self.knn_button.clicked.connect(self.run_knn)
        self.nb_button = QPushButton("Naive Bayes", self)
        self.nb_button.move(10, 40)
        self.nb_button.clicked.connect(self.run_nb)

        self.dt_button = QPushButton("Decision Tree", self)
        self.dt_button.move(10, 70)
        self.dt_button.clicked.connect(self.run_dt)

        self.result_label = QLabel(self)
        self.result_label.move(10, 100)
        label_size = self.result_label.sizeHint()
        self.result_label.resize(label_size)

    def run_knn(self):
        model = self.knn.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        matrix = confusion_matrix(self.y_test, y_pred)
        self.result_label.setText(
            f"Du doan: {y_pred}\nconfusion_matrix: \n {matrix}\nAccuracy: {accuracy:.2f}")
        self.result_label.adjustSize()

    def run_nb(self):
        model = self.nb.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        matrix = confusion_matrix(self.y_test, y_pred)
        self.result_label.setText(
            f"Du doan: {y_pred}\nconfusion_matrix: \n {matrix}\nAccuracy: {accuracy:.2f}")
        self.result_label.adjustSize()

    def run_dt(self):
        model = self.dt.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        matrix = confusion_matrix(self.y_test, y_pred)
        self.result_label.setText(
            f"Du doan: {y_pred}\nconfusion_matrix: \n {matrix}\nAccuracy: {accuracy:.2f}")
        self.result_label.adjustSize()


if __name__ == '__main__':
    app = QApplication([])
    window = ClassifierGUI()
    window.show()
    app.exec_()
