from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


class Classifier:

    def __init__(self):
        # Load data
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.data'

        names = ['date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged', 'severity', 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo', 'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf', 'leaf-mild',
                 'stem', 'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external decay', 'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'fruit spots', 'seed', 'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots']
        data = pd.read_csv(url, names=names)

        # Xóa cột 'date' vì nó không có tác động đến kết quả dự đoán
        data = data.drop(['date'], axis=1)

        # Chuyển đổi các giá trị trong các cột sang kiểu số nguyên
        data = data.apply(lambda x: pd.factorize(x)[0])
        # Số phần tử Mảng

        # Sử dụng nghi thức K-Fold
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

    def run_knn(self):
        # Fit KNN classifier to training data
        self.knn.fit(self.X_train, self.y_train)

        # Make predictions on test data
        y_pred = self.knn.predict(self.X_test)

        # Calculate accuracy score
        score = accuracy_score(self.y_test, y_pred)

        return score

    def run_nb(self):
        # Fit Naive Bayes classifier to training data
        self.nb.fit(self.X_train, self.y_train)

        # Make predictions on test data
        y_pred = self.nb.predict(self.X_test)

        # Calculate accuracy score
        score = accuracy_score(self.y_test, y_pred)

        return score

    def run_dt(self):
        # Fit Decision Tree classifier to training data
        self.dt.fit(self.X_train, self.y_train)

        # Make predictions on test data
        y_pred = self.dt.predict(self.X_test)

        # Calculate accuracy score
        score = accuracy_score(self.y_test, y_pred)

        return score
