import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
# Cau 1:
# a. Đọc dữ liệu từ tập dữ liệu đánh giá chất lượng rượu vang trắng trên trang UCI
dulieu = pd.read_csv("winequality-white.csv", delimiter=";")
a = dulieu.head(0)

# b.
# Dữ liệu có bao nhiêu thuộc tính?
print("So thuoc tinh: ", a.columns.size)
dulieu.info()

""" So thuoc tinh:  12
Cac thuoc tinh:  fixed acidity, volatile acidity, 
citric acid, residual sugar, chlorides, free sulfur dioxide,
total sulfur dioxide, density, pH, sulphates, alcohol, quality"""

# Cột nào là cột nhãn?
print("Cot nhan la: ", a.columns[11])
# Giá trị của các nhãn?
print(a.columns[11])
""" hien thi 10 dong dau """
print(dulieu.quality.head(10))
# c. Với tập dữ liệu wineWhite sử dụng nghi thức K-Fold để phân chia tập dữ liệu huấn
# luyện với K=50, sử dụng tham số “Shuffle” để xáo trộn tập dữ liệu trước khi phân
# chia. Xác định số lượng phần tử có trong tập test và tập huấn luyện nếu sử dụng
# nghi thức đánh giá này.
kf = KFold(n_splits=50, shuffle=True)
for train_index, test_index in kf.split(dulieu):
    # kiem tra tap train va test
    """ print("Train: ", test_index, "Test: ", test_index) """
    X_train, X_test = dulieu.iloc[train_index, ], dulieu.iloc[test_index, ]
    y_train, y_test = dulieu.iloc[train_index, ], dulieu.iloc[test_index, ]
    # kiem tra X_test hold_out X_test K F
    print("X_test: ", X_test.size, "x_test: ", X_test.size)

# d. Xây dựng mô hình cây quyết định dựa trên tập dữ liệu học tạo ra ở bước c.
# Xây dựng mô hình cây quyết định dựa trên chỉ số Gini với độ sâu của cây bằng 3, nút nhánh ít nhất có 5 phần tử.


# e. Đánh giá độ chính xác cho từng phân lớp dựa vào giá trị dự đoán của câu 3 cho mỗi
# lần lặp. Chép lại kết quả độ chính xác cho từng phân lớp của lần lặp cuối nộp lại
# (có thể đưa vào comment trong file code).
""" y_pred = clf_Gini.predict(X_test)
print(y_pred) 

print("accuracy_score is: ", accuracy_score(y_test, y_pred) * 100) """
