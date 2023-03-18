import numpy as np 
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# doc file
df = pd.read_csv("full_data.csv", delimiter=",")

print("\n\tTAP DU LIEU TRUOC KHI CHUYEN DOI\n")
print(df)

df['gender'] = df['gender'].map({'Male':1,'Female':0})

df['ever_married'] = df['ever_married'].map({'Yes':1,'No':0})

df['Residence_type'] = df['Residence_type'].map({'Urban':1,'Rural':0})

df['work_type'] = df['work_type'].map({'children':0,'Private':1,'Self-employed':2,'Govt_job':3})

df['smoking_status'] = df['smoking_status'].map({'Unknown':-1,'never smoked':0,'formerly smoked':1,'smokes':2})

print("\n\tTAP DU LIEU SAU KHI CHUYEN DOI\n")
print(df)

# y la nhan, x la cot con lai
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Tinh nhãn
# print("\n\tDAY LA NHAN\n")
# print("array labels ", np.unique(df.gender))
# print(df.gender.value_counts(), "\n")

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3.0, random_state=5)

acc_score = 0
max_random_state = 0
max_max_depth = 0
max_min_samples_leaf = 0
test_min_samples_leaf = 2

for test_random_state in range(50, 61, 1):
    for test_max_depth in range(1, 11, 1):
        # for test_min_samples_leaf in range(3, 10, 1):
        tree = DecisionTreeClassifier(criterion="entropy", random_state=test_random_state,
                                      max_depth=test_max_depth, min_samples_leaf=test_min_samples_leaf)

        # huan luyen
        tree.fit(x_train, y_train)

        # du doan
        y_pred_tree = tree.predict(x_test)

        # do chinh xac tong the
        acc = accuracy_score(y_test, y_pred_tree)*100  # tree
        acc = round(acc, 4)

        if (acc > acc_score):
            max_random_state = test_random_state
            max_max_depth = test_max_depth
            max_min_samples_leaf = test_min_samples_leaf
            acc_score = acc

        print(test_random_state, test_max_depth, test_min_samples_leaf, acc)

print("STOPPED.\n")

print("do chinh xac tong the cao nhat voi cac gia tri la:")

print("random state:", max_random_state, "max depth:", max_max_depth,
      "min samples leaf:", test_min_samples_leaf, "do chinh xac cao nhat la:", acc_score)

# do chinh xac sau 10 lan lap
sum_knn = 0
sum_bayes = 0
sum_tree = 0

for b in range(1, 11):
    print("LAN LAP ", b, "VOI random_state = ", str(b*5))

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=1/3.0, random_state=b*5)

    print("train:", len(x_train), "test:", len(x_test))

    # TREE
    cayquyetdinh = DecisionTreeClassifier(
        criterion="entropy", random_state=max_random_state, max_depth=max_max_depth, min_samples_leaf=test_min_samples_leaf)
    cayquyetdinh.fit(x_train, y_train)  # huan luyen
    dudoan_tree = cayquyetdinh.predict(x_test)  # du doan
    ptr1 = accuracy_score(y_test, dudoan_tree)*100

    print("do chinh xac tree lan lap thu ", b, "la\n", confusion_matrix(y_test,
          dudoan_tree, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    
    print("do chinh xac tung phan lop:\n",
          classification_report(y_test, dudoan_tree, zero_division=0))

    # BAYES
    # 1. xay dung mo hinh dua tren phan phoi xac suat tuan theo Gaussian
    model = GaussianNB()
    model.fit(x_train, y_train)

    # 2. du doan
    y_test
    dudoan_bayes = model.predict(x_test)
    ptr2 = accuracy_score(y_test, dudoan_bayes)*100

    # KNN
    # 1. xay dung mo hinh k lang gieng knn voi 5 la gieng
    
    Mohinh_KNN = KNeighborsClassifier(n_neighbors=5)
    Mohinh_KNN.fit(x_train, y_train)

    # 2. du doan
    dudoan_knn = Mohinh_KNN.predict(x_test)

    # 3. tinh do chinh xac tong the
    ptr3 = accuracy_score(y_test, dudoan_knn)*100

    # 4. do chinh xac tong the sau 10 lan lap
    sum_knn += ptr3
    sum_tree += ptr1
    sum_bayes += ptr2

    print("do chinnh xac tong the tree:", round(ptr1, 4))
    print("do chinnh xac tong the bayes:", round(ptr2, 4))
    print("do chinnh xac tong the knn:", round(ptr3, 4))
    print("\n------------------------------>\n")

print("do chinh xac tong the trung binh cua knn sau 10 lan lap la:",
      round(sum_knn/10, 4))
print("do chinh xac tong the trung binh cua bayes sau 10 lan lap la:",
      round(sum_bayes/10, 4))
print("do chinh xac tong the trung binh cua tree sau 10 lan lap la:",
      round(sum_tree/10, 4))
