from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
""" 
# Tập dữ liệu
X = np.array([150, 147, 150, 153, 158, 163, 165,
             168, 170, 173, 175, 178, 180, 183])
Y = np.array([90, 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# Tính các giá trị cần thiết cho đường hồi quy
n = len(X)
mean_X = np.mean(X)
mean_Y = np.mean(Y)
SS_xy = np.sum(X*Y) - n*mean_X*mean_Y
SS_xx = np.sum(X*X) - n*mean_X*mean_X
b_1 = SS_xy / SS_xx
b_0 = mean_Y - b_1*mean_X

# Vẽ biểu đồ scatter
plt.scatter(X, Y)

# Vẽ đường hồi quy
x_plot = np.linspace(np.min(X), np.max(X), 100)
y_plot = b_0 + b_1 * x_plot
plt.plot(x_plot, y_plot, color='red')

# Đặt tên cho trục x và trục y
plt.xlabel('Chiều cao (cm)')
plt.ylabel('Cân nặng (kg)')

plt.show()
 """

""" 
# Đọc dữ liệu từ tập tin
df = pd.read_csv('wood.csv')

# Vẽ biểu đồ scatter plot
sns.scatterplot(x='x', y='y', data=df)

# Xây dựng mô hình hồi quy tuyến tính
X = df[['x']]
y = df['y']
model = LinearRegression()
model.fit(X, y)

# Tính toán hệ số hồi quy
slope = model.coef_[0]
intercept = model.intercept_
print('Hệ số hồi quy:', slope, intercept)

# Vẽ đường hồi quy trên biểu đồ scatter plot
x_line = np.linspace(1, 15, 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, color='red')
plt.show() """


# Load data
wine_data = pd.read_csv('winequality-white.csv')
print(wine_data)

X = wine_data.iloc[:, 0:11]
Y = wine_data.quality
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42)


# Train a random forest regression model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


# Evaluate the model on the test set
y_pred = rf.predict(X_test)
r2_score(y_test, y_pred)


# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform a grid search to find the best parameters
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluate the best model on the test set
y_pred = grid_search.predict(X_test)
print(r2_score(y_test, y_pred))
