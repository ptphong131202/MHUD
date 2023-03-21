import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Tập dữ liệu
X = np.array([150, 147, 150, 153, 158, 163, 165,
             168, 170, 173, 175, 178, 180, 183])
Y = np.array([50, 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# Tính các giá trị cần thiết cho đường hồi quy
n = len(X)
mean_X = np.mean(X)  # trung bình tập x
mean_Y = np.mean(Y)  # trung bình tap y
# tổng bình phương độ lệch chéo giữa X và Y, tính bằng cách lấy tổng của (X[i]-mean_X)*(Y[i]-mean_Y)
SS_xy = np.sum(X*Y) - n*mean_X*mean_Y
# tổng bình phương độ lệch của X, tính bằng cách lấy tổng của (X[i]-mean_X)^2
SS_xx = np.sum(X*X) - n*mean_X*mean_X
theta1 = SS_xy / SS_xx
theta0 = mean_Y - theta1*mean_X

# Vẽ đường hồi quy
x_plot = np.linspace(np.min(X), np.max(X), 100)
y_plot = theta0 + theta1*x_plot
plt.plot(x_plot, y_plot, color='red')

# Vẽ dữ liệu
plt.scatter(X, Y)

# Đặt tiêu đề và tên trục
plt.title('Biểu đồ phân tán và đường hồi quy')
plt.xlabel('Chiều cao (cm)')
plt.ylabel('Cân nặng (kg)')

# Hiển thị biểu đồ
plt.show()


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
