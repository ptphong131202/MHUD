import matplotlib.pyplot as plt
import numpy as np

# Tạo dữ liệu giả định
x = np.array(["Tree", "KNN", "Bayes"])  # Đối tượng của biểu đồ
y = np.random.randint(10, size=(3, 1))  # Dữ liệu giả cho các thuật toán

# Tạo một mảng các vị trí của các cột
ind = np.arange(len(x))

# Điều chỉnh độ rộng của các cột
width = 0.2

# Vẽ biểu đồ cột
fig, ax = plt.subplots()
rects1 = ax.bar(ind - width, y[:, 0], width, label='Lần 1')
rects2 = ax.bar(ind, y[:, 1], width, label='Lần 2')
rects3 = ax.bar(ind + width, y[:, 2], width, label='Lần 3')

# Đặt tiêu đề và chú thích
ax.set_title('Biểu đồ cột với các thuật toán')
ax.set_xticks(ind)
ax.set_xticklabels(x)
ax.legend()

# Lặp lại 10 lần
for i in range(10):
    # Tùy chỉnh các giá trị của y ở đây
    y = np.random.randint(10, size=(3, 3))
    rects1.set_height(y[:, 0])
    rects2.set_height(y[:, 1])
    rects3.set_height(y[:, 2])
    plt.show()
