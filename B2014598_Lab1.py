
''' sử dụng canh lề để bao các khối lệnh của hàm, lớp, hoặc luồng điều khiển.  '''
import pandas as pd
import numpy as np
a = 5
b = 3
if a > b:
    a = a*2 + 3
    b = b - 6
    c = a / b
    print("C =", c)
print("--------------------------------------")

''' Dòng lệnh dài viết trên nhiều dòng sử dụng ký tự \ '''
d = a + b +\
    10*a - b/4 -\
    5 + a * 3
print("D =", d)
print("--------------------------------------")

''' Lệnh if '''
a = 5
b = 3
if a > b:
    print("True")
    print("a = ", a)
else:
    print("False")
    print("b = ", b)
print("--------------------------------------")

''' Lệnh while '''
a = 1
b = 10
while a < b:
    a += 1
    print(a)
print("--------------------------------------")

''' Ham tinh binh phuong 1 so '''


def binhphuong(n):
    return n * n


print("Binh phuong cua ", a, "la:", binhphuong(a))
print("--------------------------------------")

'''Các kiểu dữ liệu '''
a = "Phong"
b = 10
c = 10.5
d = ['Tom', 'Snappy', 'Kitty', 'Jessie', 'Chester']
print(type(a))
print(type(b))
print(type(c))
print(type(d))
print("--------------------------------------")

''' truy xuất các phần tử của list '''
print("List d: ", d)
print("Phan tu thu 3 cua list d la: ", d[3])
print("--------------------------------------")

''' so luong phan tu trong list '''
d = ['Tom', 'Snappy', 'Kitty', 'Jessie', 'Chester']
print("So luong phan tu trong list d la: ", len(d))
print("--------------------------------------")

''' ham max, min trong list '''
e = [1, 2, 3, 4, 5, 6]
print("Gia tri lon nhat trong list e la: ", max(e))
print("Gia tri nho nhat trong list e la: ", min(e))
print("--------------------------------------")

''' Them mot phan tu vao list '''
print("Gia su nhu ta them 10 vao e")
e.append(10)
print("List e sau khi them vao: ", e)
print("--------------------------------------")

''' Them tat ca phan tu list e vao list d '''
d.extend(e)
print("List d sau khi them list e vao: ", d)
print("--------------------------------------")

''' Trả về chỉ số bé nhất mà obj xuất hiện trong list '''
print("Vi tri xuat hien dau tien cua 5 la: ", d.index(5))
print("--------------------------------------")

''' Thêm phần tử obj vào vị trí index trong list. '''
e.insert(5, 3)
print("List e sau khi them 3 vao vi tri 5 trong list la: ", e)
print("--------------------------------------")

''' Xoá và trả về phần tử có chỉ số index trong list. '''
e.pop(5)
print("List e sau khi xoa: ", e)

''' Xóa phần tử trong list. '''
e.remove(3)
print("List e sau khi xoa 3:", e)
print("--------------------------------------")

''' Đảo ngược các phần tử trong list. '''
d.reverse()
print("List d sau khi dao nguoc: ", d)
print("--------------------------------------")

''' Sắp xếp các phần tử trong list. '''
f = [10, 5, 3, 7, 5, 8]
f.sort()
print("List f sau khi sap xep: ", f)
print("--------------------------------------")

''' ham tuple '''
print(tuple(f))
print("--------------------------------------")


''' Thao tác trên mảng với thư viện NumPy '''
a = np.array([0, 1, 2, 3, 4, 5])
print("Mang a: ", a)
print("So chieu cua a: ", a.ndim)
print("Hinh dang cua a: ", a.shape)
print("Cac phan tu xo gia tri lon hon 3: ", a[a > 3])
# Thay doi hinh dang mang a
b = a.reshape((3, 2))
print("Hinh sang mang a sau khi thay doi: \n", b)
print("Phan tu b[2][1] la: ", b[2][1])
# gan gia tri cho phan tu b[2][0] = 50
b[2][0] = 50
# nhan gia tri b voi 2
print("gia tri phan tu trong b sau khi nhan voi 2: \n", b*2)

print("--------------------------------------")
''' Đọc và xử lý dữ liệu từ file bên ngoài với thư viện pandas '''
dt = pd.read_csv("play_tennis.csv", delimiter=",")
print("Hien thi 5 dong dau:\n", dt.head())
print("Hien thi 7 dong cuoi: \n", dt.tail(7))
print("Hien thi tat ca: \n", dt)
print("Chi hien thi Outlook:\n", dt.Outlook)

print("--------------------------------------")
def unique(list1):
    list_set = set(list1)
    unique_list = (list(list_set))
    for x in unique_list:
        print(x, end=" ")
print("Nhan Play co cac gia tri:")
unique(dt.Play)
print("\n--------------------------------------")
print("Nhan Temp co cac gia tri:")
unique(dt.Temp)
print("\n--------------------------------------")
print("Nhan Outlook co cac gia tri:")
unique(dt.Outlook)
print("\n--------------------------------------")
print("Nhan Humidity co cac gia tri:")
unique(dt.Humidity)
print("\n--------------------------------------")
print("Nhan Windy co cac gia tri:")
unique(dt.Windy)