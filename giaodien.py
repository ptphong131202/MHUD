import pygame

# Khởi tạo Pygame
pygame.init()

# Đặt kích thước cửa sổ trò chơi
width = 1000
height = 500
screen = pygame.display.set_mode((width, height))

# Thiết lập tiêu đề cửa sổ trò chơi
pygame.display.set_caption("Project nhóm 15")

# Tạo font
font1 = pygame.font.Font(None, 40)
font2 = pygame.font.Font(None, 20)
# Vòng lặp chính của trò chơi

input_box = pygame.Rect(100, 120, 200, 32)
text = ''
running = True
while running:
    # Xử lý các sự kiện
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                print(text)
                text = ''
            elif event.key == pygame.K_BACKSPACE:
                text = text[:-1]
            else:
                text += event.unicode

    # Cập nhật trạng thái của trò chơi

    # Vẽ đối tượng lên màn hình
    screen.fill((255, 255, 255))  # Màu nền trắng
    # Hiển thị chữ giữa màn hình
    name = font1.render("Soybean large data set", True, "red")  # Vẽ chữ
    screen.blit(name, (width/2 - name.get_width() / 2, 10))
    text_knn = font2.render(
        "1. K-Nearest Neighbors (KNN)", True, "black")  # Vẽ chữ
    screen.blit(text_knn, (16, 40))
    text_bayes = font2.render(
        "2. Naive Bayes", True, "black")  # Vẽ chữ
    screen.blit(text_bayes, (16, 60))
    text_tree = font2.render("3.Decision Tree", True, "black")  # Vẽ chữ
    screen.blit(text_tree, (16, 80))

    text_url = font2.render("Nhập url của tập dữ liệu:", True, "black")
    screen.blit(text_url, (15, 100))
    pygame.draw.rect(screen, (0, 0, 0), input_box, 2)
    text_surface = font2.render(text, True, (0, 0, 0))
    screen.blit(text_surface, (input_box.x + 5, input_box.y + 10))
    pygame.display.flip()

# Kết thúc Pygame
pygame.quit()
