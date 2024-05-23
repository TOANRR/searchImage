import cv2
import numpy as np

# Đọc ảnh
image = cv2.imread('image.jpg')

# Chuyển đổi ảnh sang không gian màu LAB
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Tạo mask cho quần áo dựa trên ngưỡng màu
lower_clothing = np.array([0, 128, 128], dtype=np.uint8)
upper_clothing = np.array([255, 255, 255], dtype=np.uint8)
clothing_mask = cv2.inRange(lab_image, lower_clothing, upper_clothing)

# Tạo mask cho background dựa trên ngưỡng màu khác
lower_background = np.array([0, 0, 0], dtype=np.uint8)
upper_background = np.array([50, 50, 50], dtype=np.uint8)
background_mask = cv2.inRange(lab_image, lower_background, upper_background)

# Kết hợp mask của quần áo và background
final_mask = cv2.bitwise_or(clothing_mask, background_mask)

# Áp dụng mask lên ảnh gốc để chỉ giữ lại phần quần áo và xóa nền
image_with_clothing_only = cv2.bitwise_and(image, image, mask=final_mask)

# Hiển thị ảnh chỉ giữ lại phần quần áo
cv2.imshow('Clothing Only', image_with_clothing_only)
cv2.waitKey(0)
cv2.destroyAllWindows()
