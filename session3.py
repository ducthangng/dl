import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 1. Tải tập dữ liệu CIFAR-10
# Tập dữ liệu này chứa 60.000 ảnh màu kích thước 32x32 pixel chia làm 10 phân lớp
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Chuẩn hóa giá trị pixel về khoảng 0-1
train_images, test_images = train_images / 255.0, test_images / 255.0

print(f"Đã tải {len(train_images)} ảnh màu để huấn luyện và {len(test_images)} ảnh để kiểm tra.")

# 2. Xây dựng mô hình Mạng nơ-ron chập (CNN)
model = models.Sequential([
    # Lớp chập 1: Dùng 32 "kính lúp" (filter) cỡ 3x3 để quét qua ảnh.
    # input_shape=(32, 32, 3): Ảnh cao 32, rộng 32, và có 3 kênh màu (Đỏ, Xanh lá, Xanh dương - RGB)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    
    # Lớp thu nhỏ 1: Giảm kích thước ảnh đi một nửa để giữ lại các đặc điểm nổi bật nhất
    layers.MaxPooling2D((2, 2)),
    
    # Lớp chập 2 & thu nhỏ 2: Tiếp tục trích xuất các đặc điểm phức tạp hơn
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Lớp chập 3: Trích xuất các chi tiết bậc cao (VD: bánh xe, mắt mèo...)
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # --- Chuyển từ phân tích hình ảnh sang ra quyết định ---
    
    # Ép phẳng dữ liệu sau khi đã được chắt lọc qua các lớp chập
    layers.Flatten(),
    
    # Lớp nơ-ron thông thường để kết nối các đặc điểm lại với nhau
    layers.Dense(64, activation='relu'),
    
    # Lớp đầu ra 10 nơ-ron cho 10 loại vật thể
    layers.Dense(10, activation='softmax')
])

# 3. Biên dịch mô hình
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Huấn luyện mô hình
print("\nBắt đầu huấn luyện mô hình CNN (có thể mất vài phút)...")
history = model.fit(train_images, train_labels, 
                    epochs=10, 
                    validation_data=(test_images, test_labels))

# 5. Đánh giá độ chính xác
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f"\nĐộ chính xác trên ảnh mới: {test_acc * 100:.2f}%")