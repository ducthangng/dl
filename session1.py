import tensorflow as tf

# 1. Tải và chuẩn bị dữ liệu MNIST
# Dữ liệu được chia sẵn thành tập huấn luyện (train) và tập kiểm tra (test)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Chuẩn hóa dữ liệu: Giá trị pixel của ảnh ban đầu nằm trong khoảng 0-255.
# Ta chia cho 255.0 để đưa các giá trị này về khoảng 0-1, giúp AI học nhanh hơn.
x_train, x_test = x_train / 255.0, x_test / 255.0

print(f"Đã tải {len(x_train)} ảnh để huấn luyện và {len(x_test)} ảnh để kiểm tra.")

# 2. Xây dựng cấu trúc Mô hình (Neural Network)
model = tf.keras.models.Sequential([
  # Lớp 1: Ép phẳng bức ảnh 2D (28x28) thành một mảng 1D (784 pixel)
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  
  # Lớp 2: Lớp "ẩn" với 128 nơ-ron (chuyên gia tính toán) và hàm kích hoạt ReLU
  tf.keras.layers.Dense(128, activation='relu'),
  
  # Lớp 3: Dropout ngẫu nhiên bỏ qua 20% nơ-ron để tránh hiện tượng "học vẹt" (overfitting)
  tf.keras.layers.Dropout(0.2),
  
  # Lớp 4: Lớp đầu ra với 10 nơ-ron tương ứng với 10 chữ số (0 đến 9). 
  tf.keras.layers.Dense(10, activation='softmax')
])

# 3. Biên dịch mô hình (Compile)
# Cài đặt cách mô hình sẽ tự sửa sai và đo lường độ chính xác
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Huấn luyện mô hình (Training)
print("\nBắt đầu huấn luyện mô hình...")
# epochs=5 nghĩa là cho AI học đi học lại toàn bộ dữ liệu 5 lần
model.fit(x_train, y_train, epochs=5)

# 5. Đánh giá mô hình (Evaluation)
print("\nĐánh giá độ chính xác trên tập dữ liệu kiểm tra chưa từng thấy:")
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print(f"\nĐộ chính xác của mô hình là: {test_acc * 100:.2f}%")