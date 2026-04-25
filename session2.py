import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Tải tập dữ liệu IMDB
# Để đơn giản hóa, chúng ta chỉ giữ lại 10.000 từ tiếng Anh phổ biến nhất
vocab_size = 10000 
imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

print(f"Đã tải {len(train_data)} bài review để huấn luyện và {len(test_data)} bài để kiểm tra.")

# Chuẩn bị dữ liệu: Máy tính không hiểu chữ, nó chỉ hiểu số. 
# Keras đã tự động chuyển các từ thành các con số (VD: "the" = 1, "movie" = 14).
# Nhưng các bài review có độ dài ngắn khác nhau. Ta cần cắt hoặc chèn thêm số 0 
# để tất cả các bài review đều có chung chiều dài là 256 từ.
max_length = 256
train_data = pad_sequences(train_data, maxlen=max_length, padding='post')
test_data = pad_sequences(test_data, maxlen=max_length, padding='post')

# 2. Xây dựng mô hình
model = tf.keras.Sequential([
    # Lớp 1: Nhúng từ (Embedding). Lớp này biến các con số khô khan thành 
    # các vector toán học mang ý nghĩa. (VD: "tuyệt vời" và "hay" sẽ có vector gần nhau).
    tf.keras.layers.Embedding(vocab_size, 16),
    
    # Lớp 2: Gộp thông tin. Gom ý nghĩa của tất cả 256 từ lại để lấy "ý chính" của cả đoạn văn.
    tf.keras.layers.GlobalAveragePooling1D(),
    
    # Lớp 3: Lớp ẩn với 16 nơ-ron để phân tích ngữ cảnh (ví dụ phát hiện câu mỉa mai).
    tf.keras.layers.Dense(16, activation='relu'),
    
    # Lớp 4: Đầu ra chỉ có 1 nơ-ron duy nhất sử dụng hàm 'sigmoid' 
    # Hàm sigmoid sẽ nén giá trị trả về trong khoảng từ 0 đến 1.
    # Gần 0 -> Phim dở (Negative) | Gần 1 -> Phim hay (Positive)
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 3. Biên dịch mô hình
# Thay vì phân loại 10 số (sparse_categorical_crossentropy) như bài trước, 
# ở đây ta chỉ có 2 trạng thái nên dùng 'binary_crossentropy'
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Huấn luyện mô hình
print("\nBắt đầu huấn luyện mô hình đọc review...")
# Chúng ta chia tập huấn luyện ra thêm một phần nhỏ (validation_split=0.2) 
# để AI tự chấm điểm mình sau mỗi vòng học.
history = model.fit(train_data, train_labels, 
                    epochs=10, 
                    batch_size=512, 
                    validation_split=0.2,
                    verbose=1)

# 5. Đánh giá mô hình thực tế
print("\nĐánh giá trên tập dữ liệu kiểm tra:")
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print(f"\nĐộ chính xác khi dự đoán review mới: {test_acc * 100:.2f}%")