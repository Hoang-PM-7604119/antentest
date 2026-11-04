# Antenna ML Inverse Design & Digital Twin Studio 📡

Trang web tĩnh chạy hoàn toàn trên trình duyệt (Client-side Machine Learning) phục vụ bài toán thiết kế ngược (Inverse Design) ăng-ten từ dữ liệu mô phỏng điện từ `data1.xlsx`.

🔗 **Live Website**: Triển khai trực tiếp qua **GitHub Pages**.

---

## 🌟 Giới Thiệu Bài Toán & Dữ Liệu

Từ bộ dữ liệu quét tham số mô phỏng gồm **614,656 cấu hình ăng-ten**:
- **4 thông số đầu ra (Cột 1, 2, 3, 4)**: Các kích thước hình học hình học ăng-ten $d_1, d_2, d_3, d_4 \in [5, 140]\text{ mm}$ với bước quét chế tạo là $5\text{ mm}$ ($28 \times 28 \times 28 \times 28 = 614,656$ mẫu).
- **2 thông số đầu vào (Cột 5, 6)**: Các thông số đặc tính điện từ mong muốn (ví dụ Hệ số phản xạ $S_{11}$ / Return Loss và Gain / Tham số bức xạ $S_{21}$ tính bằng dB).

Mục tiêu: **Nhập 2 thông số mục tiêu (Cột 5 & 6) $\to$ Dự đoán tự động 4 kích thước hình học tối ưu (Cột 1, 2, 3, 4).**

---

## 🤖 Các Thuật Toán Học Máy (Machine Learning Models)

Hệ thống tích hợp **6 mô hình học máy từ đơn giản đến nâng cao**:

1. **Hồi quy Tuyến tính (Linear Regression)**:
   - Mô hình giải tích đơn giản nhất: $\mathbf{d} = \mathbf{W}\mathbf{v} + \mathbf{b}$.
   - Tốc độ tính toán siêu nhanh ($< 0.01\text{ ms}$).
2. **Hồi quy Ridge (L2 Regularization)**:
   - Hồi quy tuyến tính có hiệu chỉnh $L_2$ để kiểm soát độ lệch và chống đa cộng tuyến.
3. **Hồi quy Đa thức bậc 2 (Polynomial Regression)**:
   - Bổ sung các số hạng phi tuyến bậc thấp ($v_1^2, v_2^2, v_1 v_2$) để mô tả độ uốn của bề mặt đáp ứng.
4. **Cây Quyết định (Decision Tree Regressor)**:
   - Phân đoạn không gian mục tiêu thành các khối hộp nhị phân tối ưu theo tiêu chuẩn giảm phương sai (MSE).
5. **Mạng Nơ-ron Nhân tạo (Multi-Layer Perceptron - MLP)**:
   - Mạng Deep Feedforward 3 tầng ẩn ($64 \to 64 \to 32$ nơ-ron) với hàm kích hoạt ReLU, có khả năng học các mối quan hệ phi tuyến phức tạp nhất.
6. **K Láng Giềng Gần Nhất (K-Nearest Neighbors - KNN / Database Matching)**:
   - Tìm kiếm trực tiếp các cấu hình mô phỏng thực tế gần nhất trong tập 614,656 mẫu, đảm bảo cấu hình đã qua kiểm chứng mô phỏng thực nghiệm.

---

## 🔬 Bộ Mô Phỏng Digital Twin Kiểm Tra Ngược (Forward Verification)

Ngoài mô hình nghịch đảo, hệ thống còn trang bị **Mạng Nơ-ron Digital Twin Simulator (Độ chính xác $R^2 = 91.2\%$)**:
- Nhận 4 kích thước dự đoán $d_1, d_2, d_3, d_4$.
- Mô phỏng xuôi để tính ra đáp ứng điện từ thực tế $\hat{v}_1, \hat{v}_2$.
- Tính sai số tuyệt đối $|\Delta v_1|, |\Delta v_2|$ để kỹ sư biết ngay mức độ tin cậy của thiết kế.

---

## 🎨 Trực Quan Hóa CAD Ăng-ten (Interactive SVG)

Sơ đồ hình học ăng-ten (Miếng bức xạ Patch $d_1 \times d_2$, Đường tiếp điện Feed Stub $d_3$, Khe phối hợp trở kháng Slot $d_4$) tự động co giãn theo thời gian thực khi người dùng thay đổi thông số hoặc đổi mô hình.

---

## 🚀 Hướng Dẫn Đồng Bộ Lên GitHub & Kích Hoạt GitHub Pages

### Bước 1: Khởi tạo và Commit Git tại máy (Đã hoàn tất)
```bash
cd /Users/MAC/antentest
git add .
git commit -m "feat: complete Antenna ML Inverse Design static website with 6 models"
```

### Bước 2: Tạo Repository trên GitHub
1. Truy cập [GitHub](https://github.com/new).
2. Đặt tên repository là: **`antentest`** (hoặc tên tùy chọn).
3. Chọn chế độ: **Public**.
4. Nhấn **Create repository**.

### Bước 3: Đẩy mã nguồn lên GitHub
Chạy các lệnh sau trong Terminal:
```bash
cd /Users/MAC/antentest
git remote add origin https://github.com/Hoang-PM-7604119/antentest.git
git branch -M main
git push -u origin main
```

### Bước 4: Kích hoạt GitHub Pages để Expose ra Internet
1. Vào repository trên GitHub: `https://github.com/Hoang-PM-7604119/antentest`.
2. Vào **Settings** $\to$ Chọn tab **Pages** ở thanh bên trái.
3. Tại mục **Build and deployment**:
   - **Source**: Chọn `Deploy from a branch`.
   - **Branch**: Chọn `main` và thư mục `/ (root)`.
   - Nhấn **Save**.
4. Chờ khoảng 1-2 phút, GitHub sẽ cấp link trang web công khai có dạng:
   **`https://hoang-pm-7604119.github.io/antentest/`**

---

## 📁 Cấu Trúc Mã Nguồn

```text
/Users/MAC/antentest/
├── index.html           # Giao diện chính Responsive, trực quan hóa CAD
├── styles.css           # Bảng màu, theme Sáng/Tối, CSS Grid & Flexbox
├── app.js               # Logic suy luận 6 thuật toán ML trên trình duyệt
├── models_data.js       # Trọng số pre-trained của 6 mô hình ML & dữ liệu mẫu
├── all_models.json      # File JSON cấu trúc mô hình
├── .gitignore           # Bỏ qua file Excel 30MB để repo nhẹ & chuẩn GitHub Pages
└── README.md            # Tài liệu dự án & hướng dẫn triển khai
```
