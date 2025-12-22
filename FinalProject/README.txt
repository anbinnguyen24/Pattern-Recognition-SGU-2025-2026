Đề tài Nhóm 11: Xây dựng hệ thống nhận diện cảm xúc khuôn mặt
============================================================================
THÀNH VIÊN THỰC HIỆN:

3122410004 - Nguyễn Văn An 

3122410294 - Lý Minh Phát 

3122410247 - Lê Quốc Nam 

============================================================================
DỮ LIỆU SỬ DỤNG:

Bộ dữ liệu: RAF-DB (Real-world Affective Faces Database).

Link dataset: https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset.

============================================================================
DANH SÁCH THỰC NGHIỆM:
Mỗi thư mục bao gồm 01 file Notebook huấn luyện và 01 file mô hình .keras tương ứng:

CNN-baseline: Thực nghiệm số 1 - Mô hình cơ sở.

CNN_Deep: Thực nghiệm số 2 - Kiến trúc sâu.

CNN_Deep_2: Thực nghiệm số 3 - Mô hình mở rộng tham số.

CNN_Deep_3: Thực nghiệm số 4 - Tích hợp L2 Regularization (Mô hình triển khai 1).

CNN_Deep_3_Augmentation: Thực nghiệm số 5 - Tăng cường dữ liệu (Mô hình triển khai 2 - Best Model).

CNN_Deep_3_Augmentation_Balanced: Thực nghiệm số 6 - Tối ưu trọng số lớp (Mô hình triển khai 3).

EfficientNetB0_Balanced : Thực nghiệm EfficientNetB0 vì kết quả quá thấp nên không đưa vào báo cáo

EfficientNetB0_baseline : Thực nghiệm EfficientNetB0 vì kết quả quá thấp nên không đưa vào báo cáo

Evaluation_Best_Model.ipynb: File đánh giá tổng hợp cả 6 thực nghiệm trên cùng một tập kiểm thử.
============================================================================
CẤU TRÚC SẢN PHẨM (test-app):
Ứng dụng được xây dựng theo kiến trúc Client-Server:

thư mục app: Triển khai Backend bằng FastAPI (file main.py).

models: Chứa các file trọng số mô hình CNN và mô hình YOLOv11n-face để định vị khuôn mặt.

templates: Giao diện người dùng bằng Streamlit (file streamlitGUI.py).

requirements.txt: Danh sách các thư viện cần thiết (TensorFlow, Ultralytics, FastAPI, Streamlit...).

============================================================================
HƯỚNG DẪN CHẠY ỨNG DỤNG:
1. Cài đặt môi trường: pip install -r requirements.txt

2. Chạy Backend (Cửa sổ Terminal 1): cd test-app/app uvicorn main:app --host 127.0.0.1 --port 8000

3. Chạy Frontend (Cửa sổ Terminal 2): cd test-app/templates streamlit run streamlitGUI.py
============================================================================
TÀI LIỆU ĐÍNH KÈM:

Báo cáo: Assignment_report.docx và Assignment_report.pdf.

Slide thuyết trình: Slide_Nhom11.pptx.

