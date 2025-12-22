# streamlitGUI.py - ĐÃ SỬA LỖI ĐƯỜNG DẪN & CONFIG
import streamlit as st
import requests
from PIL import Image
import io
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from pathlib import Path
import os

# ====================== CẤU HÌNH HỆ THỐNG ======================
FASTAPI_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Định vị thư mục models (Logic giống main.py)
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Config này PHẢI GIỐNG main.py để đồng bộ
CNN_CONFIG = {
    "CNN_Deep3_Augmented_Balanced_79%": "CNN_Deep3_Augmentation_Balanced/CNN_Deep3_Augmentation_B.keras",
    "CNN_Deep3_Augmented_82%": "CNN_Deep3_Augmentation/CNN_Deep3_Augmentation.keras",
    "CNN_Deep3_77%": "CNN_Deep_3/CNN_Deep3.keras",
}

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😊", layout="centered")

st.markdown("""
    <h1 style='text-align: center;'>😊 Facial Emotion Recognition</h1>
    <h3 style='text-align: center; color: #666;'>YOLO Face + Custom CNN</h3>
""", unsafe_allow_html=True)

# Kiểm tra Backend
try:
    requests.get(f"{FASTAPI_URL}/", timeout=2)
    st.sidebar.success(f"✅ Backend Connected")
except:
    st.sidebar.error("❌ Backend Disconnected! Hãy chạy main.py trước.")

emotion_labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

with st.sidebar:
    st.header("⚙️ Cài đặt")
    # Dropdown lấy key từ config
    selected_cnn_name = st.selectbox("Chọn mô hình CNN", list(CNN_CONFIG.keys()), index=1)
    face_conf_threshold = st.slider("Ngưỡng YOLO", 0.1, 0.9, 0.4, 0.05)
    st.info(f"Model đang chọn: {selected_cnn_name}")


# =========================== LOGIC WEBCAM (TAB 3) ===========================
# Hàm load model Local cho Webcam
@st.cache_resource
def get_local_models():
    print(f"📷 Webcam đang tìm models tại: {MODELS_DIR}")

    # 1. Load YOLO
    yolo_p = MODELS_DIR / "yolov11n-face.pt"
    if not yolo_p.exists():
        st.error(f"❌ Không tìm thấy YOLO tại: {yolo_p}")
        return None, {}

    yolo = YOLO(str(yolo_p))

    # 2. Load CNNs
    cnns = {}
    for name, relative_path in CNN_CONFIG.items():
        full_path = MODELS_DIR / relative_path
        if full_path.exists():
            print(f"→ Loading local: {name}")
            try:
                cnns[name] = load_model(str(full_path), compile=False)
            except:
                pass
        else:
            print(f"→ ❌ File thiếu: {full_path}")

    return yolo, cnns


# =========================== GIAO DIỆN ===========================
tab1, tab2, tab3 = st.tabs(["📸 Ảnh tĩnh", "🎥 Video File", "🎦 Webcam Live"])

# --- TAB 1: ẢNH ---
with tab1:
    st.markdown("### 📸 Upload ảnh")
    uploaded_file = st.file_uploader("Chọn ảnh (JPG, PNG)", type=["jpg", "png"])

    if uploaded_file:
        image_pil = Image.open(uploaded_file)

        # --- BẮT ĐẦU ĐOẠN SỬA LỖI ---
        # Kiểm tra nếu ảnh có kênh Alpha (RGBA) hoặc hệ màu Palette (P)
        if image_pil.mode in ("RGBA", "P"):
            image_pil = image_pil.convert("RGB")
            # --- KẾT THÚC ĐOẠN SỬA LỖI ---

        st.image(image_pil, caption="Ảnh gốc", use_container_width=True)

        if st.button("🔍 Phân tích Cảm xúc"):
            # Resize để gửi nhanh hơn
            max_size = 800
            if image_pil.width > max_size:
                image_pil.thumbnail((max_size, max_size))

            img_byte_arr = io.BytesIO()
            # Bây giờ lệnh save này sẽ không còn lỗi vì ảnh đã là RGB
            image_pil.save(img_byte_arr, format='JPEG')

            with st.spinner("Đang gọi AI Server..."):
                try:
                    files = {"file": ("image.jpg", img_byte_arr.getvalue(), "image/jpeg")}
                    data = {"cnn_model_name": selected_cnn_name, "yolo_conf": face_conf_threshold}

                    response = requests.post(f"{FASTAPI_URL}/predict", files=files, data=data, timeout=60)

                    if response.status_code == 200:
                        res = response.json()
                        faces = res.get("results", [])

                        if not faces:
                            st.warning("⚠️ Không thấy mặt nào.")
                        else:
                            st.success(f"✅ Tìm thấy {len(faces)} khuôn mặt.")

                        # Hiển thị ảnh kết quả
                        img_b64 = np.array(res["image"], dtype=np.uint8)
                        img_result = cv2.imdecode(img_b64, cv2.IMREAD_COLOR)
                        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
                        st.image(img_result, caption="Kết quả AI", use_container_width=True)
                    else:
                        st.error(f"Lỗi API: {response.text}")
                except Exception as e:
                    st.error(f"Lỗi kết nối: {e}")

# --- TAB 2: VIDEO ---
with tab2:
    st.markdown("### 🎥 Xử lý Video")
    video_file = st.file_uploader("Upload video", type=["mp4", "avi"])

    if video_file and st.button("🚀 Xử lý Video"):
        st.warning("⏳ Đang xử lý, vui lòng chờ...")
        try:
            video_file.seek(0)
            files = {"file": (video_file.name, video_file.read(), video_file.type)}
            data = {"cnn_model_name": selected_cnn_name}

            response = requests.post(f"{FASTAPI_URL}/predict_video", files=files, data=data, timeout=300)

            if response.status_code == 200:
                st.success("✅ Xong!")
                st.download_button("⬇️ Tải Video về", response.content, file_name="result.mp4")
            else:
                st.error(f"Lỗi: {response.status_code}")
        except Exception as e:
            st.error(f"Lỗi: {e}")

# --- TAB 3: WEBCAM ---
with tab3:
    st.markdown("### 🎦 Webcam (Chạy Offline tại máy)")

    try:
        yolo_local, cnns_local = get_local_models()

        if not yolo_local:
            st.error("❌ Không thể chạy Webcam vì thiếu YOLO model.")
            st.stop()

        # Lấy model hiện tại, nếu không có lấy cái đầu tiên
        current_model = cnns_local.get(selected_cnn_name)
        if not current_model:
            if cnns_local:
                fallback = list(cnns_local.keys())[0]
                current_model = cnns_local[fallback]
                st.toast(f"⚠️ Model chọn bị lỗi, đang dùng: {fallback}", icon="⚠️")
            else:
                st.error("❌ Không load được model CNN nào cả.")
                st.stop()


        # Class xử lý từng khung hình
        class VideoProcessor:
            def __init__(self, model_instance, yolo_instance):
                self.model = model_instance
                self.yolo = yolo_instance

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")

                # Detect
                results = self.yolo(img, conf=0.5, verbose=False)[0]

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    face = img[y1:y2, x1:x2]

                    if face.size > 0:
                        try:
                            # Preprocess
                            face_resized = cv2.resize(face, (100, 100))
                            inp = face_resized.astype("float32") / 255.0
                            inp = np.expand_dims(inp, axis=0)

                            # Predict
                            pred = self.model.predict(inp, verbose=0)[0]
                            idx = np.argmax(pred)
                            label = f"{emotion_labels[idx]} {pred[idx]:.2f}"

                            # Draw
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        except:
                            pass

                return av.VideoFrame.from_ndarray(img, format="bgr24")


        webrtc_streamer(
            key="emotion-cam",
            video_processor_factory=lambda: VideoProcessor(current_model, yolo_local),
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
        )

    except Exception as e:
        st.error(f"Lỗi khởi tạo Webcam: {e}")