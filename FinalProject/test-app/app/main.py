# main.py - ĐÃ SỬA LỖI ĐƯỜNG DẪN TUYỆT ĐỐI
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import shutil
from fastapi.responses import FileResponse

try:
    tf.config.set_visible_devices([], 'GPU')  # Chạy CPU cho ổn định
except:
    pass

app = FastAPI(title="YOLO Face + CNN Emotion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= CẤU HÌNH ĐƯỜNG DẪN (QUAN TRỌNG) =================
# Tự động tìm đường dẫn gốc: main.py -> backend -> project_root
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

print(f"📂 Server đang tìm models tại: {MODELS_DIR}")

# Load YOLO face
yolo_path = MODELS_DIR / "yolov11n-face.pt"
print(f"⏳ Đang tải YOLO model từ: {yolo_path}")

if yolo_path.exists():
    face_detector = YOLO(str(yolo_path))
else:
    print(f"❌ LỖI NGHIÊM TRỌNG: Không tìm thấy YOLO tại {yolo_path}")
    face_detector = None

# Cấu hình danh sách Model CNN (Tên hiển thị : Đường dẫn file con)
# Lưu ý: folder và file phải chính xác
CNN_CONFIG = {
    "CNN_Deep3_Augmented_Balanced_79%": "CNN_Deep3_Augmentation_Balanced/CNN_Deep3_Augmentation_B.keras",
    "CNN_Deep3_Augmented_82%": "CNN_Deep3_Augmentation/CNN_Deep3_Augmentation.keras",
    "CNN_Deep3_77%": "CNN_Deep_3/CNN_Deep3.keras",
}

print("⏳ Đang tải các mô hình CNN...")
loaded_cnn_models = {}

for name, relative_path in CNN_CONFIG.items():
    full_path = MODELS_DIR / relative_path
    if full_path.exists():
        print(f"→ ✅ Loading {name}...")
        try:
            loaded_cnn_models[name] = load_model(str(full_path), compile=False)
        except Exception as e:
            print(f"→ ⚠️ Lỗi khi load file {full_path.name}: {e}")
    else:
        print(f"→ ❌ Không tìm thấy file: {full_path}")
        print(f"     (Vui lòng kiểm tra file '{full_path.name}' có trong folder '{full_path.parent}' không)")

emotion_labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
print("✅ Server khởi động hoàn tất!")


@app.get("/")
def root():
    return {"message": "Emotion Recognition API is running!"}


@app.post("/predict")
def predict(
        file: UploadFile = File(...),
        cnn_model_name: str = Form("CNN_Deep3_Augmented_82%"),
        yolo_conf: float = Form(0.4)
):
    if not face_detector:
        raise HTTPException(500, detail="Server chưa load được YOLO model")

    if cnn_model_name not in loaded_cnn_models:
        # Nếu model client gửi lên không có, thử dùng model đầu tiên có sẵn
        if loaded_cnn_models:
            fallback = list(loaded_cnn_models.keys())[0]
            print(f"⚠️ Model {cnn_model_name} không có, dùng tạm {fallback}")
            cnn_model_name = fallback
        else:
            raise HTTPException(400, detail=f"Không có model CNN nào được load thành công.")

    cnn_model = loaded_cnn_models[cnn_model_name]

    # Đọc file
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, detail="Không đọc được file ảnh")

    # 1. Detect Face
    results = face_detector(img, conf=yolo_conf, verbose=False)[0]
    detected_faces = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue

        try:
            # 2. Preprocess
            face_resized = cv2.resize(face, (100, 100))
            input_data = face_resized.astype("float32") / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            # 3. Predict
            pred = cnn_model.predict(input_data, verbose=0)[0]
            idx = np.argmax(pred)
            conf = float(pred[idx])
            emo = emotion_labels[idx]

            detected_faces.append({
                "emotion": emo,
                "confidence": round(conf, 4),
                "box": (x1, y1, x2, y2)
            })

            # 4. Draw
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{emo} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img, (x1, y1 - 30), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        except Exception as e:
            print(f"Lỗi xử lý khuôn mặt: {e}")
            continue

    _, buffer = cv2.imencode(".jpg", img)

    return {
        "results": detected_faces,
        "count": len(detected_faces),
        "model_used": cnn_model_name,
        "image": buffer.tolist()
    }


@app.post("/predict_video")
def predict_video(
        file: UploadFile = File(...),
        cnn_model_name: str = Form("CNN_Deep3_Augmented_82%")
):
    if not face_detector:
        raise HTTPException(500, detail="YOLO Model Error")

    temp_input = f"temp_in_{file.filename}"
    temp_output = f"temp_out_{file.filename}"

    with open(temp_input, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(temp_input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    # Chọn model
    if cnn_model_name in loaded_cnn_models:
        cnn_model = loaded_cnn_models[cnn_model_name]
    elif loaded_cnn_models:
        cnn_model = list(loaded_cnn_models.values())[0]
    else:
        raise HTTPException(500, detail="No CNN model loaded")

    print(f"🎬 Bắt đầu xử lý video: {file.filename}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = face_detector(frame, conf=0.4, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            face = frame[y1:y2, x1:x2]

            if face.size > 0:
                try:
                    face_resized = cv2.resize(face, (100, 100))
                    inp = face_resized.astype("float32") / 255.0
                    inp = np.expand_dims(inp, axis=0)

                    pred = cnn_model.predict(inp, verbose=0)[0]
                    idx = np.argmax(pred)
                    label = f"{emotion_labels[idx]} {pred[idx]:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                except Exception:
                    pass

        out.write(frame)

    cap.release()
    out.release()

    if os.path.exists(temp_input):
        os.remove(temp_input)

    return FileResponse(temp_output, media_type="video/mp4", filename=f"processed_{file.filename}")