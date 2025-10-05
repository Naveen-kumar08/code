import os
import sys
import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# ===============================
# USER CONFIGURATION
# ===============================
MODEL_PATH = "runs/detect/train/weights/best.pt"   # Your YOLO model path
CONF_THRESHOLD = 0.5                               # Minimum confidence
EXCEL_FILE = "bottle_data.xlsx"                    # Excel sheet name
SAVE_IMAGE_FOLDER = "captures"                     # Folder to save captured photos
CAMERA_INDEX = 0                                   # 0 = default webcam, 1 = USB camera

# Create folder for captures
os.makedirs(SAVE_IMAGE_FOLDER, exist_ok=True)

# ===============================
# LOAD YOLO MODEL
# ===============================
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model path not found: {MODEL_PATH}")
    sys.exit()

print("🔄 Loading YOLO model...")
model = YOLO(MODEL_PATH)
labels = model.names
print("✅ Model loaded successfully!")

# ===============================
# OPEN CAMERA
# ===============================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Camera not detected.")
    sys.exit()

print("📷 Camera connected successfully.")
print("👉 Press 'C' to capture and save bottle data.")
print("👉 Press 'Q' to quit.\n")

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame from camera.")
        break

    # Run YOLO detection
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Draw detections on frame
    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf >= CONF_THRESHOLD:
            color = (0, 255, 0)
            label_text = f"{classname} ({conf*100:.1f}%)"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, label_text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show live camera window
    cv2.imshow("Bottle Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit program
    if key == ord('q'):
        break

    # Capture and save data
    elif key == ord('c'):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_name = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        img_path = os.path.join(SAVE_IMAGE_FOLDER, img_name)

        # Run YOLO again to ensure detections are current
        results = model(frame, verbose=False)
        detections = results[0].boxes

        bottle_data = []

        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]  # e.g., Full, Half, Empty
            conf = detections[i].conf.item()

            if conf >= CONF_THRESHOLD:
                bottle_data.append({
                    "Timestamp": timestamp,
                    "Bottle_ID": i + 1,
                    "Class_Name": classname,
                    "Status": classname,  # Full, Half, Empty
                    "Confidence": round(conf, 3),
                    "Xmin": xmin,
                    "Ymin": ymin,
                    "Xmax": xmax,
                    "Ymax": ymax,
                    "Image_File": img_name
                })

        # Save image and Excel data
        if bottle_data:
            cv2.imwrite(img_path, frame)

            df_new = pd.DataFrame(bottle_data)

            if os.path.exists(EXCEL_FILE):
                df_existing = pd.read_excel(EXCEL_FILE)
                df_all = pd.concat([df_existing, df_new], ignore_index=True)
                df_all.to_excel(EXCEL_FILE, index=False)
            else:
                df_new.to_excel(EXCEL_FILE, index=False)

            print(f"💾 {len(bottle_data)} bottles saved to '{EXCEL_FILE}'")
            print(f"🖼️ Image saved as '{img_name}'\n")
        else:
            print("⚠️ No bottles detected in this frame.\n")

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
print("✅ Program closed successfully.")
