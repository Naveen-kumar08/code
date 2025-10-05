import os
import sys
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import argparse

# ===============================
# ARGUMENTS
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=r"C:\Users\Admin\Documents\YOLO\my_model\train\weights\best.pt",
                    help='Path to YOLO model file (.pt)')
parser.add_argument('--source', type=str, required=True, help='Camera/video source, e.g., 0, usb0, or video.mp4')
parser.add_argument('--resolution', type=str, default=None, help='WxH resolution, e.g., 1280x720')
args = parser.parse_args()

MODEL_PATH = args.model
SOURCE = args.source
RESOLUTION = args.resolution

# ===============================
# CHECK MODEL PATH
# ===============================
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model path not found: {MODEL_PATH}")
    sys.exit()

# ===============================
# CONFIG
# ===============================
CONF_THRESHOLD = 0.5
EXCEL_FILE = "bottle_data.xlsx"
SAVE_IMAGE_FOLDER = "captures"
os.makedirs(SAVE_IMAGE_FOLDER, exist_ok=True)

# ===============================
# LOAD YOLO MODEL
# ===============================
print("🔄 Loading YOLO model...")
model = YOLO(MODEL_PATH)
labels = model.names
print("✅ Model loaded successfully!")

# ===============================
# OPEN CAMERA OR VIDEO
# ===============================
cap = None
try:
    # Try converting SOURCE to integer (camera index)
    cam_index = int(SOURCE)
    cap = cv2.VideoCapture(cam_index)
except:
    # If SOURCE is usb0 style or a video file
    if "usb" in SOURCE:
        cam_index = int(SOURCE[3:])
        cap = cv2.VideoCapture(cam_index)
    elif os.path.isfile(SOURCE):
        cap = cv2.VideoCapture(SOURCE)
    else:
        print(f"❌ Invalid source: {SOURCE}")
        sys.exit()

if not cap or not cap.isOpened():
    print("❌ Camera/video not detected.")
    sys.exit()

# Parse resolution
resize = False
if RESOLUTION:
    try:
        resW, resH = map(int, RESOLUTION.split('x'))
        resize = True
    except:
        print("❌ Invalid resolution format. Use WxH, e.g., 1280x720")
        sys.exit()

print("📷 Press 'C' to capture bottles and save Excel data.")
print("📷 Press 'Q' to quit.\n")

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame.")
        break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run YOLO
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Draw boxes
    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()
        if conf >= CONF_THRESHOLD:
            color = (0, 255, 0)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{classname} ({conf*100:.1f}%)"
            cv2.putText(frame, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display
    cv2.imshow("Bottle Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    # Capture and save
    elif key == ord('c'):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_name = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        img_path = os.path.join(SAVE_IMAGE_FOLDER, img_name)

        # Run YOLO again for accurate capture
        results = model(frame, verbose=False)
        detections = results[0].boxes
        bottle_data = []

        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]  # Full/Half/Empty
            conf = detections[i].conf.item()
            if conf >= CONF_THRESHOLD:
                bottle_data.append({
                    "Timestamp": timestamp,
                    "Bottle_ID": i + 1,
                    "Class_Name": classname,
                    "Status": classname,
                    "Confidence": round(conf, 3),
                    "Xmin": xmin,
                    "Ymin": ymin,
                    "Xmax": xmax,
                    "Ymax": ymax,
                    "Image_File": img_name
                })

        # Save image & Excel
        if bottle_data:
            cv2.imwrite(img_path, frame)
            df_new = pd.DataFrame(bottle_data)
            if os.path.exists(EXCEL_FILE):
                df_existing = pd.read_excel(EXCEL_FILE)
                df_all = pd.concat([df_existing, df_new], ignore_index=True)
                df_all.to_excel(EXCEL_FILE, index=False)
            else:
                df_new.to_excel(EXCEL_FILE, index=False)
            print(f"💾 {len(bottle_data)} bottles saved to {EXCEL_FILE}")
            print(f"🖼️ Image saved as {img_name}\n")
        else:
            print("⚠️ No bottles detected in this frame.\n")

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
print("✅ Program closed successfully.")
