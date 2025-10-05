import os
import sys
import argparse
import time
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# ===============================
# ARGUMENTS
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Path to YOLO model file (.pt)')
parser.add_argument('--source', type=str, required=True, help='Camera/video source, e.g., 0 or usb0')
parser.add_argument('--resolution', type=str, default=None, help='WxH resolution, e.g., 1280x720')
parser.add_argument('--interval', type=float, default=5.0, help='Time interval in seconds between auto captures')
args = parser.parse_args()

MODEL_PATH = args.model
SOURCE = args.source
RESOLUTION = args.resolution
CAPTURE_INTERVAL = args.interval

CONF_THRESHOLD = 0.5
EXCEL_FILE = "bottle_data.xlsx"
SAVE_IMAGE_FOLDER = "captures"
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
# OPEN CAMERA OR VIDEO
# ===============================
cap = None
try:
    cam_index = int(SOURCE)
    cap = cv2.VideoCapture(cam_index)
except:
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

print("📷 Camera running. Press 'Q' to quit.\n")
last_capture_time = 0
global_bottle_id = 1  # Keep unique Bottle_ID across frames

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

    # Run YOLO detection
    results = model(frame, verbose=False)
    detections = results[0].boxes

    bottle_data = []

    # Draw green boxes and status text for detected bottles
    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        classidx = int(det.cls.item())
        classname = labels[classidx]  # "Full", "Half", or "Empty"
        conf = det.conf.item()
        if conf >= CONF_THRESHOLD:
            color = (0, 255, 0)  # green rectangle
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            # Draw bottle status above the rectangle
            text = classname
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = xmin
            text_y = max(ymin - 5, text_size[1] + 5)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

            # Save data for Excel with unique Bottle_ID
            bottle_data.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Bottle_ID": global_bottle_id,
                "Class_Name": classname,
                "Status": classname,
                "Confidence": round(conf, 3),
                "Xmin": xmin,
                "Ymin": ymin,
                "Xmax": xmax,
                "Ymax": ymax
            })
            global_bottle_id += 1  # Increment for next bottle

    # Display frame
    cv2.imshow("Bottle Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Auto-capture every interval seconds
    current_time = time.time()
    if current_time - last_capture_time >= CAPTURE_INTERVAL and bottle_data:
        last_capture_time = current_time
        img_name = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        img_path = os.path.join(SAVE_IMAGE_FOLDER, img_name)
        cv2.imwrite(img_path, frame)

        df_new = pd.DataFrame(bottle_data)
        df_new["Image_File"] = img_name
        if os.path.exists(EXCEL_FILE):
            df_existing = pd.read_excel(EXCEL_FILE)
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
            df_all.to_excel(EXCEL_FILE, index=False)
        else:
            df_new.to_excel(EXCEL_FILE, index=False)

        print(f"💾 {len(bottle_data)} bottles saved, image: {img_name}")

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
print("✅ Program closed successfully.")
