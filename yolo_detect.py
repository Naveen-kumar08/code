import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import argparse
from scipy.spatial import distance

# ===============================
# ARGUMENTS
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    help='Path to YOLO model file (.pt)')
parser.add_argument('--source', type=str, required=True,
                    help='Camera/video source, e.g., 0, usb0, or video.mp4')
parser.add_argument('--resolution', type=str, default=None,
                    help='WxH resolution, e.g., 1280x720')
parser.add_argument('--interval', type=float, default=5.0,
                    help='Time interval in seconds between auto captures')
parser.add_argument('--distance-thresh', type=int, default=50,
                    help='Max distance (pixels) to consider same bottle for tracking')
args = parser.parse_args()

MODEL_PATH = args.model
SOURCE = args.source
RESOLUTION = args.resolution
CAPTURE_INTERVAL = args.interval
DIST_THRESH = args.distance_thresh

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

print("📷 Auto-capture with tracking started. Press 'Q' to quit.\n")

# ===============================
# TRACKING VARIABLES
# ===============================
last_capture_time = 0
bottle_id_counter = 0
prev_centroids = []  # list of tuples (x_center, y_center, bottle_id)

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

    # Current frame centroids
    curr_centroids = []

    # Draw boxes
    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        classidx = int(det.cls.item())
        classname = labels[classidx]
        conf = det.conf.item()
        if conf >= CONF_THRESHOLD:
            # centroid
            x_center = (xmin + xmax) // 2
            y_center = (ymin + ymax) // 2
            curr_centroids.append((x_center, y_center, classname, xmin, ymin, xmax, ymax))

    # Match current centroids with previous centroids to assign unique IDs
    frame_bottles = []
    for c in curr_centroids:
        x, y, classname, xmin, ymin, xmax, ymax = c
        assigned_id = None
        min_dist = float('inf')
        for px, py, pid in prev_centroids:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < DIST_THRESH and dist < min_dist:
                assigned_id = pid
                min_dist = dist
        if assigned_id is None:
            bottle_id_counter += 1
            assigned_id = bottle_id_counter
        frame_bottles.append({
            "Bottle_ID": assigned_id,
            "Class_Name": classname,
            "Xmin": xmin, "Ymin": ymin, "Xmax": xmax, "Ymax": ymax
        })

    # Update prev_centroids
    prev_centroids = [(b["Xmin"] + (b["Xmax"]-b["Xmin"])//2,
                       b["Ymin"] + (b["Ymax"]-b["Ymin"])//2,
                       b["Bottle_ID"]) for b in frame_bottles]

    # Draw bounding boxes with IDs
    for b in frame_bottles:
        color = (0, 255, 0)
        cv2.rectangle(frame, (b["Xmin"], b["Ymin"]), (b["Xmax"], b["Ymax"]), color, 2)
        label = f'ID:{b["Bottle_ID"]} {b["Class_Name"]}'
        cv2.putText(frame, label, (b["Xmin"], b["Ymin"]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display
    cv2.imshow("Bottle Detection with Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Auto-capture every interval seconds
    current_time = time.time()
    if current_time - last_capture_time >= CAPTURE_INTERVAL:
        last_capture_time = current_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_name = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        img_path = os.path.join(SAVE_IMAGE_FOLDER, img_name)

        bottle_data = []
        for b in frame_bottles:
            bottle_data.append({
                "Timestamp": timestamp,
                "Bottle_ID": b["Bottle_ID"],
                "Class_Name": b["Class_Name"],
                "Status": b["Class_Name"],
                "Xmin": b["Xmin"],
                "Ymin": b["Ymin"],
                "Xmax": b["Xmax"],
                "Ymax": b["Ymax"],
                "Image_File": img_name
            })

        if bottle_data:
            cv2.imwrite(img_path, frame)
            df_new = pd.DataFrame(bottle_data)
            if os.path.exists(EXCEL_FILE):
                df_existing = pd.read_excel(EXCEL_FILE)
                df_all = pd.concat([df_existing, df_new], ignore_index=True)
                df_all.to_excel(EXCEL_FILE, index=False)
            else:
                df_new.to_excel(EXCEL_FILE, index=False)
            print(f"💾 {len(bottle_data)} bottles saved at {timestamp}, image: {img_name}")

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
print("✅ Program closed successfully.")
