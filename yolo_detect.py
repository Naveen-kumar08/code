import os
import sys
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# ===============================
# USER CONFIGURATION
# ===============================
MODEL_PATHS = [
    r"C:\Users\Admin\Documents\YOLO\my_model\train\weights\best.pt",
    r"C:\Users\Admin\Documents\YOLO\my_model\train\weights\best.pt"  # Replace if second model is different
]
CONF_THRESHOLD = 0.5                   # minimum confidence
EXCEL_FILE = "bottle_data.xlsx"        # Excel sheet to save results
SAVE_IMAGE_FOLDER = "captures"         # folder to save images
CAMERA_INDEX = 0                        # 0 = default laptop camera

# Create folder for captured images
os.makedirs(SAVE_IMAGE_FOLDER, exist_ok=True)

# ===============================
# LOAD MODELS
# ===============================
models = []
for path in MODEL_PATHS:
    if not os.path.exists(path):
        print(f"❌ Model path not found: {path}")
        sys.exit()
    print(f"🔄 Loading YOLO model: {path}")
    models.append(YOLO(path))
print("✅ All models loaded successfully!")

# ===============================
# OPEN CAMERA
# ===============================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Camera not detected.")
    sys.exit()
print("✅ Camera connected.")

# ===============================
# MAIN LOOP
# ===============================
print("Press 'p' to capture bottle data and save image.")
print("Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame from camera.")
        break

    all_detections = []

    # Run both models on the same frame
    for model in models:
        results = model(frame, verbose=False)
        all_detections.extend(results[0].boxes)

    # Draw green bounding boxes and prepare Excel data
    bottle_data = []
    for i, det in enumerate(all_detections):
        xyxy = det.xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        classidx = int(det.cls.item())
        classname = model.names[classidx]  # Full, Half, Empty
        conf = det.conf.item()

        if conf >= CONF_THRESHOLD:
            # Draw green bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{classname} ({conf*100:.1f}%)", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save data for Excel
            bottle_data.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Bottle_ID": i + 1,
                "Class_Name": classname,
                "Status": classname,
                "Confidence": round(conf, 3),
                "Xmin": xmin,
                "Ymin": ymin,
                "Xmax": xmax,
                "Ymax": ymax
            })

    # Display the camera feed
    cv2.imshow("Bottle Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit program
    if key == ord('q'):
        break

    # Capture data and save
    elif key == ord('p'):
        if len(bottle_data) > 0:
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

            print(f"✅ {len(bottle_data)} bottles saved to {EXCEL_FILE}")
            print(f"✅ Image saved as {img_name}\n")
        else:
            print("⚠️ No bottles detected in this frame.\n")

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
print("Program closed successfully.")
