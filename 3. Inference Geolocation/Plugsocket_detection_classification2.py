import os
import csv
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tqdm import tqdm  # progress bar

# ========== CONFIG ==========
input_folder = "C:\\Users\\Kanwal\\Desktop\\UCD_Kanwal's_database\\Hotels50\\Hotels-50K-master\\Hotels-50K-master\\organized_images_by_country3\\traffickcam\\"
output_folder = "Output_traffickam"
csv_output_path = "results_traffickam.csv"

detection_model_path = "PlugSocket_yolo11s_e200_k5.pt"
classification_model_path = "SocketXception_fold5_noise.h5"
class_names = [
    'A_Type', 'B_Type', 'C_Type', 'DM_Type', 'E_Type', 'F_Type', 'G_Type',
    'H_Type', 'I_Type', 'JN_Type', 'K_Type', 'L_Type', 'Noise'
]
conf_threshold = 0.60  # YOLO confidence threshold

mapping_csv = "C:\\Users\\Kanwal\\Desktop\\UCD_Kanwal's_database\\Hotels50\\Hotels-50K-master\\Hotels-50K-master\\final_merged_with_country2.csv"

# ========== LOAD MODELS ==========
detector = YOLO(detection_model_path)
classifier = load_model(classification_model_path, compile=False)

# ========== LOAD IMAGE → HOTEL/CHAIN MAPPING ==========
mapping_df = pd.read_csv(mapping_csv)
mapping_df["image_id"] = mapping_df["image_id"].astype(str).str.strip()
id_to_hotels = mapping_df.groupby("image_id")[["hotel_id", "chain_id"]].apply(lambda x: x.values.tolist()).to_dict()

# ========== UTILITIES ==========
def classify_image(img_cv2):
    """Classifies an image using the classifier model."""
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0

    predictions = classifier.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100
    return class_names[predicted_index], confidence

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Prepare CSV header
header = [
    "image_id", "hotel_id", "chain_id", "country_name",
    "socket_detected", "yolo_confidence", "socket_class", "class_confidence"
]

# Write header first
with open(csv_output_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)

# ========== MAIN PROCESSING ==========
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Collect all images first
all_images = []
for root, dirs, files in os.walk(input_folder):
    rel_path = os.path.relpath(root, input_folder)
    if rel_path == ".":
        continue
    for image_name in files:
        if image_name.lower().endswith(image_extensions):
            all_images.append((root, rel_path, image_name))

# Process images with progress bar
for root, rel_path, image_name in tqdm(all_images, desc="Processing images", unit="img"):
    country_name = os.path.basename(root)
    out_dir = os.path.join(output_folder, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    image_id = Path(image_name).stem.strip()
    input_path = os.path.join(root, image_name)
    img = cv2.imread(input_path)
    if img is None:
        continue

    socket_detected = "No"
    yolo_confidence = ""
    socket_class = ""
    class_confidence = ""

    # YOLO detection
    results = detector([input_path], conf=conf_threshold)
    result = results[0]

    if len(result.boxes) > 0:
        for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            cls_name = detector.names[int(cls_id)]  # e.g., "socket" or "na"

            if cls_name.lower() != "socket":
                continue  # skip "na" detections

            x1, y1, x2, y2 = map(int, box)
            socket_detected = "Yes"
            yolo_confidence = round(float(conf.item()) * 100, 1)

            # Classification
            cropped = img[y1:y2, x1:x2]
            try:
                class_name, confidence = classify_image(cropped)
                socket_class = class_name.split("_")[0]
                class_confidence = round(confidence, 1)
            except Exception:
                socket_class = "Error"
                class_confidence = ""

            # Draw bounding box & label (only for socket class)
            label = f"{socket_class} ({class_confidence:.1f}%)"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            break  # only first socket detection per image

    # Lookup hotel_id & chain_id
    if image_id in id_to_hotels:
        hotel_chain_pairs = id_to_hotels[image_id]
    else:
        hotel_chain_pairs = [["", ""]]

    # Save annotated image
    output_path = os.path.join(out_dir, f"{image_id}_output.jpg")
    cv2.imwrite(output_path, img)

    # Write rows incrementally to CSV
    with open(csv_output_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for hotel_id, chain_id in hotel_chain_pairs:
            writer.writerow([
                image_id, hotel_id, chain_id, country_name,
                socket_detected, yolo_confidence, socket_class, class_confidence
            ])
