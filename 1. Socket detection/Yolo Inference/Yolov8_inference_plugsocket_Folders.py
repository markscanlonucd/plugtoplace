import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

input_folder = "test2"
output_folder = "test_yolov11m_2328_e250"
cropped_folder = "test_yolov11m_2328_e250_cropped"

# Create base output folders
os.makedirs(output_folder, exist_ok=True)
os.makedirs(cropped_folder, exist_ok=True)

# Load the model
model = YOLO("yolov11m_2328_e250.pt")
print("Model class names:", model.names)

# Allowed image extensions
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Font for labels
try:
    font = ImageFont.truetype("arial.ttf", 15)
except:
    font = ImageFont.load_default()

# Assign colors for classes
COLORS = ["red", "blue", "green", "orange", "purple"]

for root, dirs, files in os.walk(input_folder):
    relative_path = os.path.relpath(root, input_folder)
    save_dir = os.path.join(output_folder, relative_path)
    os.makedirs(save_dir, exist_ok=True)

    for image_name in files:
        if image_name.lower().endswith(image_extensions):
            input_path = os.path.join(root, image_name)
            results = model.predict(source=input_path, conf=0.5, verbose=False)

            for result in results:
                boxes = result.boxes.xyxy
                confs = result.boxes.conf
                classes = result.boxes.cls

                if boxes is not None and len(boxes) > 0:
                    detected_classes = [model.names[int(c)] for c in classes]
                    print(f"Detected classes in {image_name}: {detected_classes}")

                    # Open image for drawing
                    image = Image.open(input_path).convert("RGB")
                    draw = ImageDraw.Draw(image)
                    img_w, img_h = image.size
                    total_image_area = img_w * img_h

                    # Draw bounding boxes for all detections
                    for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                        if conf < 0.65:
                            continue
                        cls_name = model.names[int(cls)]
                        color = COLORS[int(cls) % len(COLORS)]

                        x1, y1, x2, y2 = map(int, box.tolist())
                        object_area = (x2 - x1) * (y2 - y1)
                        area_percentage = (object_area / total_image_area) * 100

                        # Draw rectangle
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                        # Add label with confidence and area percentage
                        #label = f"{cls_name} {conf:.2f} {area_percentage:.1f}%"
                        label = f"{cls_name} {conf:.2f}"
                        text_bbox = draw.textbbox((x1, y1), label, font=font)
                        text_w = text_bbox[2] - text_bbox[0]
                        text_h = text_bbox[3] - text_bbox[1]

                        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=color)
                        draw.text((x1, y1 - text_h), label, fill="white", font=font)

                    # Save annotated image
                    annotated_path = os.path.join(save_dir, image_name)
                    image.save(annotated_path)
                    print(f"Saved annotated image: {annotated_path}")

                    # Reload original for cropping
                    original_image = Image.open(input_path).convert("RGB")

                    # Crop and save per class
                    for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                        if conf < 0.65:
                            continue
                        cls_name = model.names[int(cls)]

                        # Create class-specific folder
                        class_crop_dir = os.path.join(cropped_folder, cls_name, relative_path)
                        os.makedirs(class_crop_dir, exist_ok=True)

                        x1, y1, x2, y2 = map(int, box.tolist())
                        pad = 10
                        x1_p = max(0, x1 - pad)
                        y1_p = max(0, y1 - pad)
                        x2_p = min(img_w, x2 + pad)
                        y2_p = min(img_h, y2 + pad)

                        if x2_p > x1_p and y2_p > y1_p:
                            cropped_img = original_image.crop((x1_p, y1_p, x2_p, y2_p))
                            crop_filename = os.path.join(
                                class_crop_dir,
                                f"{os.path.splitext(image_name)[0]}_crop_{i+1}{os.path.splitext(image_name)[1]}"
                            )
                            cropped_img.save(crop_filename)
                            print(f"Saved cropped {cls_name} image: {crop_filename}")

print("Processing complete.")
