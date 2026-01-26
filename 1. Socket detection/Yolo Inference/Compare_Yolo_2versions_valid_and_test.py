from ultralytics import YOLO
import os

# Define the paths to your trained model weights
yolov8_path = 'PlugSocket_yolo11s_e200_k5.pt'
yolov11_path = 'PlugSocket_large_e234.pt'

# Define the path to your dataset configuration file
data_yaml_path = r'C:\Users\Kanwal\Python Lab Experiments\1. Experiment01 Electric Sockets\Compare Yolov models\PlugSocket_v11_2328\data.yaml'

# Load the models
model_v8 = YOLO(yolov8_path)
model_v11 = YOLO(yolov11_path)

# Create a list of models, names, and result directories for easy iteration
models_to_compare = [
    (model_v8, "YOLOv11_kFoldCrossvalidation", "yolov11_without_k_fold"),
    (model_v11, "YOLOv11_large", "yolov11_large_results")
]

# Function to run validation and testing
def run_evaluation(model, name, results_dir):
    print(f"\n--- Evaluating {name} ---")

    # Validation on 'val'
    print(f"--- Running Validation on 'val' dataset for {name} ---")
    metrics_val = model.val(data=data_yaml_path, name=f'{results_dir}_validation')
    print(f"  Validation mAP@0.5: {metrics_val.box.map50:.4f}")
    print(f"  Validation mAP@0.5-0.95: {metrics_val.box.map:.4f}")

    # Testing on 'test'
    print(f"--- Running Testing on 'test' dataset for {name} ---")
    metrics_test = model.val(data=data_yaml_path, split='test', name=f'{results_dir}_testing')
    print(f"  Test mAP@0.5: {metrics_test.box.map50:.4f}")
    print(f"  Test mAP@0.5-0.95: {metrics_test.box.map:.4f}")
    print(f"  Precision: {metrics_test.box.mp:.4f}")
    print(f"  Recall: {metrics_test.box.mr:.4f}")
    print(f"  Confusion matrix saved at: {metrics_test.save_dir}/confusion_matrix_normalized.png")

    # --- Per-class performance (focus on "socket") ---
    class_names = metrics_test.names  # dictionary of class indices to names
    if "socket" in class_names.values():
        socket_idx = list(class_names.values()).index("socket")
        print(f"\n  🔎 Socket class performance for {name}:")
        print(f"    Precision (P): {metrics_test.box.p[socket_idx]:.4f}")
        print(f"    Recall (R): {metrics_test.box.r[socket_idx]:.4f}")
        print(f"    mAP@0.5: {metrics_test.box.ap50[socket_idx]:.4f}")
        print(f"    mAP@0.5:0.95: {metrics_test.box.ap[socket_idx]:.4f}")
    else:
        print("\n  No 'socket' class found in dataset.yaml classes.")

# Run evaluations for all models
for model, name, results_dir in models_to_compare:
    run_evaluation(model, name, results_dir)

print("\n--- All evaluations complete. ---")
print("Check the specified directories to view the saved confusion matrices.")
