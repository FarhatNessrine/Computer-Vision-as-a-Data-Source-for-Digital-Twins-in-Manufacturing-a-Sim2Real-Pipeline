import os
import time
import shutil
import torch
from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

# CONFIGURATION

DEVICE = 0

DATASET_DIR = "paths/to/dataset"
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

REAL_TEST_YAML = "paths/to/test_dataset/data.yaml"

WEIGHTS_PATH = "paths/to/dataset/yolov8m-worldv2.pt"

PROJECT = "paths/to/project_name"

os.makedirs(PROJECT, exist_ok=True)

# TRAINING PARAMETERS

epochs = 150
batch = 32
lr0 = 0.0005
freeze_layers = 8
img_size = 640

run_name = f"120k_mix_e{epochs}_lr{lr0}"

# =============================
# LOAD MODEL
# =============================

model = YOLOWorld(WEIGHTS_PATH)

start_time = time.time()

# =============================
# DATA CONFIG
# =============================

data_config = dict(
    train=dict(
        yolo_data=[DATA_YAML],
    ),
    val=dict(
        yolo_data=[DATA_YAML],
    )
)

# =============================
# TRAINING
# =============================

results = model.train(
    data=data_config,
    epochs=epochs,
    batch=batch,
    imgsz=img_size,
    lr0=lr0,
    optimizer="Adam",
    freeze=freeze_layers,
    augment=True,
    dropout=0.2,
    workers=8,
    device=DEVICE,
    patience=50,
    project=PROJECT,
    name=run_name,
    trainer=WorldTrainerFromScratch,
    save=True,
    plots=True
)

train_time = time.time() - start_time

# =============================
# VALIDATION
# =============================

print("\nRunning validation on mixed dataset...")
val_metrics = model.val(data=DATA_YAML)

print(f"Validation mAP50: {val_metrics.box.map50:.4f}")

# =============================
# TEST ON SPONTANEOUS REAL DATA
# =============================

print("\nTesting on spontaneous real data...")
test_metrics = model.val(data=REAL_TEST_YAML)

print("\nReal-world Test Results")
print(f"Precision: {test_metrics.box.mp:.4f}")
print(f"Recall: {test_metrics.box.mr:.4f}")
print(f"mAP@50: {test_metrics.box.map50:.4f}")
print(f"mAP@50-95: {test_metrics.box.map:.4f}")

# =============================
# SAVE BEST MODEL
# =============================

best_model = os.path.join(PROJECT, run_name, "weights", "best.pt")
saved_model = os.path.join(PROJECT, "best_yoloworld_120k.pt")

if os.path.exists(best_model):
    shutil.copy(best_model, saved_model)
    print(f"\nBest model saved to: {saved_model}")

# =============================
# SUMMARY
# =============================

print("\nTraining completed")
print(f"Total training time: {train_time/3600:.2f} hours")
