import os
import time
from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

# ======================================================
# CONFIGURATION
# ======================================================

DEVICE = 0

# Model pretrained on 120k dataset
PRETRAINED_MODEL = "path/to/yolo_world_Before-Finetune-with-Spont-Data.pt"

# Dataset YAML for spontaneous dataset (train + val)
DATA_YAML = "path/to/spontaneous_data/data.yaml"

# Separate YAML for test dataset
TEST_YAML = "path/to/test_data/data.yaml"

PROJECT_DIR = "path/to/runs_yoloworld_finetune_spontaneous2"

os.makedirs(PROJECT_DIR, exist_ok=True)

# ======================================================
# TRAINING PARAMETERS
# ======================================================

EPOCHS = 40
BATCH = 16
IMGSZ = 640

LR = 0.0001
OPTIMIZER = "Adam"

# ======================================================
# LOAD MODEL
# ======================================================

model = YOLOWorld(PRETRAINED_MODEL)

start_time = time.time()

# ======================================================
# DATA CONFIGURATION
# ======================================================

data_config = dict(
    train=dict(
        yolo_data=[DATA_YAML],
    ),
    val=dict(
        yolo_data=[DATA_YAML],
    )
)

# ======================================================
# TRAINING
# ======================================================

print("\nStarting fine-tuning on spontaneous data...")

results = model.train(

    data=data_config,

    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMGSZ,

    optimizer=OPTIMIZER,
    lr0=LR,

    freeze=10,          # keep backbone features
    patience=20,

    augment=True,

    mosaic=0.5,
    mixup=0.2,

    device=DEVICE,

    project=PROJECT_DIR,
    name="finetune_spontaneous",

    plots=True,

    trainer=WorldTrainerFromScratch
)

# ======================================================
# TRAINING TIME
# ======================================================

train_time = time.time() - start_time

print(f"\nTraining time: {train_time/3600:.2f} hours")

# ======================================================
# VALIDATION (on validation split)
# ======================================================

print("\nRunning validation...")

val_metrics = model.val(
    data=DATA_YAML,
    split="val",
    imgsz=IMGSZ,
    device=DEVICE
)

print("\nValidation Results")
print(f"Precision: {val_metrics.box.p.mean():.4f}")
print(f"Recall: {val_metrics.box.r.mean():.4f}")
print(f"mAP@0.5: {val_metrics.box.map50:.4f}")
print(f"mAP@0.5-0.95: {val_metrics.box.map:.4f}")

# ======================================================
# TEST (independent dataset)
# ======================================================

print("\nTesting on independent spontaneous dataset...")

test_metrics = model.val(
    data=TEST_YAML,
    split="test",
    imgsz=IMGSZ,
    device=DEVICE
)

print("\nTest Results")
print(f"Precision: {test_metrics.box.p.mean():.4f}")
print(f"Recall: {test_metrics.box.r.mean():.4f}")
print(f"mAP@0.5: {test_metrics.box.map50:.4f}")
print(f"mAP@0.5-0.95: {test_metrics.box.map:.4f}")

print("\nPipeline finished successfully.")
