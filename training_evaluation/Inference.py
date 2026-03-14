import os
import torch
import cv2
from ultralytics import YOLOWorld
from ultralytics.utils.metrics import ConfusionMatrix
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIGURATION ===
DEVICE = 0  # GPU index
WEIGHTS_PATH = "paths/to/yolo_world_Before-Finetune-with-Spont-Data.pt"
VIDEO_PATH = "paths/to/test.mp4"
OUTPUT_DIR = "paths/to/inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD MODEL
print("Loading YOLO-World model...")

model = YOLOWorld(WEIGHTS_PATH)
model.to(DEVICE)

text_prompts = [
    "multi-color tidal turbine",
    "blue body assembled",
    "blue body not assembled",
    "black hub assembled",
    "black hub assembled",
    "red hub assembled",
    "hub not assembled",
    "blue rear cap assembled",
    "blue rear cap not assembled",
    "assembled hub",
    "unassembled hub",
    "assembled body",
    "unassembled body",
    "rear cap",
    "blue part",
    "red part",
    "black part",
    "grey part"
]

# Register text prompts for zero-shot inference
model.set_classes(text_prompts)

# Run inference on video
model.predict(
    source=VIDEO_PATH,
    project=OUTPUT_DIR,
    name="video_results",
    save=True,
    show=False,
    conf=0.60,
    device=DEVICE,
)

print(f"🎬 Video results saved in: {os.path.join(OUTPUT_DIR, 'video_results_mergedspont')}")
print("\n✅ Complete: Evaluation metrics, confusion matrix, and video inference generated.")
