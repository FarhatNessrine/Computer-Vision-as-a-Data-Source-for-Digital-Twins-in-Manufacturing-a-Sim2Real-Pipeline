# Computer-Vision-as-a-Data-Source-for-Digital-Twins-in-Manufacturing-a-Sim2Real-Pipeline
This repository accompanies the paper "Computer Vision as a Data Source for Digital Twins in Manufacturing: a Sim2Real Pipeline", which is a reproducible a Sim2Real pipeline for industrial object detection using YOLO-World, combining synthetic and controlled data for assembly monitoring in Digital Twin manufacturing environments..

Key components of the repository include:

Scripts for training, evaluation, and inference of object detection models (YOLO-World)

A CVAT setup for semi-automatic annotation guide.

Scripts for data preparation: data split, Conversion from unity perception annotation to YOLO format

Instructions for environment setup and dependencies

## 📁 Repository Structure
TTA-Sim2Real/

├── training_evaluation/        # YOLO-World training and evaluation scripts

├── dataset_preparation/        # Dataset preprocessing and merging scripts

│    ├── convert_annotations.py

│    ├── dataset_split.py

│    └── merge_datasets.py

├── checkpoints/                # Saved trained models

├── docs/

│    └── cvat_tutorial.md        # Annotation tutorial using CVAT

├── requirements.txt

├── LICENSE

└── README.md

## 📦 Models Checkpoints
To ensure reproducibility and maintain anonymity during the review process, all pretrained and finetuned models used in the **TTA-S2R (Tidal Turbine Assembly – Sim2Real)** study are hosted on an **anonymous Hugging Face repository**.  


🚀 Training
YOLO-World training, evaluation, and enference scripts are included in the training_evaluation/ directory.

✍️ CVAT Annotation
See cvat_tutorial.md for:
