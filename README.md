# Computer-Vision-as-a-Data-Source-for-Digital-Twins-in-Manufacturing-a-Sim2Real-Pipeline
This repository accompanies the paper "Computer Vision as a Data Source for Digital Twins in Manufacturing: a Sim2Real Pipeline", which is a reproducible a Sim2Real pipeline for industrial object detection using YOLO-World, combining synthetic and controlled data for assembly monitoring in Digital Twin manufacturing environments..

Key components of the repository include:

Scripts for training, evaluation, and inference of object detection models (YOLO-World)

A CVAT setup for semi-automatic annotation guide.

Scripts for data preparation: data split, Conversion from unity perception annotation to YOLO format

Instructions for environment setup and dependencies

## 📁 Repository Structure
TTA-Sim2Real/
├── training and Evaluation scripts/                 # Training scripts and YOLO configuration     
├── Checkpoints/              # Models checkpoints
├── dataset_preparation/          
│   ├── convert_annotations.py    # Convert CVAT annotations to YOLO format
│   ├── dataset_split.py          # Train/validation split scripts
│   └── merge_datasets.py         # Combine synthetic and controlled datasets
├── cvat_tutorial.md          # Step-by-step CVAT setup and usage guide
├── requirements.txt          # Python requirements for YOLO training/inference
├── LICENSE
└── README.md

## 📦 Models Checkpoints
To ensure reproducibility and maintain anonymity during the review process, all pretrained and finetuned models used in the **TTA-S2R (Tidal Turbine Assembly – Sim2Real)** study are hosted on an **anonymous Hugging Face repository**.  


🚀 Training
YOLO training scripts are included in the training/ directory.

📊 Evaluation
Run evaluation using the model predictions and ground truth annotations, the YOLO evaluation scripts are included in evaluation/ directory.

🔍 Inference
TH eenference scripts are included in inference/ directory.

✍️ CVAT Annotation
See cvat_tutorial.md for:
