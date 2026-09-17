# CNN-Based Human Action Recognition using MobileNetV2 and BiLSTM

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![License](https://img.shields.io/badge/License-MIT-blue)

A deep learning-based **Human Action Recognition** system that uses a Convolutional Neural Network (CNN) for spatial feature extraction and a Bidirectional LSTM (BiLSTM) for temporal sequence learning.
\

---

## Overview

This project implements a video-based **Human Action Recognition** system using **MobileNetV2**, a Convolutional Neural Network architecture, as the primary spatial feature extractor.

The system processes video sequences by extracting frames, preprocessing them, and using MobileNetV2 to learn spatial features from individual frames. These features are then passed to a **Bidirectional LSTM** to learn temporal relationships across the video sequence.

The model classifies videos into three categories:

* **Explosion**
* **Fighing**
* **Road_acc**

The project demonstrates an end-to-end deep learning workflow covering:

* Video preprocessing
* Frame extraction
* CNN-based feature extraction
* Temporal sequence learning
* Model training
* Model evaluation
* Video prediction

---

## Key Features

* CNN-based Human Action Recognition
* MobileNetV2 feature extraction
* Video frame extraction using OpenCV
* Image resizing and normalization
* Temporal sequence learning using BiLSTM
* Multi-class video classification
* Model evaluation using classification metrics
* Input video prediction
* Jupyter Notebook implementation

---

## Model Architecture

The overall architecture is:

```text
Input Video
     │
     ▼
Frame Extraction
     │
     ▼
Resize to 128 × 128
     │
     ▼
Normalization
     │
     ▼
MobileNetV2 CNN
     │
     ▼
Spatial Feature Extraction
     │
     ▼
Bidirectional LSTM
     │
     ▼
Fully Connected Layers
     │
     ▼
Softmax Classification
     │
     ▼
Explosion / Fighing / Road_acc
```

### MobileNetV2 CNN

**MobileNetV2** is used as the main CNN backbone for extracting spatial features from individual video frames.

Configuration:

* Architecture: **MobileNetV2**
* Pretrained weights: **ImageNet**
* Input size: **128 × 128 × 3**
* Last 40 layers configured as trainable
* Used for spatial feature extraction

### Bidirectional LSTM

The frame-level CNN features are passed to a **Bidirectional LSTM** to learn temporal dependencies and motion patterns across consecutive frames.

Configuration:

* Sequence length: **20 frames**
* LSTM units: **32**
* Bidirectional processing

### Classification Network

The BiLSTM output is passed through fully connected layers:

```text
BiLSTM
   ↓
Dense(256)
   ↓
Dense(128)
   ↓
Dense(64)
   ↓
Dense(32)
   ↓
Dense(3)
   ↓
Softmax
```

Dropout layers are applied between the dense layers.

---

## Dataset

The expected dataset structure is:

```text
data/
└── Augmented Data/
    ├── Explosion/
    ├── Fighing/
    └── Road_acc/
```

### Classes

| Class     | Description                    |
| --------- | ------------------------------ |
| Explosion | Explosion-related activity     |
| Fighing   | Fighting-related activity      |
| Road_acc  | Road accident-related activity |

### Frame Processing

Each video goes through the following preprocessing pipeline:

1. Open the video using OpenCV.
2. Determine the number of frames.
3. Sample frames from the video.
4. Extract **20 frames** per sequence.
5. Resize each frame to **128 × 128**.
6. Normalize pixel values by dividing by 255.
7. Create the input sequence for the CNN-BiLSTM model.

> **Note:** The dataset is not included in this repository. The dataset directories are retained using `.gitkeep` files to document the expected structure.

---

## Training Configuration

| Parameter          | Value                    |
| ------------------ | ------------------------ |
| CNN Backbone       | MobileNetV2              |
| Pretrained Weights | ImageNet                 |
| Image Size         | 128 × 128 × 3            |
| Sequence Length    | 20                       |
| Number of Classes  | 3                        |
| LSTM Units         | 32                       |
| Optimizer          | SGD                      |
| Loss Function      | Categorical Crossentropy |
| Epochs             | 50                       |
| Batch Size         | 24                       |
| Test Size          | 10%                      |
| Validation Split   | 20%                      |
| Random Seed        | 27                       |
| Dropout            | 0.25                     |

---

## Project Workflow

```text
Dataset
   │
   ▼
Video Input
   │
   ▼
Frame Extraction
   │
   ▼
Frame Resizing
   │
   ▼
Normalization
   │
   ▼
MobileNetV2 CNN
   │
   ▼
Spatial Feature Extraction
   │
   ▼
Bidirectional LSTM
   │
   ▼
Dense Layers
   │
   ▼
Softmax Classification
   │
   ▼
Action Prediction
```

---

## Evaluation

The project includes an evaluation script that calculates:

* Accuracy
* Precision
* Recall
* Classification Report

Run:

```bash
python -m src.evaluate
```

The evaluation uses the generated feature and label artifacts together with the trained model.

---

## Video Prediction

The repository includes a prediction pipeline for classifying an input video.

Usage:

```bash
python -m src.predict <input_video_path>
```

Example:

```bash
python -m src.predict test_videos/input.mp4
```

The processed output video is saved as:

```text
test_videos/Output-Test-Video.mp4
```

---

## Project Structure

```text
Deep-Learning-Human-Action-Recognition/
│
├── artifacts/
│   └── .gitkeep
│
├── data/
│   └── Augmented Data/
│       ├── Explosion/
│       │   └── .gitkeep
│       ├── Fighing/
│       │   └── .gitkeep
│       └── Road_acc/
│           └── .gitkeep
│
├── models/
│   ├── cnn_lstm_model.py
│   ├── cnn_model.py
│   ├── lstm_model.py
│   └── __init__.py
│
├── notebooks/
│   └── Human_Action_Recognition.ipynb
│
├── saved_models/
│   └── .gitkeep
│
├── src/
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   ├── utils.py
│   └── __init__.py
│
├── test_videos/
│   └── .gitkeep
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── run.py
```

---

## Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **MobileNetV2**
* **OpenCV**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Jupyter Notebook**

---

## Installation

Clone the repository:

```bash
git clone https://github.com/harshaldhande/Deep-Learning-Human-Action-Recognition.git
```

Navigate to the project directory:

```bash
cd Deep-Learning-Human-Action-Recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

After placing the dataset in the required directory structure, start the training pipeline using:

```bash
python run.py train
```

The training pipeline generates the required feature and label artifacts and saves the trained model.

Generated artifacts:

```text
artifacts/
```

Saved model:

```text
saved_models/
```

---

## Notebook

The complete experimental workflow is available in:

```text
notebooks/Human_Action_Recognition.ipynb
```

The notebook contains the project's experimentation, preprocessing, model development, training, and evaluation workflow.

---

## Research

This project focuses on deep learning-based **video action recognition**, using a CNN-based approach to extract spatial information from video frames and a Bidirectional LSTM to learn temporal information.

The combination of **MobileNetV2 and BiLSTM** allows the system to utilize both spatial and temporal information present in video sequences.

---

## Author

### Harshal Dhande

**Post Graduate Diploma in Computing – IT Infrastructure, Systems and Security (PGCP-ITISS)**
Centre for Development of Advanced Computing (**C-DAC**)

**B.Tech – Electronics & Telecommunication Engineering**
Vishwakarma Institute of Technology, Pune

### Areas of Interest

* DevOps
* DevSecOps
* Kubernetes
* Cloud Computing
* CI/CD
* Linux
* Infrastructure Automation
* Container Security
* Machine Learning
* Deep Learning

---

## License

This project is licensed under the **MIT License**.
