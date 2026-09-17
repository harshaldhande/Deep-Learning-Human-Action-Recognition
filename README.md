# CNN-Based Human Action Recognition using MobileNetV2 and BiLSTM

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![License](https://img.shields.io/badge/License-MIT-blue)

A deep learning-based **Human Action Recognition** system that uses a Convolutional Neural Network (CNN) for spatial feature extraction and a Bidirectional LSTM (BiLSTM) for temporal sequence learning.

## Overview

This project implements a video-based action recognition system using **MobileNetV2**, a CNN architecture, as the primary feature extractor.

The system processes video sequences by extracting and preprocessing frames, using MobileNetV2 to learn spatial features from individual frames, and then using a Bidirectional LSTM to learn temporal relationships between frames.

The model classifies videos into three categories:

* **Explosion**
* **Fighing**
* **Road_acc**

The project includes video preprocessing, CNN-based feature extraction, temporal sequence learning, model training, evaluation, and video prediction.

---

## Features

* CNN-based video classification
* MobileNetV2 feature extraction
* Video frame extraction using OpenCV
* Image resizing and normalization
* Temporal sequence learning using BiLSTM
* Multi-class action classification
* Model evaluation
* Input video prediction
* Jupyter Notebook implementation

---

## CNN-Based Architecture

The main deep learning pipeline is:

```text
Input Video
     │
     ▼
Frame Extraction
     │
     ▼
Resize Frames
128 × 128 × 3
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
Softmax
     │
     ▼
3 Action Classes
```

### MobileNetV2 CNN

**MobileNetV2** is used as the CNN backbone for extracting spatial features from video frames.

Configuration:

* Architecture: MobileNetV2
* Pretrained weights: ImageNet
* Input size: **128 × 128 × 3**
* CNN layers: Last 40 layers configured as trainable
* Output: Spatial feature representation

The CNN processes individual frames and extracts meaningful visual features that are subsequently used by the temporal model.

### Bidirectional LSTM

The CNN features extracted from consecutive frames are passed to a **Bidirectional LSTM**.

The BiLSTM learns temporal dependencies and motion patterns across the sequence of frames.

Configuration:

* Sequence length: **20 frames**
* LSTM units: **32**
* Bidirectional processing

### Fully Connected Classification Network

The BiLSTM output is passed through fully connected layers:

```text
BiLSTM
   ↓
Dense 256
   ↓
Dense 128
   ↓
Dense 64
   ↓
Dense 32
   ↓
Dense 3
   ↓
Softmax
```

Dropout layers are used between the dense layers.

---

## Dataset

The dataset is organized into three action classes:

```text
data/
└── Augmented Data/
    ├── Explosion/
    ├── Fighing/
    └── Road_acc/
```

### Frame Processing

Each input video goes through the following preprocessing pipeline:

1. Open the video using OpenCV.
2. Determine the number of frames.
3. Sample frames from the video.
4. Extract **20 frames** per sequence.
5. Resize each frame to **128 × 128**.
6. Normalize pixel values by dividing by 255.
7. Create the input sequence for the CNN-BiLSTM model.

> The dataset itself is not included in this repository. The dataset directories are retained using `.gitkeep` files to document the expected structure.

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

## Evaluation

The trained model can be evaluated using:

* Accuracy
* Precision
* Recall
* Classification Report

Run:

```bash
python -m src.evaluate
```

The evaluation script uses the generated feature and label artifacts together with the trained model.

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

The processed video is saved as:

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
* **Scikit-learn**
* **Matplotlib**
* **Pandas**
* **Jupyter Notebook**

---

## Training

After placing the dataset according to the required directory structure, start training using:

```bash
python run.py train
```

The training pipeline:

```text
Dataset
   ↓
Frame Extraction
   ↓
Preprocessing
   ↓
CNN Feature Extraction
   ↓
BiLSTM
   ↓
Classification
   ↓
Model Saving
```

Generated artifacts are stored under:

```text
artifacts/
```

The trained model is stored under:

```text
saved_models/
```

---

## Notebook

The complete experimental implementation is available in:

```text
notebooks/Human_Action_Recognition.ipynb
```

The notebook contains the project's experimentation and deep learning workflow.

---

## Research

This project focuses on applying deep learning techniques to **video-based human action recognition**, with a CNN-based approach for extracting spatial information from video frames and recurrent neural networks for learning temporal information.

---

## Author

**Harshal Dhande**

B.Tech – Electronics & Telecommunication Engineering
Vishwakarma Institute of Technology, Pune

---

## License

This project is licensed under the **MIT License**.
