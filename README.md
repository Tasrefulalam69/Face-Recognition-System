# Face Recognition Using Python and OpenCV

## About the Project
This project implements a face recognition system using Python and OpenCV. It uses the Local Binary Patterns Histograms (LBPH) face recognizer to recognize faces from images captured by a camera.

## Features
- Face detection using OpenCV Haar Cascade
- Face recognition using the LBPH algorithm
- Webcam-based face recognition
- Training a recognition model using labeled face images
- Recognition results displayed on the screen

## Technologies Used
- Python
- OpenCV
- NumPy

## Installation
Install the required libraries:

```bash
pip install opencv-contrib-python numpy
```

The `opencv-contrib-python` package includes the `cv2.face` module required for LBPH face recognition.

## Project Structure

```text
Face-Recognition-System/
├── face_recognition.py
├── train_model.py
├── requirements.txt
├── README.md
└── dataset/
    └── user_1/
        └── (private face images stored locally)
```

## Dataset
The system requires labeled face images for training. The personal face images used during development are not included in this public repository for privacy reasons.

Create your own local dataset folder and organize images according to the structure expected by the training script.

## How to Run
1. Clone or download this repository.
2. Install the required dependencies.
3. Add your own face images to the local dataset folder.
4. Run the training script:

```bash
python train_model.py
```

5. Run the face recognition application:

```bash
python face_recognition.py
```

Replace the script names above if your actual filenames are different.

## Learning Outcomes
- Understanding face detection and recognition
- Working with OpenCV Haar Cascade classifiers
- Learning the LBPH face recognition algorithm
- Training a model using labeled images
- Building a computer vision application with Python

## Privacy Notice
Personal face images and other biometric data are not included in this repository. Use face images only with appropriate consent.

## Author
Mohammad Tasreful Alam

## Disclaimer
This project is intended for educational purposes.
