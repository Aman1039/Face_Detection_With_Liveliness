# Face Recognition with Liveness Detection

## Overview

This project implements real-time **Face Recognition** with **Liveness Detection** using OpenCV, Dlib, and Face Recognition libraries. It enhances security by preventing spoofing attempts using action-based liveness detection and blink detection.

## Features

- **Real-time face recognition** using pre-trained models
- **Liveness detection** via blink detection and action-based verification
- **Action-based authentication** requiring user responses to prevent spoofing
- **Fake attempt detection** with warnings for spoofing

## Technologies Used

- Python
- OpenCV
- Dlib
- Face Recognition Library
- NumPy

## Installation

To run this project on your system, follow these steps:

1. Download the face_detect.py file.
2. Install required Dependencies.
3. Run on Vscode or other.

## Usage

The program will start the webcam and detect faces in real time.
Users will be prompted to perform actions (e.g., "Turn Left", "Tilt Right").
If the actions are completed correctly, liveness detection is confirmed.
Fake attempts will be flagged with an alert message.
Press q to exit the program.

