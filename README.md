# Real-Time Face Recognition System 

An end-to-end computer vision project built using Python and OpenCV to detect and recognize faces in real-time using the laptop's built-in webcam.

---

##  Features
* **Live Camera Integration:** Captures real-time video stream using OpenCV.
* **Face Detection:** Utilizes **Haar Cascade Classifiers** to accurately locate human faces in frames.
* **Automated Dataset Collection:** Captures and stores grayscale face samples for training.
* **Model Training:** Implements **LBPH (Local Binary Patterns Histograms)** face recognizer to learn unique facial patterns.
* **Real-Time Recognition:** Identifies the person and displays confidence scores dynamically on the live video stream.

---

##  Tech Stack
* **Language:** Python 3.13
* **Libraries:** OpenCV (`opencv-python`, `opencv-contrib-python`), NumPy, OS, Time

---

## 📂 Project Structure
```text
face_recognition_project/
│
├── 01_camera_stream.py        # Tests the camera feed
├── 02_face_detection.py       # Detects faces using Haar Cascades
├── 03_dataset_capture.py      # Captures and saves face dataset images
├── 04_train_model.py          # Trains the LBPH model and generates trainer.yml
├── 05_face_recognition.py     # Performs real-time face recognition
│
├── haarcascade_frontalface_default.xml # Face detection cascade file
└── README.md                  # Project documentation