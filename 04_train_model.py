import os
import cv2
import numpy as np

dataset_path = "face_dataset"
trainer_path = "trainer.yml"

recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []

if not os.path.exists(dataset_path):
  print(
      f"Error: The dataset folder '{dataset_path}' does not exist! Run step 3"
      " first."
  )
  exit()

for file_name in os.listdir(dataset_path):
  img_path = os.path.join(dataset_path, file_name)
  img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  if img is not None:
    faces.append(img)
    labels.append(1) 

if len(faces) == 0:
  print("Error: No images found in the dataset folder!")
else:
  recognizer.train(faces, np.array(labels))
  recognizer.save(trainer_path)
  print(f"Training completed! Trainer saved successfully as '{trainer_path}'")