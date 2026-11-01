import os
import time
import urllib.request
import cv2

url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
cascade_filename = "haarcascade_frontalface_default.xml"

if not os.path.exists(cascade_filename):
  print("Downloading Haar Cascade file...")
  urllib.request.urlretrieve(url, cascade_filename)
  print("Download completed successfully!")

face_cascade = cv2.CascadeClassifier(cascade_filename)

dataset_path = "face_dataset"
os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)
frame_width, frame_height = 1000, 600
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

count = 0
capture_interval = 0.5
last_capture_time = 0
MIN_FACE_SIZE = 100 

print("Starting to capture face images. Look at the camera...")

while True:
  ret, frame = cap.read()
  if not ret:
    print("Failed to grab frame")
    break

  frame = cv2.resize(frame, (frame_width, frame_height))
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

  current_time = time.time()
  for (x, y, w, h) in faces:
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
      continue
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if current_time - last_capture_time >= capture_interval:
      count += 1
      face_img = gray[y : y + h, x : x + w]
      cv2.imwrite(os.path.join(dataset_path, f"user_{count}.jpg"), face_img)
      print(f"Captured image {count}/100")
      last_capture_time = current_time

  cv2.imshow("Capturing Faces", frame)

  if cv2.waitKey(1) & 0xFF == 27 or count >= 100:
    break

cap.release()
cv2.destroyAllWindows()
print("Dataset captured successfully!")