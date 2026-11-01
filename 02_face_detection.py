import os
import urllib.request
import cv2

url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
cascade_filename = "haarcascade_frontalface_default.xml"


if not os.path.exists(cascade_filename):
  print("Downloading Haar Cascade file...")
  urllib.request.urlretrieve(url, cascade_filename)
  print("Download completed successfully!")

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cascade_filename)

if not cap.isOpened():
  print("Error: Could not open video stream.")
  exit()

while True:
  ret, frame = cap.read()
  if not ret:
    print("Failed to grab frame.")
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
  )

  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

  cv2.imshow("Face Detection", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()