import cv2
import os
import urllib.request

cascade_filename = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_filename):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_filename)

trainer_path = "trainer.yml"
if not os.path.exists(trainer_path):
    exit()

face_cascade = cv2.CascadeClassifier(cascade_filename)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

cap = cv2.VideoCapture(0)

frame_width = 1000
frame_height = 600
skip_frames = 5
frame_count = 0
MIN_FACE_SIZE = 150

last_name = "Nouran"
last_confidence = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_count % skip_frames == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue
            face_img = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_img)
            last_name = "Nouran" if confidence < 55 else "Unknown"
            last_confidence = int(confidence)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{last_name} {last_confidence}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()