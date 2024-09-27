import torch
import cv2
import face_recognition
from gtts import gTTS
from playsound import playsound
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

known_face_encodings = []
known_face_names = []


image = face_recognition.load_image_file("abdog.jpg")
face_encoding = face_recognition.face_encodings(image)[0]

known_face_encodings.append(face_encoding)
known_face_names.append("Your Name")


def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = "voice.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# Send Email function
def send_email(subject, body):
    sender_email = "your_email@example.com"
    receiver_email = "receiver_email@example.com"
    password = "your_password"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.example.com', 587)
    server.starttls()
    server.login(sender_email, password)
    text = msg.as_string()
    server.sendmail(sender_email, receiver_email, text)
    server.quit()


cap = cv2.VideoCapture(0)  

detected_objects = set()
recognized_faces = set()

log_file = open("detection_log.txt", "a")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model(frame)
    current_objects = set()
    labels = results.names

    for detection in results.xyxy[0]:
        class_id = int(detection[-1])
        label = labels[class_id]
        current_objects.add(label)

        x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    new_objects = current_objects - detected_objects
    detected_objects = current_objects

    for obj in new_objects:
        speak(f"Detected object: {obj}")
        log_file.write(f"{datetime.datetime.now()}: Detected object - {obj}\n")


    rgb_frame = frame[:, :, ::-1]  
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            if name not in recognized_faces:
                recognized_faces.add(name)
                speak(f"Recognized face: {name}")
                log_file.write(f"{datetime.datetime.now()}: Recognized face - {name}\n")
                send_email("Face Recognized", f"Recognized {name} at {datetime.datetime.now()}")

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Object and Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
