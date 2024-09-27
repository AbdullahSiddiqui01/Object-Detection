🖼️ Object and Face Detection with Voice Feedback and Email Notification
📜 Project Overview
This project implements a real-time system that utilizes computer vision techniques to detect objects and recognize faces using a webcam. The system enhances user interaction by providing voice feedback for detected objects and sending email notifications when recognized faces are identified. The core components of the project include object detection, face recognition, text-to-speech (TTS) for voice output, and email notifications.

🔑 Key Components
📚 Libraries Used:
PyTorch: Utilizes the YOLOv5 model for object detection.
OpenCV: Captures video from the webcam and processes the frames.
face_recognition: Detects and recognizes faces in the video stream.
gTTS (Google Text-to-Speech): Converts text to speech to provide audio feedback.
smtplib: Used for sending emails when faces are recognized.
🔍 Model Loading:
The project loads the pre-trained YOLOv5 model using PyTorch, which is used to identify and label objects in real-time from the webcam feed.

👤 Known Faces Setup:
A known face is loaded from an image file (abdog.jpg), which is encoded and stored in a list. The name associated with this face is also saved, allowing the system to recognize it later.

⚙️ Core Functions
🎤 Speak Function:
This function converts text into speech using gTTS. The spoken message is saved as an MP3 file and played back to the user using the playsound library.

✉️ Email Sending Function:
This function sends an email notification when a recognized face is detected. It constructs an email message with a subject and body, logs in to the SMTP server, and sends the email to a specified receiver.

🔄 Main Loop
📹 Video Capture:
The webcam is accessed using OpenCV’s VideoCapture method, continuously capturing frames in a loop.

🥼 Object Detection:
Each frame is processed using the YOLOv5 model, detecting objects and drawing bounding boxes around them with labels.
Newly detected objects that have not been seen before trigger the speak function to provide audio feedback, and the detection is logged with a timestamp.
👁️ Face Recognition:
The RGB format of the captured frame is used to locate faces.
The face encodings are compared with known face encodings to identify recognized faces.
When a face is recognized for the first time, audio feedback is provided, and an email notification is sent. The detection is logged with a timestamp.
🖥️ Displaying Output:
The processed frame (with detected objects and recognized faces) is displayed in a window using OpenCV. The loop continues until the user presses the 'q' key to exit.

🎉 Conclusion
This project demonstrates the integration of multiple technologies to create an intelligent system that enhances user interaction through voice feedback and email alerts. It showcases skills in machine learning, computer vision, and application development, making it a valuable addition to any portfolio. The implementation provides a strong foundation for further exploration in areas such as security systems, home automation, and interactive user interfaces.



