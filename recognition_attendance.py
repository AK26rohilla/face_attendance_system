import cv2
import numpy as np
import csv
from datetime import datetime

# Load trained recognizer and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
label_dict = np.load("labels.npy", allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

attendance = {}  # Track punch-in/out

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        label_id, confidence = recognizer.predict(face_img)
        name = label_dict[label_id]

        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Punch-in / Punch-out logic
        time_now = datetime.now().strftime("%H:%M:%S")
        date_today = datetime.now().strftime("%Y-%m-%d")
        if name not in attendance:
            attendance[name] = {"in": time_now, "out": None}
            with open("attendance.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, date_today, "Punch-In", time_now])
            print(f"{name} punched in at {time_now}")
        else:
            if attendance[name]["out"] is None:
                attendance[name]["out"] = time_now
                with open("attendance.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, date_today, "Punch-Out", time_now])
                print(f"{name} punched out at {time_now}")

    cv2.imshow("Recognition + Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
