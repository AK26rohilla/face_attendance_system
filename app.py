import streamlit as st
import cv2
import pickle
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINER_FILE = os.path.join(BASE_DIR, "face_trainer.yml")
LABELS_FILE = os.path.join(BASE_DIR, "labels.pickle")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Face Attendance System")
st.title("📸 Face Attendance System")
st.write("Web-based AI attendance system using face recognition")

# ---------------- CHECK FILES ----------------
if not os.path.exists(TRAINER_FILE):
    st.error("❌ face_trainer.yml not found")
    st.stop()

if not os.path.exists(LABELS_FILE):
    st.error("❌ labels.pickle not found")
    st.stop()

# ---------------- LOAD MODEL ----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_FILE)

with open(LABELS_FILE, "rb") as f:
    label_map = pickle.load(f)

# ---------------- CREATE CSV IF NEEDED ----------------
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Status", "Time"])

# ---------------- UPLOAD IMAGE ATTENDANCE ----------------
st.subheader("📁 Upload Image to Mark Attendance")
uploaded_file = st.file_uploader(
    "Choose an image (JPG, PNG, JPEG)",
    type=["jpg", "png", "jpeg"]
)

def mark_attendance(name):
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date, "Marked", time])

if uploaded_file:
    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        st.warning("⚠️ No face detected")
    else:
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            label_id, confidence = recognizer.predict(roi)
            name = label_map.get(label_id, "Unknown")
            mark_attendance(name)
            st.success(f"✅ Attendance marked for {name}")
            break  # Mark only first face

# ---------------- LIVE CAMERA ATTENDANCE ----------------
st.subheader("📸 Live Camera Attendance")
if st.button("Start Camera"):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    run_camera = True
    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Cannot access camera")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            label_id, confidence = recognizer.predict(roi)
            name = label_map.get(label_id, "Unknown")

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # Mark attendance
            mark_attendance(name)
            st.success(f"✅ Attendance marked for {name}")
            break  # Only first face

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB")

        if st.button("Stop Camera"):
            run_camera = False

    cap.release()
    cv2.destroyAllWindows()

# ---------------- DISPLAY ATTENDANCE TABLE ----------------
st.subheader("📄 Attendance Records")
try:
    df = pd.read_csv(
        ATTENDANCE_FILE,
        engine="python",
        on_bad_lines="skip"
    )
    st.dataframe(df)
except Exception as e:
    st.error(f"Error reading attendance file: {e}")


