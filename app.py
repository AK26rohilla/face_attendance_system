import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="Face Attendance System", layout="centered")
st.title("📸 Face Attendance System")
st.write("Web-based AI attendance system using face recognition")

ATTENDANCE_FILE = "attendance.csv"

# Use Haar cascade for simple face detection (cloud compatible)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Demo known users
KNOWN_USERS = ["Akansha"]

# Upload image
uploaded_file = st.file_uploader("Upload Image to Mark Attendance", type=["jpg", "jpeg", "png"])

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    file_exists = os.path.exists(ATTENDANCE_FILE)

    with open(ATTENDANCE_FILE, "a") as f:
        if not file_exists:
            f.write("Name,Date,Time\n")
        f.write(f"{name},{date},{time}\n")

if uploaded_file:
    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("No face detected")
    else:
        # For cloud demo, assume first detected face is Akansha
        name = KNOWN_USERS[0]
        mark_attendance(name)
        st.success(f"✅ Attendance marked for {name}")

st.subheader("📄 Attendance Records")
if os.path.exists(ATTENDANCE_FILE):
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading attendance file: {e}")
else:
    st.info("No attendance records yet")



