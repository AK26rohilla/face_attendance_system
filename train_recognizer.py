import cv2
import os
import numpy as np

dataset_path = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces = []
labels = []
label_id = 0
label_dict = {}

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    label_dict[label_id] = person_name
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(label_id)
    label_id += 1

recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")
np.save("labels.npy", label_dict)
print("Training complete. Model saved as trainer.yml")
