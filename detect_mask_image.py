import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('model/mask_detector_model_binary.h5')

# Define labels
labels = ['with_mask', 'without_mask']

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load test image
image_path = input("Enter image path: ")  # e.g. 'test.jpg'
img = cv2.imread(image_path)

if img is None:
    print("❌ Error: Image not found. Check the file path.")
    exit()

# Convert to grayscale for face detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (128, 128))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    prediction = model.predict(face)[0][0]
    label = labels[0] if prediction < 0.5 else labels[1]
    color = (0, 255, 0) if label == 'with_mask' else (0, 0, 255)

    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

cv2.imshow('Mask Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
