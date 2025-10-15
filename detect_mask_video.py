import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime

# ✅ Load trained model
model = load_model('model/best_mask_model.h5')

# ✅ Load Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ✅ Create folder to save "No Mask" violations
os.makedirs("violations", exist_ok=True)

# ✅ Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access webcam.")
    exit()

print("✅ Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Frame not captured.")
        break

    # Flip horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ✅ Adjust face detection sensitivity
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        # Extract face region
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        # ✅ Preprocess same as training
        face_resized = cv2.resize(face, (128, 128))
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_resized = face_resized / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        # ✅ Predict
        prob = model.predict(face_resized, verbose=0)[0][0]

        # ✅ Confidence thresholds
        if prob >= 0.6:
            label = "No Mask"
            color = (0, 0, 255)
            confidence = prob * 100
        else:
            label = "Mask"
            color = (0, 255, 0)
            confidence = (1 - prob) * 100

        label_text = f"{label}: {confidence:.2f}%"
        print(f"[INFO] Face detected → {label_text}")

        # ✅ Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 📸 Save if "No Mask"
        if label == "No Mask" and confidence > 80:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"violations/NoMask_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"🚨 Violation saved: {filename}")

    # ✅ Show live feed
    cv2.imshow("😷 Real-Time Mask Detection (Press 'q' to Quit)", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release resources
cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed. Program ended.")
