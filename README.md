# 😷 Real-Time Face Mask Detection System (Webcam-Based)

A **real-time face mask detection system** built with **Python**, **OpenCV**, and **TensorFlow/Keras**.  
It detects faces via webcam and classifies each face as **“Mask”** or **“No Mask”**, saving images of violations automatically.

---

## 🚀 Features
- Real-time mask detection using webcam  
- Haar Cascade face detection  
- Auto-save images for “No Mask” detections  
- Adjustable confidence thresholds  
- Lightweight and runs efficiently on CPU  

---

## 🧠 Tech Stack
- **Python 3.x**
- **OpenCV**
- **TensorFlow / Keras**
- **NumPy**
- **Datetime (Python standard library)**

---

## 📂 Project Structure

```
FaceMaskDetection/
│
├── model/
│   └── best_mask_model.h5
│
├── haarcascade_frontalface_default.xml
├── detect_mask_video.py
├── requirements.txt
└── README.md
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
python detect_mask_video.py
```

---

## 🧾 Requirements File (`requirements.txt`)

```
opencv-python
tensorflow
numpy
```

---

## 🧩 How It Works
1. The program starts the **webcam**.
2. Faces are detected using **Haar Cascade Classifier**.
3. The detected face is resized and passed to the trained **Keras model**.
4. The model predicts whether the person is **wearing a mask** or **not**.
5. “No Mask” detections are automatically **saved** in the `/violations/` folder with timestamps.

---

## 📸 Example Output
- Green box → Mask detected 😷  
- Red box → No Mask detected 🚨 (auto-saved)

---

## 🧠 Model Details
- Model used: `best_mask_model.h5`
- Trained using: TensorFlow/Keras
- Output: Binary classification (Mask / No Mask)
- Optimized for real-time detection on CPU

---

## 🧑‍💻 Author
**Maryam Fazal Gill**  
📧 *Developed as part of a deep learning and computer vision project.*

---

## 🏁 License
This project is released under the **MIT License** — feel free to modify and share.


