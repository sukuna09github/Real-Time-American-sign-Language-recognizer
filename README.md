
```markdown
# 🖐️ ASL Hand Sign Recognition with MediaPipe + Scikit-learn

This project is a real-time hand gesture recognition system that detects and classifies American Sign Language (ASL) alphabets (A–Y, excluding J and Z) using a webcam, MediaPipe for landmark detection, and a machine learning model trained on collected hand landmark data.

---

## 🔍 Features

- Real-time hand tracking using MediaPipe
- Depth-aware 3D landmark visualization with BGR color mapping
- Live dataset creation (grayscale image + landmark CSV) with key-triggered labeling
- Automatic folder organization by label
- Enhanced grayscale hand ROI using CLAHE
- Model training with `RandomForestClassifier`
- ASL prediction from live webcam feed

---

## 📁 Dataset Structure

```
asl_dataset/
├── a/
│   ├── a1.png
│   └── ...
├── b/
│   └── ...
├── ...
├── all_landmarks.csv  # Combined landmark data with labels
```

Each `.csv` row contains 63 values (21 landmarks × [x, y, z]) + 1 label.

---

## 🧠 Model Training

```bash
# Train and save the model
python train_model.py
```

- Uses `RandomForestClassifier` from scikit-learn
- Model is saved as `asl_model.pkl`

---

## 🎯 Live Prediction

```bash
python predict_live.py
```

- Loads trained model
- Captures webcam input and predicts ASL letter in real time

---

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn joblib
```

### 2. Collect Data

```bash
python collect_data.py
```

- Press `a–y` to set the current label
- Press `SPACE` to capture and save the hand ROI + landmark
- Press `ESC` to exit

---

## 📈 Accuracy

Validation accuracy on test set (80/20 split):

```
Random Forest Accuracy: ~XX.XX%
```

_(Depends on dataset quality and size)_

---

## 🚧 To Do

- Add support for J and Z (motion tracking)
- Build deep learning version (CNN on grayscale ROI)
- Add GUI for easier labeling
- Upload dataset to HuggingFace or Kaggle

---

## 📸 Example Screenshots

<img src="demo_images/sample_tracking.png" width="400"/>
<img src="demo_images/sample_prediction.png" width="400"/>

---

## 🙌 Acknowledgements

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [Scikit-learn](https://scikit-learn.org/)
- American Sign Language (ASL) community ❤️

---

## 📃 License

This project is licensed under the MIT License.
```

---
