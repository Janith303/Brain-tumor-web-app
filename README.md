# 🧠 Multi-Class Brain Tumor Classification using Deep Transfer Learning

This repository implements an end-to-end deep transfer learning framework for multi-class brain tumor classification from MRI scans using **DenseNet121** and **ResNet50** architectures in TensorFlow and Keras.

---

## 📌 Overview

Automated analysis of brain MRI scans helps accelerate diagnosis and reduces inter-observer variability. This project utilizes a two-stage transfer learning strategy (Feature Extraction and Fine-Tuning) to classify MRI scans into four distinct categories:

* **Glioma**
* **Meningioma**
* **Pituitary**
* **No Tumor**

---

## 🏗️ Architecture & Pipeline Strategy

* **Backbone Architecture**: Pre-trained `DenseNet121` with ImageNet weights.
* **Native Preprocessing Integration**: Embedded `densenet_preprocess_input` layer directly inside the model graph to handle scaling automatically during training and inference.
* **Two-Stage Training Strategy**:
  1. **Stage 1 (Feature Extraction)**: Backbone frozen (`base_model.trainable = False`), training the custom classification head (`Adam`, $lr = 10^{-3}$).
  2. **Stage 2 (Fine-Tuning)**: Unfreezing upper dense blocks with a lower learning rate ($lr = 10^{-5}$) to fine-tune deep visual representations.
* **Regularization & Callbacks**: `GlobalAveragePooling2D`, `Dropout(0.3)`, `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau`.

---

## 📂 Repository Structure

```text
├── Backend/
│   ├── models/
│   ├── app.py
│   └── requirements.txt
├── Frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css             
└── README.md
```

## How to run

* **Run the installation command**
```text
pip install -r requirements.txt
```
* **Run the backend**
```text
.\venv\Scripts\activate
python app.py
```
