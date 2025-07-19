# 🧠 MNIST Digit Classification with Artificial Neural Network (ANN)

This project builds and trains an Artificial Neural Network (ANN) using **TensorFlow/Keras** to classify handwritten digits from the **MNIST dataset**. It demonstrates the end-to-end process of building, training, evaluating, and saving a deep learning model using a simple feedforward architecture.

---

## 📌 Project Overview

- **Problem**: Classify 28x28 pixel grayscale images of handwritten digits (0–9).
- **Goal**: Achieve high test accuracy using a fully connected neural network (ANN).
- **Dataset**: MNIST — 70,000 images (60k train, 10k test) from [`tensorflow.keras.datasets`](https://keras.io/api/datasets/mnist/).

---

## 🧰 Tools & Libraries

- Python 🐍
- TensorFlow / Keras
- NumPy
- Matplotlib / Seaborn
- scikit-learn (for evaluation metrics)

---

## 🚀 Project Workflow

### 1. **Load Dataset**
- Load MNIST from Keras datasets.
- Split into training, validation, and test sets.

### 2. **Preprocess Data**
- Normalize pixel values to `[0, 1]`.
- One-hot encode the target labels.

### 3. **Build ANN Model**
- Sequential model with:
  - Flatten input layer
  - Dense(128) + ReLU + Dropout
  - Dense(64) + ReLU + Dropout
  - Dense(10) + Softmax

### 4. **Compile & Train**
- Loss function: `categorical_crossentropy`
- Optimizer: `adam`
- Metrics: `accuracy`
- Trained for 10 epochs with validation split.

### 5. **Evaluate**
- Final test accuracy: **93.04%**
- Final test loss: **0.2633**

### 6. **Visualize & Analyze**
- Confusion matrix using `seaborn.heatmap`
- Classification report with precision, recall, F1-score

### 7. **Save Model**
- Model saved as `mnist_ann_model.h5` using Keras API.

---

## 📊 Results Summary

| Metric         | Value       |
|----------------|-------------|
| Test Accuracy  | **93.04%**  |
| Test Loss      | **0.2633**  |
| Final Epoch Val Accuracy | **94.28%** |

### ✅ Confusion Matrix

> *Add `confusion_matrix.png` here if saved*

### ✅ Classification Report (Sample)

          precision    recall  f1-score   support

       0       0.94      0.97      0.95       980
       1       0.98      0.98      0.98      1135
       2       0.91      0.92      0.91      1032
       3       0.91      0.91      0.91      1010
       4       0.94      0.93      0.93       982
       5       0.91      0.88      0.89       892
       6       0.95      0.95      0.95       958
       7       0.92      0.94      0.93      1028
       8       0.89      0.87      0.88       974
       9       0.91      0.90      0.90      1009

accuracy                           0.93     10000

macro avg 0.93 0.93 0.93 10000
weighted avg 0.93 0.93 0.93 10000


---

## 📁 File Structure

```bash
📦mnist-ann-classifier/
 ┣ 📜mnist_ann_model.h5
 ┣ 📜mnist_ann.ipynb
 ┣ 📜README.md
 ┗ 📊confusion_matrix.png (optional)
```
---

✍️ Author
Joe Maina
Data Science & Machine Learning Enthusiast

---

📘 References
MNIST Dataset: 

 - Yann LeCun’s MNIST

 - TensorFlow Documentation: Keras API

 - Scikit-learn Evaluation Tools

---
