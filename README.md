

# 🧠 Handwritten Digit Prediction using RNN

This project demonstrates how to classify handwritten digits from the **MNIST dataset** using a **Recurrent Neural Network (RNN)** model built with TensorFlow/Keras. While CNNs are more commonly used for image classification, this project explores the application of RNNs for sequential processing of image data.

---

## 📌 Overview

* **Dataset**: MNIST (28x28 grayscale images of handwritten digits 0-9)
* **Model**: RNN (SimpleRNN / LSTM / GRU)
* **Goal**: Predict the digit shown in an image
* **Framework**: TensorFlow / Keras

---

## 🔍 Problem Statement

Predict the correct digit (0-9) from a 28x28 pixel grayscale image using a recurrent neural network architecture by treating image rows as sequences.

---

## 🧑‍💻 Tech Stack

* Python 
* TensorFlow / Keras
* NumPy, Matplotlib, Seaborn
* Google Colab 

---

## 🛠️ How it Works

1. **Data Preprocessing**

   * Load MNIST dataset
   * Normalize the pixel values (0-255 → 0-1)
   * Reshape images to sequences (28 timesteps of 28 pixels each)

2. **Model Architecture**

   * Input Layer → RNN Layer (SimpleRNN / LSTM) → Dense → Softmax Output

3. **Training**

   * Loss: Categorical Crossentropy
   * Optimizer: Adam
   * Accuracy Metric

4. **Evaluation**

   * Model is evaluated on the test set
   * Accuracy, loss, and confusion matrix plotted

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Handwritten-Digit-Prediction-using-RNN.git
   cd Handwritten-Digit-Prediction-using-RNN
   ```

2. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:

   ```bash
   jupyter notebook rnn_digit_classifier.ipynb
   ```

---

## 📌 Requirements

* Python
* TensorFlow
* NumPy
* Matplotlib
* Seaborn
