# 🥦 Food Freshness Detection using Deep Learning and Image Analysis

## 📘 Overview
This project focuses on automatically identifying the **freshness level of food items** using **image-based analysis**.  
By leveraging a **Convolutional Neural Network (CNN)** model trained on the *Food Freshness Dataset* from Kaggle,  
the system classifies food samples (such as fruits, vegetables, and other perishable goods) into freshness categories.  

The goal is to develop a **fast, accurate, and efficient AI model** that can assist in food quality inspection and help reduce waste.

---

## 🧠 Dataset
- **Source:** [`ulnnproject/Food-Freshness-Dataset`](https://www.kaggle.com/datasets/ulnnproject/food-freshness-dataset)
- **Classes:** Multiple freshness levels (e.g., *Fresh*, *Stale*, *Spoiled*)
- **Subset Used:** 10,000 images for fast training
- **Preprocessing Steps:**
  - Resized to 96×96 pixels  
  - Normalized (pixel values scaled to [0,1])  
  - Converted to RGB  
  - 80–20 train/validation split  

---

## 🏗️ Model Architecture
- **Framework:** TensorFlow / Keras  
- **Type:** Custom **Convolutional Neural Network (CNN)**  
- **Core Layers:**
  - Convolutional layers with ReLU activation  
  - MaxPooling layers  
  - Dropout for regularization  
  - Fully connected Dense layers for classification  
- **Optimizer:** Adam (Learning rate = 0.001)  
- **Loss Function:** Categorical Crossentropy  
- **Metrics:** Accuracy  
- **Training Configuration:**  
  - Batch Size: 128  
  - Epochs: 10  
  - Mixed precision enabled for faster training  

---

## ⚙️ Implementation Workflow
1. **Dataset Download & Setup**  
   - Dataset is fetched via the `kagglehub` API and preprocessed for uniformity.  

2. **Data Preprocessing**  
   - Images are resized, normalized, and loaded into memory.  
   - Labels are one-hot encoded.  

3. **Model Design & Compilation**  
   - A lightweight CNN is built and compiled with Adam optimizer and categorical loss.  

4. **Training & Validation**  
   - The model is trained for 10 epochs with an 80–20 split.  
   - Training and validation curves are plotted.  

5. **Evaluation**  
   - Performance metrics (accuracy, loss, confusion matrix) are computed.  

6. **Visualization**  
   - Displays sample predictions and model accuracy plots.  

---

## 📊 Results
- **Training Accuracy:** High convergence within 10 epochs  
- **Validation Accuracy:** Stable and consistent across categories  
- **Key Observations:**  
  - Mixed precision training drastically reduced runtime  
  - Model demonstrated robust classification across food types  

---

## 🧩 Technologies Used
| Category | Tools / Libraries |
|-----------|-------------------|
| Programming Language | Python |
| Framework | TensorFlow / Keras |
| Data Handling | NumPy, Pandas |
| Image Processing | OpenCV, Pillow |
| Visualization | Matplotlib |
| Model Evaluation | scikit-learn |
| Dataset Access | KaggleHub |

---

## 💻 How to Run
1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-username>/Food-Freshness-Detection.git
   cd Food-Freshness-Detection
