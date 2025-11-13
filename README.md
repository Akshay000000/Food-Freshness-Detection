Project Title:

Food Freshness Detection using Deep Learning and Image Analysis

Overview:

This project focuses on automatically identifying the freshness level of food items using image-based analysis. By leveraging a Convolutional Neural Network (CNN) model trained on the Food Freshness Dataset from Kaggle, the system classifies food samples (such as fruits, vegetables, and other perishable goods) into freshness categories.

The goal is to develop a fast, accurate, and efficient AI model that can assist in food quality inspection and help reduce waste.

Dataset:

Name: ulnnproject/Food-Freshness-Dataset (Kaggle)

Classes: Multiple freshness categories (e.g., fresh, stale, spoiled)

Subset used: 10,000 images for fast training and experimentation

Preprocessing:

Image resizing to 96×96 pixels

Normalization (pixel values scaled to [0,1])

RGB conversion

Train-validation split (80%-20%)

Model Details:

Framework: TensorFlow / Keras

Architecture: Custom Convolutional Neural Network (CNN)

Layers include:

Convolutional + ReLU activation

MaxPooling for spatial reduction

Dropout for regularization

Dense layers for classification

Optimizer: Adam (Learning rate = 0.001)

Loss Function: Categorical Crossentropy

Metrics: Accuracy

Training Configuration:

Batch Size: 128

Epochs: 10

Mixed precision enabled for faster training

Implementation Workflow:

Dataset Download & Setup
The dataset is downloaded using the kagglehub API and preprocessed for uniform size and format.

Data Preprocessing
Each image is resized, normalized, and loaded into memory. Labels are one-hot encoded.

Model Design & Compilation
A lightweight CNN model is built for multi-class classification. The model is compiled with Adam optimizer and categorical loss.

Training & Validation
The model is trained for 10 epochs using an 80-20 train-validation split, displaying accuracy and loss curves.

Evaluation
After training, the model is evaluated on unseen validation data to determine classification accuracy.

Visualization

Sample predictions visualized with predicted labels

Confusion matrix and training graphs plotted using Matplotlib

Results Summary:

Training accuracy: High convergence within 10 epochs

Validation accuracy: Stable generalization across food types

Observations:

Effective use of mixed precision reduced training time significantly

Model demonstrated strong performance on visual classification

Dependencies:

TensorFlow / Keras

NumPy

OpenCV

Matplotlib

scikit-learn

Pillow

KaggleHub

How to Run:

Clone the repository:

git clone https://github.com/<your-username>/Food-Freshness-Detection.git
cd Food-Freshness-Detection


Install required dependencies:

pip install -r requirements.txt


Run the notebook:

jupyter notebook AIproject.ipynb


Follow the step-by-step cells for dataset loading, model training, and evaluation.

Future Enhancements:

Deploy the model via a web or mobile interface for real-time testing

Expand dataset with more food types

Introduce freshness scoring rather than categorical classification

Integrate transfer learning (e.g., MobileNet, EfficientNet) for higher accuracy
