# 🌿 Plant Disease Detection using Deep Learning

This repository contains a plant disease detection system built using deep learning techniques, trained on the **PlantVillage Dataset** from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease).

## 📽️ Demo Video

Watch the working demo here:

https://github.com/user-attachments/assets/23f08c23-5832-4a25-86f0-c707bda26f33


---

## 📂 Dataset - PlantVillage

The **PlantVillage Dataset** is a curated dataset of over 50,000 images of healthy and diseased crop leaves categorized into 38 different classes. It was published on Kaggle and is widely used for building plant disease detection models.

- 📊 Classes: 38
- 🌱 Plant species: Apple, Tomato, Grape, Potato, etc.
- 🐞 Disease labels: Healthy, Early Blight, Late Blight, Leaf Mold, etc.
- 📁 Image format: JPG
- 📐 Size: ~2 GB

Dataset Link: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## 🧠 Project Overview

The goal of this project is to accurately classify plant leaf images into healthy or diseased categories using a Convolutional Neural Network (CNN).

### 🔧 Features

- Deep learning model using TensorFlow/Keras (or PyTorch)
- Preprocessing pipeline for image normalization and augmentation
- Training and evaluation scripts
- Model accuracy and performance metrics
- Real-time prediction script (optional: web app or GUI)

---

## 🚀 How to Use

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/plant-disease-detection.git
   cd plant-disease-detection
Download the dataset
Download from Kaggle and extract it into the data/ directory.

Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Train the model

bash
Copy
Edit
python train.py
Predict

bash
Copy
Edit
python predict.py --image sample.jpg
📊 Results
Accuracy: 92%+ on validation set

Precision, Recall, and F1-score also evaluated

Confusion matrix for class-level performance

🙌 Contributing
Pull requests are welcome! If you find bugs or have ideas to improve, feel free to open an issue or submit a PR.

📜 License
This project is licensed under the MIT License. See LICENSE for more details.

📬 Contact
Created by Bhanu Srinija Pasupuleti
