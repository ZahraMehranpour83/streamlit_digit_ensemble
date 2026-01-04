# Streamlit Digit Ensemble 🎯

A handwritten digit recognition application built with **scikit-learn pipelines**
and **ensemble learning (Voting Classifier)**, featuring an interactive
**Streamlit** user interface.

This project supports **training models, saving/loading them, and predicting
digits from user-uploaded images**.

---

## ✨ Features
- Data normalization using `StandardScaler`
- Dimensionality reduction with `PCA`
- Ensemble learning via `VotingClassifier`
- Interactive UI built with Streamlit
- Save and load trained models (`.pkl` / `.joblib`)
- Detect and predict multiple digits from a single image

---

## 🧠 Model Architecture
The core machine learning pipeline is structured as:

##🖼️ How to Use
Choose Train Model or Load Pretrained Model
If training, save the trained model
Upload an image containing handwritten digits
View detected digits and their predicted labels
##📦 Input Image Format
Supported formats: jpg, jpeg, png
Images may contain multiple digits
Each digit is extracted and classified individually
##🔧 Possible Improvements
Add HOG or other advanced image features
Improve image preprocessing and segmentation
Include more models in the ensemble
Export predictions to a file
##🛠️ Technologies Used
Python
scikit-learn
OpenCV
Streamlit
Joblib
