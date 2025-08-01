# CortexAI: Hybrid Ensemble Brain Tumor Detector 🧠

This project is a desktop application that uses a sophisticated deep learning ensemble model to classify brain MRI scans as either "Tumor" or "Healthy." It combines the strengths of a pre-trained VGG16 model with a custom-built hybrid CNN-Transformer model to achieve high accuracy and robust performance. The application features a user-friendly graphical interface built with Tkinter for real-time image analysis.


Screenshots:
![App Screenshot](/app_images/bt1.png)
![App Screenshot](/app_images/bt2.png)
![App Screenshot](/app_images/bt3.png)

---

## ✨ Features

- **Upload & Classify:** Easily upload an MRI scan image from your computer.
- **Instant Prediction:** Get a real-time classification of "Tumor" or "Healthy."
- **Confidence Score:** View the model's confidence in its prediction.
- **Ensemble Power:** Utilizes an advanced ensemble of two distinct deep learning models for enhanced accuracy.
- **Intuitive GUI:** A simple and clean interface built with Python's Tkinter library.

---

## 🤖 Models Used

This project employs an **ensemble learning** strategy by averaging the predictions of two powerful models:

1.  **VGG16 (Transfer Learning):** A deep Convolutional Neural Network (CNN) pre-trained on the ImageNet dataset. It excels at recognizing general image features, providing a strong baseline for classification.
    - **Validation Accuracy:** 98.85%

2.  **Hybrid CNN-Transformer:** A custom-built model that uses a CNN base to extract local features (edges, textures) and a Transformer encoder to analyze the global relationships between these features, providing a holistic understanding of the image.
    - **Validation Accuracy:** 59.24%

**Ensemble Performance:**
By combining the predictions of these two models, the final ensemble achieves a superior accuracy of **99.51%**, demonstrating how a weaker, diverse model can help correct the errors of a stronger one.

---

## 🛠️ Setup and Installation

1. Install Dependencies
This project requires the following libraries. You can install them using pip:

```pip install tensorflow opencv-python Pillow```

2. Download the Pre-trained Models
You will need the two trained model files (.h5). Place them in the root directory of the project:

brain_tumor_vgg16.h5

brain_tumor_hybrid_cnn_transformer.h5

🚀 How to Run
Once the setup is complete, you can launch the application by running the app.py script:

```python app.py```

The Tkinter window will open, and you can begin uploading MRI scans for classification.

📂 File Structure
```
├── app.py                                  # The main Tkinter application script
├── brain_tumor_vgg16.h5                    # The saved VGG16 model
├── brain_tumor_hybrid_cnn_transformer.h5   # The saved CNN-Transformer model
├── README.md                               # This README file
└── (optional) requirements.txt             # File listing dependencies
```
