import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import sys

# --- 1. Load the trained models ---
# This application requires both model files to be in the same directory.
try:
    print("Loading VGG16 model...")
    vgg_model = tf.keras.models.load_model('brain_tumor_vgg16.h5')
    print("Loading Hybrid CNN-Transformer model...")
    # When loading a model with custom components like a Transformer,
    # you might need to provide a 'custom_objects' dictionary.
    # However, Keras often handles this automatically if the model was saved correctly.
    hybrid_model = tf.keras.models.load_model('brain_tumor_hybrid_cnn_transformer.h5')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure both 'brain_tumor_vgg16.h5' and 'brain_tumor_hybrid_cnn_transformer.h5' are in the same folder as this script.")
    sys.exit()


CATEGORIES = ['Tumor', 'Healthy']
IMG_SIZE = 150

def preprocess_image(image_path):
    """
    Takes an image path, reads it, and preprocesses it to be ready
    for the models.
    """
    try:
        img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        processed_image = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
        return processed_image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def classify_image(image_path):
    """
    Takes an image path, preprocesses it, gets predictions from both models,
    averages them, and returns the final classification.
    """
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return "Error: Could not process image."

    # --- ENSEMBLE LOGIC ---
    # 1. Get prediction from the VGG16 model
    vgg_prediction = vgg_model.predict(processed_image)[0][0]
    
    # 2. Get prediction from the Hybrid CNN-Transformer model
    hybrid_prediction = hybrid_model.predict(processed_image)[0][0]
    
    # 3. Average the predictions to get the ensemble result
    ensemble_prediction = (vgg_prediction + hybrid_prediction) / 2.0

    # --- Interpret the final ensemble prediction ---
    if ensemble_prediction < 0.5:
        predicted_class = CATEGORIES[0] # Tumor
        confidence = 1 - ensemble_prediction
    else:
        predicted_class = CATEGORIES[1] # Healthy
        confidence = ensemble_prediction
    
    return f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}"

def upload_image():
    """
    This function is called when the user clicks the 'Upload Image' button.
    """
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return # User cancelled

        # Update the UI with the prediction
        result_text = classify_image(file_path)
        result_label.config(text=result_text)

        # Display the chosen image in the UI
        img = Image.open(file_path)
        img = img.resize((250, 250), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        image_label.config(image=img_tk)
        image_label.image = img_tk
    except Exception as e:
        result_label.config(text=f"An error occurred:\n{e}")


# --- 2. Set up the GUI Window ---
root = tk.Tk()
root.title("Hybrid Ensemble Brain Tumor Detector")
root.geometry("400x450")
root.configure(bg='#f0f4f8') # A professional light grey-blue

# Create a title label
title_label = Label(root, text="Hybrid Ensemble Tumor Detector", font=("Helvetica", 16, "bold"), bg='#f0f4f8')
title_label.pack(pady=15)

# Create a button to upload images
upload_btn = Button(root, text="Upload MRI Scan", command=upload_image, font=("Helvetica", 12), bg="#4a69bd", fg="white", relief=tk.FLAT, padx=10, pady=5)
upload_btn.pack(pady=10)

# Create a label to display the image
image_label = Label(root, bg='#f0f4f8')
image_label.pack(pady=10)

# Create a label to display the result
result_label = Label(root, text="Prediction will appear here", font=("Helvetica", 14, "italic"), bg='#f0f4f8', fg='#1e272e')
result_label.pack(pady=10)

# Start the main event loop
root.mainloop()
