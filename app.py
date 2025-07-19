import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

# --- 1. Load the trained model ---
# We load the model once when the application starts
model = tf.keras.models.load_model('brain_tumor_vgg16.h5')
CATEGORIES = ['Tumor', 'Healthy']
IMG_SIZE = 150

def preprocess_image(image_path):
    """
    Takes an image path, reads it, and preprocesses it to be ready
    for the model.
    """
    try:
        # Read the image in color and convert it to RGB
        img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Resize it to the standard size
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        
        # Reshape for the model and normalize
        # The model expects a "batch" of images, so we add an extra dimension
        # Final shape: (1, 150, 150, 3)
        processed_image = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
        return processed_image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def classify_image(image_path):
    """
    Takes an image path, preprocesses it, and returns the model's prediction.
    """
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return "Error: Could not process image."

    # Get the model's prediction (a value between 0 and 1)
    prediction = model.predict(processed_image)[0][0]

    # --- Interpret the prediction ---
    # The model outputs a single number. 'Tumor' is class 0, 'Healthy' is class 1.
    # A low value (closer to 0) means 'Tumor'. A high value (closer to 1) means 'Healthy'.
    # We can use a threshold of 0.5.
    if prediction < 0.5:
        # It's more likely to be a Tumor (class 0)
        predicted_class = CATEGORIES[0] 
        confidence = 1 - prediction # Confidence in the 'Tumor' class
    else:
        # It's more likely to be Healthy (class 1)
        predicted_class = CATEGORIES[1]
        confidence = prediction # Confidence in the 'Healthy' class
    
    return f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}"

def upload_image():
    """
    This function is called when the user clicks the 'Upload Image' button.
    """
    # Open a file dialog to let the user choose an image
    file_path = filedialog.askopenfilename()
    if not file_path:
        return # User cancelled the dialog

    # Update the UI with the prediction
    result_text = classify_image(file_path)
    result_label.config(text=result_text)

    # Display the chosen image in the UI
    img = Image.open(file_path)
    img = img.resize((250, 250), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    
    image_label.config(image=img_tk)
    image_label.image = img_tk # Keep a reference to avoid garbage collection

# --- 2. Set up the GUI Window ---
root = tk.Tk()
root.title("Brain Tumor Detector")
root.geometry("400x400")
root.configure(bg='#f0f0f0')

# Create a title label
title_label = Label(root, text="Brain Tumor Detector", font=("Helvetica", 18, "bold"), bg='#f0f0f0')
title_label.pack(pady=10)

# Create a button to upload images
upload_btn = Button(root, text="Upload MRI Scan", command=upload_image, font=("Helvetica", 12), bg="#4a90e2", fg="white")
upload_btn.pack(pady=10)

# Create a label to display the image
image_label = Label(root, bg='#f0f0f0')
image_label.pack(pady=10)

# Create a label to display the result
result_label = Label(root, text="Prediction will appear here", font=("Helvetica", 14, "italic"), bg='#f0f0f0')
result_label.pack(pady=10)

# Start the main event loop
root.mainloop()