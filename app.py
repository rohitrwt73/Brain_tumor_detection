
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

IMG_SIZE = 150

model = tf.keras.models.load_model("brain_tumor_vgg16.h5")

def predict(image):
    image = np.array(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    image = image / 255.0

    prediction = float(
        model.predict(image, verbose=0)[0][0]
    )

    healthy_probability = prediction
    tumor_probability = 1 - prediction

    if prediction < 0.5:
        result = "Tumor"
        confidence = tumor_probability
    else:
        result = "Healthy"
        confidence = healthy_probability

    return (
        result,
        f"{confidence * 100:.2f}%",
        {
            "Tumor": float(tumor_probability),
            "Healthy": float(healthy_probability)
        }
    )

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
        gr.Label(label="Probabilities")
    ],
    title="🧠 Brain Tumor Detection",
    description="Upload an MRI image to detect brain tumors using a VGG16 deep learning model."
)

demo.launch()

