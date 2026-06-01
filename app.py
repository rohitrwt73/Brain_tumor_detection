from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = 150

print("Loading VGG16 model...")
vgg_model = tf.keras.models.load_model("brain_tumor_vgg16.h5")
print("VGG16 model loaded successfully.")


def preprocess_image(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img_array is None:
        raise ValueError("Unable to read image.")

    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    img_array = img_array / 255.0

    return img_array


def classify_image(image_path):
    processed_image = preprocess_image(image_path)

    prediction = float(
        vgg_model.predict(processed_image, verbose=0)[0][0]
    )

    healthy_probability = prediction
    tumor_probability = 1 - prediction

    if prediction < 0.5:
        predicted_class = "Tumor"
        confidence = tumor_probability
    else:
        predicted_class = "Healthy"
        confidence = healthy_probability

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2),
        "tumor_probability": round(tumor_probability * 100, 2),
        "healthy_probability": round(healthy_probability * 100, 2),
        "vgg_prediction": round(prediction * 100, 2)
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        if "image" not in request.files:
            return render_template(
                "index.html",
                error="Please upload an MRI image."
            )

        file = request.files["image"]

        if file.filename == "":
            return render_template(
                "index.html",
                error="Please select an MRI image."
            )

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        result = classify_image(filepath)

        return render_template(
            "index.html",
            image_path="/" + filepath.replace("\\", "/"),
            prediction=result["prediction"],
            confidence=result["confidence"],
            tumor_probability=result["tumor_probability"],
            healthy_probability=result["healthy_probability"],
            vgg_prediction=result["vgg_prediction"],
            hybrid_prediction="N/A"
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

