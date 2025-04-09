# gan_server.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
generator = tf.keras.models.load_model("generator_model.h5")

@app.route("/generate", methods=["POST"])
def generate():
    seed = request.json.get("seed", 42)
    tf.random.set_seed(seed)
    noise = tf.random.normal([1, 128])
    generated_image = generator(noise, training=False)[0]
    image_array = ((generated_image + 1) * 127.5).numpy().astype("uint8")

    image = Image.fromarray(image_array)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({"image": encoded})