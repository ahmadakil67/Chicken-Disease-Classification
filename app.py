from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin

import os
import subprocess

from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline


os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")


app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

        self.classifier = PredictionPipeline(
            filename=self.filename
        )


# Create object globally
clApp = ClientApp()


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
@cross_origin()
def trainRoute():
    try:
        subprocess.run(
            ["python", "main.py"],
            check=True
        )

        return jsonify({
            "message": "Training completed successfully!"
        })

    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "Training failed",
            "details": str(e)
        }), 500


@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "error": "Image data is required"
            }), 400

        encoded_image = data["image"]

        decodeImage(
            encoded_image,
            clApp.filename
        )

        result = clApp.classifier.predict()

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )