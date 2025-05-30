# File: main.py
from flask import Flask, render_template, request
import numpy as np
import joblib
import logging
import os
import gdown

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Google Drive file IDs
MODEL_ID = "1S9eNqqvmDluWVZtPs3-T3Veio2Ag_oA"
SCALER_ID = "1Ijq4hI4hCKW0_uv_8HkgFiqucnie4BkD"
FEATURES_ID = "1Q4jCVxw6claA8XjvonADuPwkqcG_YcJE"

def download_file(file_id, filename):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        logging.info(f"Downloading {filename} from {url}...")
        gdown.download(url, filename, quiet=False)
        logging.info(f"{filename} downloaded.")

# Ensure required files are available
download_file(MODEL_ID, "model_xgb.pkl")
download_file(SCALER_ID, "scaler.pkl")
download_file(FEATURES_ID, "features.pkl")

# Load models and features
model = joblib.load("model_xgb.pkl")  
scaler = joblib.load("scaler.pkl")
FEATURES = joblib.load("features.pkl")

FEATURE_DESCRIPTIONS = {
    "danceability": "How suitable a track is for dancing. Higher values may increase popularity.",
    "energy": "Intensity and activity. High energy songs often attract listeners.",
    "valence": "Musical positivity. Happier songs may be perceived as more popular.",
    "speechiness": "Presence of spoken words. Moderate levels may help in rap or talk tracks.",
    "acousticness": "Higher acousticness means more unplugged sound. Popularity varies by genre.",
    "tempo": "Speed in BPM. Moderate to upbeat tempos are usually more popular.",
    "duration": "Length of the song. Very long or short songs may affect attention."
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    input_values = [0.5] * len(FEATURES)
    low_confidence_warning = False
    high_precision_flag = False

    if request.method == "POST":
        try:
            app.logger.info(f"\U0001F4DD Received form keys: {list(request.form.keys())}")
            input_values = [float(request.form.get(f, 0.5)) for f in FEATURES]
            if len(input_values) != len(FEATURES):
                raise ValueError(f"Expected {len(FEATURES)} features, got {len(input_values)}")

            scaled = scaler.transform([input_values])
            proba = model.predict_proba(scaled)[0]
            confidence = round(100 * proba[1], 2)

            if proba[1] > 0.3:
                prediction = "Popular"
                if confidence > 80:
                    high_precision_flag = True
            else:
                prediction = "Not Popular"

            if confidence < 60:
                low_confidence_warning = True

        except Exception as e:
            prediction = f"Error: {str(e)}"
            confidence = "N/A"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        features=FEATURES,
        input_values=input_values,
        descriptions=FEATURE_DESCRIPTIONS,
        low_confidence_warning=low_confidence_warning,
        high_precision_flag=high_precision_flag,
        zip=zip
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
