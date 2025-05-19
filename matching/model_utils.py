import os
import joblib
import gdown
from django.conf import settings


MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'rf_model.pkl')
MODEL_URL = os.getenv('MODEL_URL')  # Read from Railway env variable

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    else:
        print("Model already exists. Skipping download.")

def load_model():
    download_model()
    return joblib.load(MODEL_PATH)