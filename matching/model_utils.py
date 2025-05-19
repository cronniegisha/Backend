import os
import requests
import joblib  
from django.conf import settings

MODEL_URL = 'https://drive.google.com/uc?export=download&id=18g_MV77otCEWUn9ahiYNp0WrMeMXOUT8'
MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'rf_model.pkl')


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model downloaded and saved to:", MODEL_PATH)


def load_model():
    download_model()
    return joblib.load(MODEL_PATH)
