import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/lightgbm_0.8106.pkl")
    MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.5"))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))
    API_PORT = int(os.getenv("API_PORT", "8000"))
