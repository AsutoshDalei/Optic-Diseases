import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "ResNet50Model.pth"))
    MODEL_TYPE = os.getenv("MODEL_TYPE", "pytorch")
    IMAGE_SIZE = tuple(map(int, os.getenv("IMAGE_SIZE", "224,224").split(",")))
    CLASS_NAMES = os.getenv("CLASS_NAMES", "cataract,diabetic_retinopathy,glaucoma,normal").split(",")
    
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "5000"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
    
    MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")
    
    @classmethod
    def validate(cls) -> None:
        if not os.path.exists(cls.MODEL_PATH):
            import logging
            logging.warning(f"Model file not found at {cls.MODEL_PATH}")
        if len(cls.CLASS_NAMES) != 4:
            raise ValueError("CLASS_NAMES must contain exactly 4 classes")
        if len(cls.IMAGE_SIZE) != 2:
            raise ValueError("IMAGE_SIZE must be in format 'width,height'")

