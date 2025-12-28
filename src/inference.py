import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple, Optional
import logging
import os
from .config import Config

logger = logging.getLogger(__name__)

class ModelLoader:
    _instance: Optional['ModelLoader'] = None
    _model: Optional[nn.Module] = None
    _loaded: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._loaded:
            self._loaded = True
    
    def load_model(self) -> nn.Module:
        if self._model is not None:
            return self._model
        
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")
        
        try:
            device = torch.device(Config.MODEL_DEVICE)
            
            if Config.MODEL_TYPE.lower() == "pytorch":
                if Config.MODEL_PATH.endswith('.pth'):
                    model = torch.load(Config.MODEL_PATH, map_location=device)
                    if isinstance(model, nn.Module):
                        self._model = model
                    else:
                        raise ValueError("Loaded file is not a PyTorch model")
                else:
                    base_model = models.resnet50(weights=None)
                    base_model.fc = nn.Linear(2048, len(Config.CLASS_NAMES))
                    self._model = torch.load(Config.MODEL_PATH, map_location=device)
            else:
                raise ValueError(f"Unsupported model type: {Config.MODEL_TYPE}")
            
            self._model.eval()
            self._model.to(device)
            logger.info(f"Model loaded successfully from {Config.MODEL_PATH}")
            return self._model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, image_tensor: torch.Tensor) -> Dict[str, any]:
        if self._model is None:
            self.load_model()
        
        device = torch.device(Config.MODEL_DEVICE)
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = self._model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
            predicted_class = Config.CLASS_NAMES[predicted_idx]
        
        probabilities_dict = {
            class_name: float(prob)
            for class_name, prob in zip(Config.CLASS_NAMES, probabilities.cpu().numpy())
        }
        
        return {
            "class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": probabilities_dict
        }
    
    def is_loaded(self) -> bool:
        return self._model is not None

