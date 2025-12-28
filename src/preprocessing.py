import torch
from PIL import Image
import io
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0
        image_array = image_array.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        
        return image_tensor
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")

def validate_image_format(filename: str, allowed_extensions: set) -> bool:
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_extensions

def validate_image_size(image_bytes: bytes, max_size: int) -> bool:
    return len(image_bytes) <= max_size

