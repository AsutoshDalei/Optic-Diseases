from flask import Blueprint, request, jsonify
import logging
from werkzeug.exceptions import RequestEntityTooLarge

from .schemas import prediction_response, health_response, error_response
from .exceptions import InvalidImageException, PredictionException, ModelNotFoundException, ModelLoadException
from src.inference import ModelLoader
from src.preprocessing import preprocess_image, validate_image_format, validate_image_size
from src.config import Config

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__)

model_loader = ModelLoader()

@bp.route("/health", methods=["GET"])
def health():
    try:
        model_loaded = model_loader.is_loaded()
        if not model_loaded:
            try:
                model_loader.load_model()
                model_loaded = True
            except Exception:
                model_loaded = False
        return jsonify(health_response(status="healthy", model_loaded=model_loaded)), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify(health_response(status="unhealthy", model_loaded=False)), 503

@bp.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return error_response("No image file provided", "MISSING_IMAGE", 400)
        
        file = request.files["image"]
        
        if file.filename == "":
            return error_response("Empty filename", "INVALID_IMAGE", 400)
        
        if not validate_image_format(file.filename, Config.ALLOWED_EXTENSIONS):
            return error_response(
                f"Invalid file format. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}",
                "INVALID_IMAGE_FORMAT",
                400
            )
        
        image_bytes = file.read()
        
        if not validate_image_size(image_bytes, Config.MAX_IMAGE_SIZE):
            return error_response(
                f"Image too large. Max size: {Config.MAX_IMAGE_SIZE} bytes",
                "IMAGE_TOO_LARGE",
                400
            )
        
        image_tensor = preprocess_image(image_bytes, Config.IMAGE_SIZE)
        
        prediction = model_loader.predict(image_tensor)
        
        return jsonify(prediction_response(prediction)), 200
        
    except RequestEntityTooLarge:
        return error_response("File too large", "FILE_TOO_LARGE", 413)
    except InvalidImageException as e:
        return error_response(str(e), e.error_code, e.status_code)
    except (ModelNotFoundException, ModelLoadException) as e:
        return error_response(str(e), e.error_code, e.status_code)
    except PredictionException as e:
        return error_response(str(e), e.error_code, e.status_code)
    except ValueError as e:
        logger.error(f"Value error during prediction: {str(e)}")
        return error_response(str(e), "INVALID_INPUT", 400)
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        return error_response("Internal server error", "INTERNAL_ERROR", 500)

