from typing import Dict, Any, Optional
from flask import jsonify

def prediction_response(prediction_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": True,
        "prediction": {
            "class": prediction_data.get("class"),
            "confidence": prediction_data.get("confidence"),
            "probabilities": prediction_data.get("probabilities", {})
        }
    }

def health_response(status: str = "healthy", model_loaded: bool = False) -> Dict[str, Any]:
    return {
        "status": status,
        "service": "ocular-disease-classification-api",
        "model_loaded": model_loaded
    }

def error_response(message: str, error_code: str = "GENERAL_ERROR", status_code: int = 400) -> tuple:
    response = {
        "success": False,
        "error": {
            "message": message,
            "code": error_code
        }
    }
    return jsonify(response), status_code

def success_response(data: Any = None, message: Optional[str] = None) -> Dict[str, Any]:
    response = {"success": True}
    if message:
        response["message"] = message
    if data:
        response["data"] = data
    return response

