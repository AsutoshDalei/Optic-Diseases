class APIException(Exception):
    status_code = 400
    error_code = "GENERAL_ERROR"
    
    def __init__(self, message: str, status_code: int = None, error_code: str = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if error_code is not None:
            self.error_code = error_code

class ModelNotFoundException(APIException):
    status_code = 503
    error_code = "MODEL_NOT_FOUND"
    
    def __init__(self, message: str = "Model file not found"):
        super().__init__(message, self.status_code, self.error_code)

class ModelLoadException(APIException):
    status_code = 503
    error_code = "MODEL_LOAD_ERROR"
    
    def __init__(self, message: str = "Failed to load model"):
        super().__init__(message, self.status_code, self.error_code)

class InvalidImageException(APIException):
    status_code = 400
    error_code = "INVALID_IMAGE"
    
    def __init__(self, message: str = "Invalid image format or size"):
        super().__init__(message, self.status_code, self.error_code)

class PredictionException(APIException):
    status_code = 500
    error_code = "PREDICTION_ERROR"
    
    def __init__(self, message: str = "Failed to generate prediction"):
        super().__init__(message, self.status_code, self.error_code)

