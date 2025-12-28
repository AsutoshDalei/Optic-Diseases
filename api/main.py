from flask import Flask
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler
import os

from .routes import bp
from .exceptions import APIException
from .schemas import error_response
from src.config import Config

def create_app() -> Flask:
    app = Flask(__name__)
    
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_IMAGE_SIZE
    
    CORS(app)
    
    app.register_blueprint(bp, url_prefix="/api")
    
    configure_logging(app)
    register_error_handlers(app)
    
    Config.validate()
    
    return app

def configure_logging(app: Flask) -> None:
    if not app.debug:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "app.log"),
            maxBytes=10240000,
            backupCount=10
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
            )
        )
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper()))
        app.logger.info("Application startup")

def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(APIException)
    def handle_api_exception(error: APIException):
        return error_response(error.message, error.error_code, error.status_code)
    
    @app.errorhandler(404)
    def handle_not_found(error):
        return error_response("Endpoint not found", "NOT_FOUND", 404)
    
    @app.errorhandler(413)
    def handle_request_entity_too_large(error):
        return error_response("Request entity too large", "FILE_TOO_LARGE", 413)
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        return error_response("Internal server error", "INTERNAL_ERROR", 500)

