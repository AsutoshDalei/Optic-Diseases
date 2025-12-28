# Ocular Diseases Classification using Deep Learning Techniques
## Project Overview
This project aims to develop a deep learning model for the classification of retinal images to identify various eye diseases. The focus is on four specific conditions: **Normal, Diabetic Retinopathy, Cataract, and Glaucoma**. 

By analyzing these images, we can assist in early diagnosis and treatment, ultimately improving patient outcomes.

## Dataset Description
The dataset used for this project comprises retinal images categorized into four classes:
* Normal: Healthy retinal images with no abnormalities
* Diabetic Retinopathy: Images showing signs of diabetic eye disease.
* Cataract: Images indicating the presence of cataracts in the lens of the eye.
* Glaucoma: Images that reveal signs of glaucoma, which affects the optic nerve.

Each class contains approximately 1,000 images, providing a balanced dataset for training and evaluation. The images have been sourced from various reputable databases, including:

* IDRiD: Indian Diabetic Retinopathy Image Dataset
* Ocular Recognition: A database focused on ocular disease recognition
* HRF: High-Resolution Fundus dataset

Source of Data:
https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data


## Algorithms & Models

In this project, we leverage CNN-based models to classify retinal images and detect specific eye diseases. Through deep learning, we aim to assist in early diagnosis and improve patient outcomes by automating the image analysis process.

Convolutional Neural Networks (CNNs) are a class of deep neural networks commonly used in computer vision tasks due to their ability to automatically and adaptively learn spatial hierarchies of features from images. Unlike traditional machine learning models, CNNs excel at capturing patterns like edges, textures, and shapes by using convolutional layers that scan small portions of the image with filters (kernels). These networks are particularly effective in tasks like image classification, object detection, and medical image analysis, where local patterns in the images play a critical role.

### 1. Custom Model 1 (TensorFlow)
A custom-built model using TensorFlow, designed to perform basic image classification tasks. It uses a relatively simple architecture with multiple convolutional layers.
- **Test Accuracy**: 25.28%, **Test F1 Score**: 25.23%, **Test Precision**: 25.25%, **Test Recall**: 25.22%

### 2. Custom Model 2 (TensorFlow)
A second custom model also built using TensorFlow, with slight modifications to the architecture for improved performance over the first model.
- **Test Accuracy**: 26.07%, **Test F1 Score**: 25.40%, **Test Precision**: 25.66%, **Test Recall**: 25.87%

### 3. AlexNet (PyTorch, Scratch Training)
AlexNet, a well-known deep convolutional neural network, trained from scratch using PyTorch. This model leverages a deeper architecture and a larger number of parameters for better feature extraction from the images.
- **Test Accuracy**: 72.47%, **Test F1 Score**: 71.66%, **Test Precision**: 73.92%, **Test Recall**: 72.95%

### 4. ResNet (PyTorch, Transfer Learning)
ResNet, a state-of-the-art deep learning architecture, was fine-tuned for this task using transfer learning with a pre-trained model on ImageNet. The model leverages residual connections to enable training of very deep networks.
- **Test Accuracy**: 85.76%, **Test F1 Score**: 85.12%, **Test Precision**: 86.33%, **Test Recall**: 85.59%

## Conclusion

Among the models tested, the **ResNet model with transfer learning** achieved the highest performance across all metrics, demonstrating superior ability in classifying retinal images. This makes it the most effective model for detecting eye diseases in this project.

## API Deployment

This project includes a production-ready Flask API for serving the trained model predictions via REST endpoints.

### Project Structure

```
/
├── api/                    # Flask API application
│   ├── __init__.py
│   ├── main.py            # Flask app factory
│   ├── routes.py          # API endpoints (Blueprint)
│   ├── schemas.py         # Response helpers
│   └── exceptions.py      # Custom exceptions
├── src/                    # Core application logic
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── inference.py       # Model loading and prediction
│   └── preprocessing.py   # Image preprocessing
├── docker/
│   └── Dockerfile         # Production Dockerfile
├── models/                 # Model files (ResNet50Model.pth)
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
├── app.py                 # Application entry point
└── README.md
```

### Prerequisites

- Docker and Docker Compose installed
- ResNet50 model file (`ResNet50Model.pth`) placed in the `models/` directory

### Quick Start with Docker

1. **Place the model file**:
   ```bash
   # Ensure ResNet50Model.pth is in the models/ directory
   ls models/ResNet50Model.pth
   ```

2. **Start the API service**:
   ```bash
   docker-compose up --build
   ```

3. **Verify the service is running**:
   ```bash
   curl http://localhost:5000/api/health
   ```

The API will be available at `http://localhost:5000`

### API Endpoints

#### Health Check

**GET** `/api/health`

Check the health status of the API and model loading status.

**Response**:
```json
{
  "status": "healthy",
  "service": "ocular-disease-classification-api",
  "model_loaded": true
}
```

#### Prediction

**POST** `/api/predict`

Classify an ocular disease from a retinal image.

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with `image` field containing the image file

**Supported formats**: PNG, JPG, JPEG, BMP, TIFF
**Max file size**: 10MB

**Example using curl**:
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@/path/to/retinal_image.jpg"
```

**Response**:
```json
{
  "success": true,
  "prediction": {
    "class": "normal",
    "confidence": 0.9234,
    "probabilities": {
      "cataract": 0.0123,
      "diabetic_retinopathy": 0.0456,
      "glaucoma": 0.0187,
      "normal": 0.9234
    }
  }
}
```

**Error Response**:
```json
{
  "success": false,
  "error": {
    "message": "Invalid image format or size",
    "code": "INVALID_IMAGE"
  }
}
```

### Configuration

Configuration can be managed via environment variables or a `.env` file. See `.env.example` for available options:

- `MODEL_PATH`: Path to the model file (default: `models/ResNet50Model.pth`)
- `MODEL_TYPE`: Model type - `pytorch` (default)
- `IMAGE_SIZE`: Target image size - `224,224` (default)
- `CLASS_NAMES`: Comma-separated class names
- `API_HOST`: API host address (default: `0.0.0.0`)
- `API_PORT`: API port (default: `5000`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `MODEL_DEVICE`: Device for inference - `cpu` or `cuda` (default: `cpu`)
- `MAX_IMAGE_SIZE`: Maximum image file size in bytes (default: `10485760`)

### Local Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

   Or using gunicorn for production-like setup:
   ```bash
   gunicorn --bind 0.0.0.0:5000 --workers 2 app:app
   ```

### Docker Configuration

The Docker setup uses a multi-stage build for optimized image size and includes:

- Python 3.10 slim base image
- Non-root user for security
- Health checks
- Gunicorn WSGI server with 2 workers
- Automatic restart policies

### Model Requirements

The API expects a ResNet50 PyTorch model file (`.pth`) trained for 4-class classification:
1. cataract
2. diabetic_retinopathy
3. glaucoma
4. normal

The model should accept input images of size 224x224 and output logits for 4 classes.

### Troubleshooting

**Model not found error**:
- Ensure `ResNet50Model.pth` exists in the `models/` directory
- Check the `MODEL_PATH` environment variable matches the actual file location

**Image upload errors**:
- Verify the image format is supported (PNG, JPG, JPEG, BMP, TIFF)
- Check that the file size is under 10MB
- Ensure the image is a valid image file

**API not responding**:
- Check Docker container logs: `docker-compose logs api`
- Verify the port 5000 is not already in use
- Check health endpoint: `curl http://localhost:5000/api/health`

### Production Deployment

For production deployment, consider:

- Using a reverse proxy (nginx) in front of the API
- Setting up SSL/TLS certificates
- Configuring proper logging and monitoring
- Using GPU support by setting `MODEL_DEVICE=cuda` and ensuring CUDA is available
- Scaling with multiple worker processes or containers
- Setting up proper backup and recovery procedures

