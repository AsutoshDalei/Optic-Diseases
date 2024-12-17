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

