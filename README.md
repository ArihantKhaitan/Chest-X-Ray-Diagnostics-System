# 🩺 AI Chest X-Ray Diagnostics System

An advanced **Artificial Neural Network (ANN)** based medical diagnostic system that detects multiple pulmonary diseases from chest X-ray images using deep learning. This project implements and compares multiple CNN architectures for multi-class disease classification.

## 📋 Overview

This project uses deep learning to analyze chest X-ray images and classify them into four categories:
- **🫁 Pneumonia** - Acute lung inflammation from infection
- **🦠 Tuberculosis (TB)** - Bacterial infection of the lungs  
- **🎗️ Lung Cancer** - Malignant pulmonary growth
- **✅ Normal** - Healthy lung tissue

## ✨ Features

- **Multiple CNN Architectures**: Custom CNN, VGG16, ResNet50, and EfficientNet
- **Transfer Learning**: Leverages pre-trained models for improved accuracy
- **Fine-tuning**: Optimizes models for better performance
- **Interactive Web Interface**: Streamlit-based web application for real-time diagnosis
- **Comprehensive Evaluation**: Classification reports, confusion matrices, and ROC curves
- **Ensemble Methods**: Combines predictions from multiple models

## 🏗️ Project Structure

```
ANN Project/
├── src/
│   ├── app_streamlit.py      # Streamlit web application
│   ├── train.py               # Model training script
│   ├── models.py              # CNN model architectures
│   ├── data_prep.py           # Data preprocessing and augmentation
│   └── evaluate.py             # Model evaluation and metrics
├── saved_models/              # Trained model weights
│   ├── custom_cnn.h5
│   ├── vgg16.h5
│   ├── resnet50.h5
│   └── efficientnet.h5
├── multi_disease_chest_xray/  # Combined dataset
├── subsampled_chest_xray/     # Pneumonia dataset
├── TB_dataset/                # Tuberculosis dataset
├── LungCancer_dataset/         # Lung cancer dataset
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.15.0
- CUDA-compatible GPU (recommended for training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ArihantKhaitan/ANN-Project.git
   cd ANN-Project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup

The project uses multiple chest X-ray datasets:
- **Pneumonia Dataset**: From subsampled_chest_xray/
- **Tuberculosis Dataset**: From TB_dataset/
- **Lung Cancer Dataset**: From LungCancer_dataset/

The `data_prep.py` script automatically combines and preprocesses these datasets into a unified multi-class dataset.

## 💻 Usage

### 1. Data Preparation

Run the data preparation script to combine and preprocess datasets:

```bash
cd src
python data_prep.py
```

This will:
- Subsample images from each dataset
- Create train/validation/test splits
- Apply data augmentation
- Generate data generators for training

### 2. Model Training

Train all models:

```bash
python train.py
```

This will train:
- Custom CNN
- VGG16 (Transfer Learning)
- ResNet50 (Transfer Learning)
- EfficientNet (Transfer Learning)

Each model undergoes:
- Initial training (20 epochs)
- Fine-tuning (10 epochs)
- Test evaluation

### 3. Model Evaluation

Evaluate trained models and generate metrics:

```bash
python evaluate.py
```

This generates:
- Classification reports
- Confusion matrices
- ROC curves
- Ensemble predictions

### 4. Web Application

Launch the interactive Streamlit web app:

```bash
streamlit run app_streamlit.py
```

The web interface allows you to:
- Upload chest X-ray images
- Get real-time predictions from all models
- View model performance metrics
- Generate activation heatmaps
- See confidence scores

## 🧠 Model Architectures

### Custom CNN
- 3 convolutional layers with batch normalization
- Max pooling and dropout for regularization
- Fully connected layers for classification

### Transfer Learning Models
- **VGG16**: 16-layer deep CNN pre-trained on ImageNet
- **ResNet50**: 50-layer residual network
- **EfficientNet**: Efficient architecture with compound scaling

All transfer learning models use:
- Pre-trained ImageNet weights
- Global average pooling
- Custom classification head
- Fine-tuning on last 5 layers

## 📊 Performance Metrics

The models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Per-class metrics
- **Confusion Matrix**: Classification breakdown
- **ROC-AUC**: Receiver Operating Characteristic curves

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **OpenCV**: Image processing
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Evaluation metrics
- **NumPy**: Numerical computations

## 📝 Requirements

See `requirements.txt` for complete list:
- tensorflow==2.15.0
- streamlit
- opencv-python
- matplotlib
- pillow
- kaggle

## 📄 Research Papers

This project is based on research in medical AI and includes:
- ANN similarity analysis
- IEEE Pneumonia Detection research
- Comprehensive project reports

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

**Arihant Khaitan**
- GitHub: [@ArihantKhaitan](https://github.com/ArihantKhaitan)

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. The predictions made by this system should **not** be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 📜 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using Artificial Neural Networks**

