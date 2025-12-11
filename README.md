# CIFAR-100 Image Classification – Baseline CNN (PyTorch)

This project implements a clean, professional, and fully reproducible pipeline for training and evaluating a Convolutional Neural Network (CNN) on the **CIFAR-100** dataset.

The repository includes:
- A modular **PyTorch training pipeline** (`src/train.py`)
- A clean **project architecture** following ML engineering best practices
- GPU evaluation on **Google Colab** with GitHub integration
- Two notebooks for dataset exploration and model evaluation
- Checkpoint saving and reproducible results

---

## 🎯 Project Objectives

- Build a simple but solid **baseline CNN model** for CIFAR-100  
- Train it locally using a modular and extensible structure  
- Evaluate accuracy and visualize predictions  
- Prepare a project ready for **portfolios, interviews, and GitHub showcase**

---

## 📁 Project Structure

cifar100_project/
│
├── notebooks/
│ ├── O1_data_exploration.ipynb # Dataset exploration
│ └── O2_evaluation.ipynb # Model evaluation & visualization
│
├── src/
│ ├── train.py # Training script (PyTorch)
│ ├── models/
│ │ └── baseline_cnn.py # CIFAR100BaselineCNN model
│ └── utils/
│ ├── dataset_utils.py # Data loaders + transforms
│ └── init.py
│
├── checkpoints/ # Saved model weights (.pt / .pth)
├── data/ # CIFAR-100 auto-download directory
├── requirements.txt
└── README.md

---

## 🚀 Training the Model (Local)

From the project root:

```bash
python src/train.py --epochs 20 --batch-size 128 --lr 1e-3
The script will:

download CIFAR-100 (if needed)

create dataloaders

train for the required number of epochs

save the best checkpoint in checkpoints/

Evaluation (Notebook)
notebooks/O2_evaluation.ipynb
The notebook performs:

Loading the trained checkpoint

Computing test accuracy

Visualizing predictions and true labels

Displaying 8 random images with model predictions
Example accuracy after baseline training:
Test accuracy: XX.XX%
(Accuracy depends on training duration and hardware.)
🖼️ Example Prediction Visualization

8 random test images are displayed with:

Predicted label

True label

This helps assess model performance beyond accuracy.

Run on Google Colab (GPU)

You can use Colab to evaluate the model or retrain with GPU acceleration.

Steps:

Open a new notebook in Google Colab

Clone the repository:

!git clone https://github.com/<your_username>/cifar100_project.git


Install dependencies:

!pip install -r cifar100_project/requirements.txt


Run evaluation inside Colab:

from src.models.baseline_cnn import CIFAR100BaselineCNN


A Colab badge can also be added here later.

Baseline CNN Architecture

The baseline model includes:

3× Conv blocks

BatchNorm layers

MaxPool

Fully connected classifier

Designed for:

clarity

reproducibility

serving as a starting point for deeper models (ResNet, ViT…)

Next Improvements (Future Work)

Add ResNet18/34 comparison

Add training resume with --resume option

Add wandb or TensorBoard logging

Add confusion matrix + per-class accuracy

Upload final models to Hugging Face Hub

Implement hyperparameter search (Optuna)

Credits

Developed by Khalid Oqaidi
Course project – deep learning pipeline for CIFAR-100.
Includes PyTorch, torchvision, and a clean engineering structure.