# CIFAR-100 Image Classification – Baseline CNN (PyTorch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oqaidik/cifar100_CNN/blob/main/notebooks/O2_evaluation.ipynb)

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

yaml
Copier le code

---

## 🚀 Training the Model (Local)

From the project root:

```bash
python src/train.py --epochs 20 --batch-size 128 --lr 1e-3
The script will:

Download CIFAR-100 (if needed)

Create dataloaders

Train for the required number of epochs

Save the best checkpoint in checkpoints/

📊 Evaluation (Notebook)
Notebook: notebooks/O2_evaluation.ipynb

It performs:

Loading the trained checkpoint

Computing test accuracy

Visualizing predictions and true labels

Displaying 8 random images with model predictions

Example accuracy after baseline training:
Test accuracy: XX.XX%
(Depends on hardware and number of epochs.)

🖼️ Example Prediction Visualization
8 random test images are displayed with:

True label

Predicted label

This helps assess model performance beyond accuracy.

🚀 Run on Google Colab (GPU)
Steps:

bash
Copier le code
!git clone https://github.com/oqaidik/cifar100_CNN.git
!pip install -r cifar100_CNN/requirements.txt
Then open:

Copier le code
notebooks/O2_evaluation.ipynb
🧠 Baseline CNN Architecture
The model includes:

3× Conv blocks

BatchNorm layers

MaxPool

Fully connected classifier

Designed for:

clarity

reproducibility

serving as a starting point for deeper models (ResNet, ViT…)

🔮 Next Improvements (Future Work)
Add ResNet18/34 comparison

Add training resume with --resume

Add wandb or TensorBoard logging

Add confusion matrix + per-class accuracy

Upload final models to Hugging Face Hub

Implement hyperparameter search (Optuna)

👤 Credits
Developed by Khalid Oqaidi
Includes PyTorch, torchvision, and a clean engineering structure.