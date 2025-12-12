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

## 🧠 Discussion

The experimental results show that the model trained on Google Colab using GPU achieves a higher overall test accuracy compared to the locally trained model. This improvement is mainly attributed to longer training duration, faster convergence enabled by GPU acceleration, and more stable optimization.

Interestingly, during qualitative inspection on a small subset of test images, the locally trained model occasionally produced more visually convincing predictions than the GPU-trained model. This apparent discrepancy does not contradict the quantitative results and can be explained by several factors.

First, qualitative visualization is performed on a very small sample of images, which is not statistically representative of the entire CIFAR-100 test set. Due to sampling variance, a weaker model may appear superior on a limited number of examples purely by chance.

Second, CIFAR-100 is a challenging dataset with 100 fine-grained classes, many of which are visually similar. As training progresses, the GPU-trained model learns more complex and specialized decision boundaries that improve overall generalization but may occasionally misclassify visually intuitive examples.

Finally, the observed behavior does not indicate overfitting of the locally trained model. Overfitting is characterized by a large gap between training and test performance, which was not observed here. Instead, the results highlight the importance of relying on quantitative evaluation metrics computed on the full test set rather than qualitative inspection alone.

Overall, the comparison confirms that GPU-accelerated training leads to better generalization performance, while qualitative analysis remains a useful but complementary diagnostic tool.


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