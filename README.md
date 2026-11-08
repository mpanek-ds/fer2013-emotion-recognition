# Facial Expression Recognition (FER-2013) using CNNs and Transfer Learning

## 📌 Project Overview
This project focuses on the automated classification of human facial expressions into 7 emotional states: *Angry, Disgust, Fear, Happy, Neutral, Sad,* and *Surprise*. Operating on the **FER-2013 dataset** (approx. 35,000 "in-the-wild" grayscale images of 48x48 pixels), the pipeline addresses key computer vision challenges including low spatial resolution, high intra-class variance, visual noise, and severe class imbalance.

## 🎯 Key Engineering Challenges & Solutions
* **Severe Class Imbalance:** Exploratory Data Analysis (EDA) revealed a major disproportion between classes (e.g., over 7,200 training samples for *Happy* vs. only 436 for *Disgust*). To prevent the models from ignoring minority classes, inverse class frequency weights were computed and integrated directly into the **Weighted Cross-Entropy Loss** (`nn.CrossEntropyLoss`).
* **Preventing Data Leakage:** Since the original dataset only provides `train` and `test` directories, the training set was randomly split (`seed = 82`) into **85% training (24,402 images)** and **15% validation (4,307 images)**. The test set (**7,178 images**) was kept completely isolated for final evaluation.
* **Dual DataLoader Strategy:** Built two separate preprocessing and augmentation pipelines (`RandomHorizontalFlip`, `RandomRotation`) tailored to different architectural requirements:
  * **Custom CNN Pipeline:** Operates on native `1x48x48` grayscale tensors normalized to `[-1, 1]`.
  * **Transfer Learning Pipeline:** Converts images to 3-channel RGB, upscales them to `3x224x224` using bicubic interpolation, and applies standard ImageNet normalization.

## 🧠 Model Architectures & Results
Three distinct convolutional architectures were implemented, trained, and compared in **PyTorch**:

1. **MimicNet (Custom VGG-Lite CNN):**
   * A custom modular network inspired by VGGNet, designed specifically for `48x48` inputs.
   * Consists of 3 convolutional blocks (`64 -> 128 -> 256` channels). Each block uses stacked double `3x3` convolutional layers with `BatchNorm2d` and `ReLU` activations before `MaxPool2d` (`2x2`) and `Dropout (0.35)` to capture complex facial features while mitigating overfitting.
   * Topped with a fully connected classifier (`512` units + `BatchNorm1d` + `Dropout`).
2. **ResNet-18 (Transfer Learning & Fine-Tuning):**
   * Initialized with pre-trained ImageNet weights, leveraging residual skip connections.
   * Froze early feature extraction layers, selectively unfreezing the final convolutional block (`layer4`) and replacing the fully connected (`fc`) head for 7-class classification.
3. **EfficientNet-B0 (Transfer Learning & Fine-Tuning):**
   * Initialized with pre-trained ImageNet weights utilizing compound scaling and MBConv blocks.
   * Selectively unfroze the top decision blocks (`features[6]`, `features[7]`, `features[8]`) alongside a custom linear classifier head.

### 📊 Model Performance (Test Set)
| Model | Architecture Type | Test Accuracy |
| :--- | :--- | :--- |
| **MimicNet** | Custom VGG-Lite | **65.13%** |
| **EfficientNet-B0** | Transfer Learning (Fine-Tuned) | 62.15% |
| **ResNet-18** | Transfer Learning (Fine-Tuned) | 59.13% |

*(Note: Custom lightweight CNN outperformed heavily parameterized pre-trained models on this low-resolution dataset).*

## ⚙️ Training Strategy
* **Optimizer:** Adam with architecture-specific learning rates (`1e-3` for MimicNet, lower rates for fine-tuning pre-trained backbones).
* **Learning Rate Scheduler:** `ReduceLROnPlateau` monitoring validation loss (`factor = 0.1`, `patience = 5`).
* **Regularization:** Early Stopping (`patience = 7`) with automatic checkpointing (`best_model_*.pth`).

## 🛠️ Technology Stack
* **Language:** Python
* **Deep Learning:** PyTorch (`torch`, `torch.nn`, `torch.optim`), `torchvision` (`datasets`, `transforms`, `models`)
* **Data Manipulation & Metrics:** NumPy, Pandas, Scikit-learn (`confusion_matrix`, `classification_report`)
* **Visualization & Image Processing:** Matplotlib, Seaborn, Pillow (PIL)

## 🚀 Repository Structure & How to Run
* `fer2013_emotion_recognition.ipynb` – Complete Jupyter Notebook containing EDA, data pipelines, model definitions, training loops, learning curves, and final evaluation.

**To reproduce the experiments:**
1. Clone this repository.
2. Download the official dataset from Kaggle: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).
3. Name the downloaded archive `FER-2013.zip` and place it in the root directory (the notebook automatically extracts it into `dataset_emotions/`).
4. Run `fer2013_emotion_recognition.ipynb` in Google Colab or a local GPU environment.

*(Note: The Jupyter Notebook containing the code, theoretical background, and EDA is written in Polish as it was part of a university submission, but the code structure and logic follow universal data science standards).*

## 👥 Contributors
* **Mateusz Panek**
* **Piotr Jasiak**
