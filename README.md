# Aircraft Damage Classification Using Deep Learning

**Binary Image Classification | Computer Vision | Transfer Learning | TensorFlow/Keras**

## Overview

This project focuses on classifying aircraft surface damage into two categories: **Dent** and **Crack**, using deep learning and computer vision techniques.

The model is built using **VGG16**, a convolutional neural network pretrained on ImageNet. Transfer learning is applied to extract visual features from aircraft images, followed by a custom neural network classifier trained to distinguish between the two damage categories.

The project demonstrates a complete image classification workflow, including data preprocessing, model development, training, evaluation, and prediction visualization.

## Project Background

This project was completed as the final assignment of the **[Introduction to Deep Learning & Neural Networks with Keras](https://www.coursera.org/learn/introduction-to-deep-learning-with-keras)** course offered by **IBM on Coursera**.

It provided hands-on experience in implementing an image classification pipeline with TensorFlow and Keras, applying transfer learning, and evaluating a pretrained convolutional neural network on a practical computer vision task.

This repository documents my implementation and learning experience as part of the course.

## Technologies & Tools

| Technology | Application |
|---|---|
| Python | Model implementation and data processing |
| TensorFlow / Keras | Deep learning model development |
| VGG16 | Pretrained CNN for feature extraction |
| NumPy | Numerical computations |
| Matplotlib | Training and prediction visualization |
| ImageDataGenerator | Image loading and preprocessing |
| Jupyter Notebook | Experimentation and documentation |

## Dataset

The project uses the **Aircraft Damage Dataset**, provided through the IBM course materials.

Images are labeled into two categories:

- **Dent:** Surface indentations or deformations
- **Crack:** Visible fractures or cracks on aircraft surfaces

The dataset is organized into three subsets:

```text
aircraft_damage_dataset_v1/
├── train/
│   ├── dent/
│   └── crack/
├── valid/
│   ├── dent/
│   └── crack/
└── test/
    ├── dent/
    └── crack/
```

### Data Preprocessing

Before training, the images are prepared using Keras `ImageDataGenerator`:

- Resize images to 224 × 224 pixels
- Normalize pixel values to the range [0, 1]
- Use a batch size of 32
- Apply binary class labels
- Separate training, validation, and test data

A random seed of 42 is used to support reproducibility.

## Model Architecture

The classification model is based on **VGG16 with transfer learning**.

Rather than training a convolutional neural network from scratch, the project uses VGG16 pretrained on ImageNet as a fixed feature extractor.

The original classification layers are removed and replaced with a custom binary classification head.

### Architecture

```text
Input Image (224 × 224 × 3)
          |
          v
  Pretrained VGG16
  (ImageNet Weights)
          |
          v
  Frozen CNN Layers
          |
          v
       Flatten
          |
          v
    Dense (512, ReLU)
          |
          v
      Dropout (0.3)
          |
          v
    Dense (512, ReLU)
          |
          v
      Dropout (0.3)
          |
          v
    Dense (1, Sigmoid)
          |
          v
    Binary Prediction
     Dent / Crack
```

### Transfer Learning

The pretrained VGG16 convolutional layers are frozen during training, preserving their learned feature representations.

Only the newly added classification layers are trained on the aircraft damage dataset.

This approach allows the model to reuse general image features without retraining the entire convolutional network.

## Model Training

The model is trained using the following configuration:

| Hyperparameter | Value |
|---|---|
| Base architecture | VGG16 |
| Pretrained weights | ImageNet |
| Input size | 224 × 224 × 3 |
| Batch size | 32 |
| Epochs | 5 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Binary Cross-Entropy |
| Activation function (output) | Sigmoid |
| Evaluation metric | Accuracy |

Training and validation performance are monitored during the learning process.

## Model Evaluation

After training, the model is evaluated using a separate test dataset.

The evaluation includes:

- **Training Loss:** Monitoring optimization during training
- **Validation Loss:** Assessing performance on unseen validation data
- **Training and Validation Accuracy:** Tracking classification performance
- **Test Accuracy and Loss:** Evaluating the trained model on test images
- **Prediction Visualization:** Comparing predicted and actual image labels

### Prediction Visualization

The notebook includes functionality to display test images alongside their true and predicted damage categories.

A sigmoid output greater than 0.5 is mapped to class 1, while an output of 0.5 or below is mapped to class 0.

This provides a visual way to inspect individual predictions.

**Note:** Numerical evaluation results should be taken directly from a completed notebook execution rather than assumed from the model architecture.

## Project Workflow

The implementation follows six main stages:

1. **Dataset Preparation** — Download and extract the labeled aircraft damage images.
2. **Image Preprocessing** — Resize, normalize, and load images in batches.
3. **Model Development** — Configure pretrained VGG16 and add custom classification layers.
4. **Model Training** — Train the classification head using the training dataset.
5. **Performance Evaluation** — Examine learning curves and evaluate the test data.
6. **Prediction Visualization** — Inspect predicted and true labels for sample images.

## Getting Started

### Prerequisites

- Python
- Jupyter Notebook or VS Code with Jupyter support
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pillow

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/negin79/aircraft-damage-classification-vgg16.git
cd aircraft-damage-classification-vgg16
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

**4. Open the notebook**

```bash
jupyter notebook "Classification Problem.ipynb"
```

**5. Execute the notebook**

Skip the notebook's original installation cell (cell 2) after installing requirements.txt; it also installs packages for other course tasks that this classification notebook does not use, and its CPU-specific packages may be unavailable on macOS. Run the remaining cells sequentially to download the dataset, preprocess images, train the model, evaluate its performance, and visualize predictions.

The dataset is downloaded automatically from the course-provided source.

## Future Improvements

Potential improvements to extend this project beyond the original coursework include:

- **Model Comparison:** Evaluate alternative CNN architectures such as ResNet50 and EfficientNet.
- **Data Augmentation:** Explore rotation, brightness adjustments, and other transformations to improve generalization.
- **Advanced Evaluation:** Add precision, recall, F1-score, and confusion matrices.
- **Model Explainability:** Implement Grad-CAM to visualize which image regions contribute to predictions.
- **Hyperparameter Optimization:** Experiment with learning rates, batch sizes, dropout, and training epochs.
- **Fine-Tuning:** Unfreeze selected convolutional layers to investigate whether task-specific feature adaptation improves performance.

## Key Learning Outcomes

Through this project, I gained practical experience in:

- Implementing convolutional neural networks using TensorFlow and Keras
- Applying transfer learning to image classification
- Using pretrained deep learning models for feature extraction
- Preparing image datasets for supervised learning
- Building custom neural network classification heads
- Training and evaluating binary classification models
- Visualizing learning curves and image predictions

The project strengthened my understanding of deep learning workflows and their applications to computer vision problems.

## Limitations

This is a course-based educational project intended to demonstrate transfer learning for image classification.

The model has not been validated for real-world aircraft inspection or safety-critical applications. Additional evaluation, dataset diversity, and independent validation would be required before considering such use.

## Acknowledgments

This project was developed as part of the IBM course:

**[Introduction to Deep Learning & Neural Networks with Keras — Coursera](https://www.coursera.org/learn/introduction-to-deep-learning-with-keras)**

The aircraft damage dataset and the original assignment context were provided through the course materials.

---

**Author:** Negin Hannani

**Focus Areas:** Deep Learning | Computer Vision | Machine Learning | AI Engineering

## Repository Notes and Reproducibility

- The uploaded notebook is preserved unchanged, including its original setup cell and saved outputs. No completed training curves or numerical test results are stored in it; this repository does not report an accuracy claim. Training has not been rerun for publication.
- `requirements.txt` lists dependencies used by this classification workflow. Use Python 3.10 or 3.11; this environment specification has not been runtime-tested here. The portable `tensorflow` package replaces the notebook's CPU-only installation command.
- Internet access is needed for the course dataset and ImageNet weights. The download cell removes an existing `aircraft_damage_dataset_v1` directory before extracting the archive; run in a dedicated project folder.
- The original evaluation uses floor division for `steps`, which can omit the final partial test batch. For a full-test evaluation, replace it with `test_loss, test_accuracy = model.evaluate(test_generator)` before reporting results.
- The notebook uses rescaling by 1/255, rather than VGG16's application-specific `preprocess_input`. Comparing preprocessing choices is a possible future experiment.
- Seed 42 supports repeatability but does not guarantee identical results across systems.
- The dataset and trained weights are not included. Course materials and dataset retain their original rights; no redistribution license is asserted by this repository.

