# Image Classification with ResNet18

## Overview

This project leverages ResNet18, a robust Convolutional Neural Network (CNN), to tackle small-scale content moderation. The focus is on binary classification of images into "violent" and "safe" categories using a curated dataset. ResNet18 demonstrates its ability to automate content filtering, enhancing scalability and efficiency even on smaller datasets.

## Project Structure

.
├── eval_ds/
│   ├── safe/
│   └── violent/
├── models/
│   └── resnet18.pth
├── results/
│   └── prediction_result.png
├── test/
├── bar_chart_visualization.py
├── confusion_matrix.png
├── eval.py
├── LICENSE
├── model.pth
├── README.md
├── test.py
├── train.py

## Dataset

### Description

- The dataset consists of two categories:
  - **Violent**: Graphic content featuring physical aggression.
  - **Safe**: Benign visuals such as landscapes and daily activities.

### Preprocessing

- Images resized to **256 × 256 pixels**.
- Augmentation techniques:
  - Rotations
  - Flips
- Normalization to ensure compatibility with ResNet18’s pre-trained requirements.
- Resized input dimensions for ResNet18: **224 × 224 × 3**.

## Model Details

### Architecture

- **Input Layer**: Processes normalized images.
- **Hidden Layers**:
  - Detect patterns
  - Reduce dimensionality
  - Classify features
- **Output Layer**:
  - Customized for binary classification.
  - Softmax activation to predict probabilities for "violent" or "safe" classes.

## Training

### Parameters
- **Batch Size**: 16
- **Epochs**: 10

### Results
- Training time: **21 minutes and 56 seconds**.
- Training loss reduced from **0.2007** to **0.0420**.
- Validation accuracy: **97.85%**.

### Performance Metrics
- Confusion matrix results:
  - True Positives (Safe): **1093**
  - True Positives (Violent): **54**
  - False Positives: **3**
  - False Negatives: **12**

## Evaluation

- ResNet18 achieved high accuracy, demonstrating its effectiveness for content moderation tasks.
- Areas for improvement include further reducing false negatives to enhance classification reliability.

## How It Works

1. **Fully Connected Layer**: Produces two logits (e.g., "violent" and "safe").
2. **Softmax Activation**: Converts logits to probabilities.
3. **Prediction**: The class with the higher probability is selected as the output.

## Implementation Details

- **Training Code**: Adapted from the open-source repository [PyTorch Image Classification Repository](https://github.com/anilsathyan7/pytorch-image-classification).
- **Dataset**: [Kaggle Dataset: Graphical Violence and Safe Images](https://www.kaggle.com/datasets/kartikeybartwal/graphical-violence-and-safe-images-dataset)
- Custom modifications:
  - Data loading
  - Preprocessing
  - Augmentation pipelines

## Usage

This project was developed using Python 3.10.

1. Clone the repository.
2. Install dependencies using:
   ```
   pip install torch torchvision matplotlib numpy pandas openpyxl
   ```
3. Run the training script:
   ```
   python train.py
   ```
4. Evaluate the model:
   ```
   python eval.py
5. Test the model with personal dataset (in this case (9 images)):
   ```
   python test.py
   ```

## Future Work

- Explore additional augmentation techniques to enhance generalization.
- Experiment with other CNN architectures for comparison.
- Implement advanced techniques to further reduce false negatives.

## Acknowledgments

- [PyTorch Image Classification Repository](https://github.com/anilsathyan7/pytorch-image-classification)
- [Kaggle Dataset: Graphical Violence and Safe Images](https://www.kaggle.com/datasets/kartikeybartwal/graphical-violence-and-safe-images-dataset)

## License

This project is licensed under the MIT License.

© 2025 isyiraliyah. All Rights Reserved.
