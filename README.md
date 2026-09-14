# Letter Recognition Neural Network

A neural network built with TensorFlow/Keras that classifies handwritten letters (A–Z) from 28x28 grayscale images, trained on the EMNIST Letters dataset.


## Overview

- **Task:** Multi-class classification predicts which letter (A–Z) a handwritten image represents
- **Dataset:** (https://www.kaggle.com/datasets/crawford/emnist)

## Data Pipeline

1. Load `emnist-letters-train.csv` and `emnist-letters-test.csv` with Pandas
2. Split into pixel features (`X`) and labels (`y`)
3. Normalize pixel values using training set mean and standard deviation
4. Reshape flattened pixel vectors into 28x28 images (with axis swap to correct EMNIST's orientation)
5. Shift labels down by 1 so classes are zero-indexed (0–25)

## Model Architecture

```
Input (28, 28)
→ Flatten
→ Dense(128, ReLU, L2 regularization)
→ Dense(64,  ReLU, L2 regularization)
→ Dense(32,  ReLU, L2 regularization)
→ Dense(26,  Softmax)
```

- **Loss:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Regularization:** L2 (0.0005) on each hidden layer to reduce overfitting
- **Epochs:** 10

## Results

<img width="1093" height="465" alt="image" src="https://github.com/user-attachments/assets/0a7c06aa-813b-4aa2-bda2-4a6b7cceaf13" />


## Custom Image Testing

The model can also predict on a custom handwritten letter image:

1. Load and convert the image to grayscale
2. Resize to 28x28
3. Normalize using the same training mean/std
4. Run through the model and map the predicted index back to a letter (A–Z)


Place `emnist-letters-train.csv`, `emnist-letters-test.csv`, and a test image (`img1.png`) in the same directory as the notebook.

