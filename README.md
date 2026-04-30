# Handwritten Digit Recognition using LeNet-5 on MNIST dataset

## Overview
This project implements an enhanced LeNet-5 CNN architecture for handwritten digit recognition on the MNIST dataset using PyTorch. The model achieves **99.65% test accuracy**.

## Architecture

<img width="945" height="516" alt="image" src="https://github.com/user-attachments/assets/59f8b068-c73f-4055-8299-5cbb62db991e" />

- 3 Convolutional blocks with filters: 32 → 64 → 128
- BatchNorm2d after each conv layer
- MaxPooling (2×2)
- ReLU activations
- Fully connected layers: 3200 → 256 → 10
- Dropout (0.5) for regularization
- ~949K trainable parameters

## Enhancements over Original LeNet-5
| Component | Original LeNet-5 | This Model |
|-----------|-----------------|------------|
| Activation | Sigmoid/Tanh | ReLU |
| Pooling | Average | Max |
| Normalization | None | BatchNorm |
| Regularization | None | Dropout (0.5) |
| Conv Blocks | 2 | 3 |
| Filters | 6, 16 | 32, 64, 128 |

## Training Setup
- Optimizer: Adam (lr=1e-3, weight decay=1e-4)
- LR Schedule: Cosine Annealing (1e-3 → 1e-5)
- Batch size: 128
- Epochs: 20
- Data augmentation: Random rotation ±10°, affine transforms

## Results
- **Test Accuracy: 99.65%**
- Only 35 misclassifications out of 10,000 test images
- Macro-averaged F1-score: 1.00
- Surpassed 98% target at Epoch 1

## Ablation Study
Removing BatchNorm dropped accuracy from 99.67% → 99.65%, confirming BatchNorm's contribution to faster early convergence.
