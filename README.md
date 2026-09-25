# Alzheimer's Disease Classification Using Deep Neural Networks

A deep learning project classifying Alzheimer's disease severity from MRI brain scans into four stages, comparing an ANN baseline against multiple pretrained CNN architectures. Built as a team project (Group 18).

## Problem Statement

Alzheimer's disease is a progressive neurological disorder affecting millions worldwide. Early and accurate diagnosis from MRI brain scans is critical but difficult — manual analysis by doctors is time-consuming, costly, and prone to human error. An automated deep learning system is needed to classify Alzheimer's severity quickly and accurately from MRI images.

## Solution

We started with an ANN as a baseline, but since ANN cannot effectively learn spatial image patterns, we moved to CNN architectures purpose-built for image recognition. Using pretrained CNN models (EfficientNetB0, MobileNetV2, DenseNet121, ResNet50, VGG16), the system learns deep features from MRI images and classifies them into four stages: **NonDemented, VeryMildDemented, MildDemented, ModerateDemented.**

## Use Case

Designed to assist hospitals and diagnostic centers with early Alzheimer's detection — helping radiologists classify MRI scans faster and more accurately, reducing diagnosis time and human error, and supporting faster treatment decisions.

## Dataset

- **Name:** Alzheimer's Multiclass Dataset
- **Classes:** MildDemented, ModerateDemented, NonDemented, VeryMildDemented
- **Total images:** 44,000
- **Test set:** 8,800 images (2,000 MildDemented, 2,000 ModerateDemented, 2,560 NonDemented, 2,240 VeryMildDemented)

## My Contribution — ANN & MobileNetV2

I implemented and evaluated two models:

### ANN (Baseline)

| Metric | Score |
|---|---|
| Accuracy | 0.6281 |
| Precision | 0.6292 |
| Recall | 0.6281 |
| F1 Score | 0.6038 |

The ANN model showed moderate, balanced performance but was limited by its inability to learn spatial/pixel-relationship features — validation loss fluctuated across epochs, indicating instability and weaker generalization. Confirmed by recomputing all four metrics directly from the confusion matrix — all match to 4 decimal places.

### CNN — MobileNetV2

| Metric | Score |
|---|---|
| Accuracy | 0.890 |
| Precision | 0.890 |
| Recall | 0.890 |
| F1 Score | 0.890 |

MobileNetV2 significantly outperformed the ANN baseline (~89% vs. ~63%), with smooth, steadily decreasing training/validation loss curves — confirming stable convergence and minimal overfitting despite its lightweight architecture. Independently verified against the confusion matrix — exact match.

Both models were evaluated on the identical 8,800-image test set, making the ANN vs. MobileNetV2 comparison directly fair.




## Key Findings

Deep CNN architectures significantly outperformed the ANN baseline across all metrics. DenseNet121 achieved the best results (~91.5%), followed closely by MobileNetV2 (~89%), EfficientNetB0 (~88.5%), and ResNet50 (~87.8%), while VGG16 (~83%) and ANN (~62%) trailed behind. CNN models showed smooth loss convergence and strong class separation in their confusion matrices, while ANN exhibited unstable validation performance and significant inter-class confusion — confirming that hierarchical spatial feature learning is essential for accurate MRI-based classification.

## Tech Stack

- Python
- TensorFlow / Keras 
- Pretrained CNN architectures: EfficientNetB0, MobileNetV2, DenseNet121, ResNet50, VGG16

 ## Limitations & Future Improvements

- The dataset used ("equal-and-augmented") was pre-augmented before download, meaning our validation split may contain near-duplicate images of training samples (augmented copies of the same original scan). This could mean real-world accuracy on genuinely unseen MRI scans is somewhat lower than the reported ~89% for MobileNetV2.
- **Next step:** split the original (non-augmented) images into train/val/test first, then apply augmentation only to the training set — this would give a cleaner, leakage-free measure of generalization.
- Despite this, the relative ranking of models (ANN << VGG16 < ResNet50 < EfficientNetB0 ≈ MobileNetV2 < DenseNet121) is still meaningful, since all models were evaluated under identical conditions — the comparison between architectures remains valid even if the absolute accuracy numbers carry some optimism.

## References

- Multi-Modal Diagnosis of Alzheimer's Disease Using Interpretable Graph Convolutional Networks
- GKAN: Explainable Diagnosis of Alzheimer's Disease Using Graph Neural Network
