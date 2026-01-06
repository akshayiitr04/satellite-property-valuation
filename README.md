# 🛰️ House Price Prediction using Tabular Data & Satellite Imagery
## 📌 Project Overview

This project aims to predict house prices by combining structured tabular data with satellite imagery.
While traditional models rely only on numerical features, real-world property valuation is also influenced by visual surroundings such as greenery, road density, urban planning, and proximity to water bodies.

To capture both numerical and spatial signals, a multi-modal deep learning model is built that fuses:

Tabular features processed using a neural network

Satellite images processed using a CNN (ResNet-18)

A unified regression head for final price prediction

The project also ensures model explainability using Grad-CAM, highlighting image regions that influence predictions.

# 📊 Dataset Description
## Tabular Data

The tabular dataset consists of numerical housing attributes such as:

Living area and above-ground area

Neighborhood statistics

Location-based numerical indicators

The target variable is house price, which is log-transformed to reduce skewness and stabilize model training.

Satellite Images

RGB satellite images mapped to each property using its unique id

Images capture spatial context including:

Green cover

Road connectivity

Building density

Urban vs suburban layout

# 🧠 Models Implemented
## 1️⃣ Tabular-Only Models (Baseline)

Models trained using only tabular features

Used as baseline to evaluate the benefit of satellite imagery

## 2️⃣ Tabular Neural Network

A Multi-Layer Perceptron (MLP)

Learns non-linear relationships among tabular features

## 3️⃣ Combined Model (Final Model)

CNN (ResNet-18) extracts visual features from satellite images

MLP extracts embeddings from tabular data

Both embeddings are concatenated and passed to a regression head

Outputs the final house price prediction

This approach enables the model to jointly learn numerical + spatial representations.

# 🏗️ Model Architecture (Fusion Strategy)

Architecture Flow:

Satellite Image → ResNet-18 → 512-dim visual embedding

Tabular Features → MLP → 64-dim tabular embedding

Feature Concatenation → Fully Connected Layers → Price Prediction

This fusion allows the model to capture how visual context complements numerical property features.

# ⚙️ Training Strategy

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Learning Rate: 3e-5

Early Stopping: Enabled to prevent overfitting

CNN Backbone: Frozen pretrained ResNet-18 (ImageNet)

# 🔍 Model Explainability with Grad-CAM

To ensure transparency, Grad-CAM is applied to the CNN component.

### Purpose of Grad-CAM

Highlights image regions that influence the model’s predictions

Verifies that the model focuses on meaningful spatial features

### Observed Insights

High-price properties: Attention on greenery, open spaces, waterfronts

Mid-price properties: Mixed attention on roads and residential layouts

Low-price properties: Dense construction and limited green areas

This confirms that the model learns real-world visual cues, not noise.

# 🛠️ Project Setup & Execution
## 1️⃣ Environment Requirements

Python 3.8+

Google Colab or local machine

GPU recommended for training (CPU works for prediction & Grad-CAM)
```bash
satellite_project/
│
├── data/
│   └── processed/
│       ├── train_processed.csv        # Processed training data
│       └── test_processed.csv         # Processed test data
│
├── images/
│   ├── raw/                            # Original train dataset satellite images (id.jpg)
│   └── processed/                     # Original test dataset satellite images (id.jpg)
│
├── data_fetcher.py                    # Script to download satellite images
│
├── preprocessing.ipynb                # Data cleaning & feature engineering
│
├── model_training.ipynb               # Model training (Tabular + CNN), evaluation, Grad-CAM
│
├── 22119005_final.csv                 # Final prediction file (id, predicted_price)|
│
├── 22119005_report.pdf                 # Project Report (PDF) **Overview **EDA **Financial/Visual Insights **Architecture Diagram **Results 
│
├── best_model.pth                     # Saved best trained model weights
│
├── README.md                          # Project setup, instructions & documentation
