🛰️ House Price Prediction using Tabular Data & Satellite Imagery
📌 Project Overview

This project aims to predict house prices by combining structured tabular data with satellite imagery.
While traditional models rely only on numerical features, real-world property valuation is also influenced by visual surroundings such as greenery, road density, urban planning, and proximity to water bodies.

To capture both numerical and spatial signals, a multi-modal deep learning model is built that fuses:

Tabular features processed using a neural network

Satellite images processed using a CNN (ResNet-18)

A unified regression head for final price prediction

The project also ensures model explainability using Grad-CAM, highlighting image regions that influence predictions.
