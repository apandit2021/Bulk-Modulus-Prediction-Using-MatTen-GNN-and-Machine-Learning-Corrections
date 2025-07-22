# Bulk Modulus Prediction Using MatTen GNN and Machine Learning Corrections

This repository provides a reproducible workflow for predicting the pressure-dependent bulk modulus of elemental crystalline solids using the MatTen equivariant graph neural network (GNN) and a suite of regression-based correction models. The objective is to deliver high-throughput, accurate, and physically meaningful elastic property prediction across varying pressure conditions, directly relevant to computational materials science and informatics.

---

## Table of Contents

- [Overview](#overview)
- [Scientific Motivation](#scientific-motivation)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
  - [1. MatTen GNN Elastic Tensor Prediction](#1-matten-gnn-elastic-tensor-prediction)
  - [2. Bulk Modulus Calculation](#2-bulk-modulus-calculation)
  - [3. Regression-Based Correction Models](#3-regression-based-correction-models)
- [Results and Analysis](#results-and-analysis)
  - [Prediction Accuracy](#prediction-accuracy)
  - [Extrapolation, Robustness, and Trends](#extrapolation-robustness-and-trends)
  - [Physical Interpretation](#physical-interpretation)
  - [Limitations](#limitations)
- [Usage Instructions](#usage-instructions)
- [Future Directions](#future-directions)
- [Contact](#contact)
- [References](#references)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project demonstrates a data-driven approach for predicting the bulk modulus—a fundamental mechanical property—of elemental crystals as a function of applied pressure. By leveraging MatTen, an equivariant graph neural network designed for tensorial properties, and supplementing it with regression-based corrections, the workflow achieves a substantial reduction in computational cost compared to direct DFT calculations, while maintaining high accuracy and physical interpretability.

---

## Scientific Motivation

Elastic moduli, especially the bulk modulus, are critical for understanding the mechanical behavior and phase stability of solids under extreme conditions. Traditional quantum-mechanical simulations (e.g., DFT) provide reliable results but are computationally prohibitive for systematic studies across many compositions and pressure ranges.

Recent advances in machine learning, particularly graph neural networks with physical symmetries, offer a promising pathway for rapid property prediction. However, pretrained GNNs may still exhibit systematic biases with respect to reference quantum data, especially under extrapolation. This work addresses these issues by:
- Quantifying the performance of the MatTen GNN for elemental systems under pressure,
- Applying machine learning regression models to correct systematic errors and calibrate predictions to DFT data,
- Evaluating both interpolation (within DFT-sampled pressure ranges) and extrapolation (beyond those ranges).

---

## Dataset Description

- **Systems**: Elemental crystals (examples: Fe, Cr, Mn) across several pressures.
- **Each data entry includes**:
  - `POSCAR` (crystal structure, VASP format)
  - `target.json` (DFT-computed properties, notably `bulk_modulus` in GPa)
- **Pressure Range**: Typically 0–200 GPa, sampled at discrete intervals per system.
- **MatTen Model Pre-training**:  
  - Trained on 10,276 elasticity tensors from the Materials Project, split 8:1:1 for training, validation, and testing.

The dataset thus combines realistic quantum-calculated properties with high-quality crystal representations, serving as a robust testbed for ML model benchmarking and property prediction.

---

## Methodology

### 1. MatTen GNN Elastic Tensor Prediction

- **Feature Construction**:
  - Atomic positions encoded as displacement vectors.
  - Atomic species encoded as one-hot vectors.
- **Network Architecture**:
  - Multiple equivariant GNN layers capture spatial symmetries and interactions.
  - Radial basis functions and spherical harmonics ensure geometric expressiveness.
  - Interaction blocks update atomic features via tensor operations and normalization.
- **Output**:
  - Full elasticity tensor (6x6 Voigt matrix), respecting symmetry constraints by design.
  - Predictions are structure-aware and physically plausible, even for systems not in the training set.

### 2. Bulk Modulus Calculation

- **Mathematical Expression**:  
  The Voigt average is used to convert the predicted tensor to a scalar bulk modulus:
  \[
  K_\text{Voigt} = \frac{C_{11} + C_{22} + C_{33} + 2(C_{12} + C_{13} + C_{23})}{9}
  \]
- **Interpretation**:  
  The Voigt average is a standard method for estimating isotropic moduli from full tensors, appropriate for polycrystalline or randomly oriented systems.

### 3. Regression-Based Correction Models

After obtaining MatTen-predicted bulk moduli, several supervised ML regressors are fit to the DFT reference data:
- **Linear Regression**: Captures global proportionality.
- **Polynomial Regression**: Captures systematic nonlinear deviations (degree 2–4).
- **Isotonic Regression**: Suitable for monotonic but nonlinear relationships.
- **k-Nearest Neighbors**: Captures local structure in the data.
- **Random Forest Regression**: Models complex, nonlinear dependencies and reduces overfitting.
- **Gaussian Process Regression**: Provides both prediction and uncertainty quantification.
- **Exponential Regression**: Tests for log-scale or exponential relationships.

**Features**:  
- MatTen-predicted bulk modulus,
- Pressure,
- Tensor norm (characterizing overall elastic response).

**Model Selection and Validation**:  
- Cross-validation and grid search are used to optimize hyperparameters.
- Models are evaluated using R² and MAE.

---

## Results and Analysis

### Prediction Accuracy

- **Raw MatTen Prediction**:  
  - Shows strong correlation (R² ≈ 0.9) with DFT reference.
  - Tends to systematically over- or under-predict, reflecting the domain gap between MP pretraining and new pressure conditions.
- **ML-Corrected Predictions**:  
  - Best correction models (typically Random Forest or Polynomial Regression) achieve R² > 0.98 and MAE as low as 1–3 GPa.
  - Residuals (DFT - prediction) are minimized and show no significant pressure-dependent drift post-correction.
  - Performance metrics for all models are compiled in `results_ml_ecs/bulkmod_correction_metrics.csv`.

### Extrapolation, Robustness, and Trends

- **Interpolation**:  
  - Corrected models closely match DFT within the observed pressure range, with small residuals and no evidence of overfitting.
- **Extrapolation**:  
  - The pipeline extrapolates trends beyond available DFT data, with MatTen providing plausible physical behavior (e.g., monotonic increase/decrease in bulk modulus under compression).
  - Caution: Extrapolated predictions should be validated with new DFT data or physical reasoning, as ML corrections may be unreliable far outside the training domain.
- **Systematic Trends**:  
  - The workflow is capable of capturing differences among elemental systems, reflecting distinct crystal structures and pressure dependencies.

### Physical Interpretation

- **Error Analysis**:  
  - The main source of residual error is the mismatch between MatTen's training domain and new pressure/structure combinations.
  - Regression correction is effective because errors are largely systematic rather than random.
- **Scientific Value**:  
  - The approach enables rapid screening of elastic properties for materials discovery or design under varying environmental conditions, supporting experimental prioritization and hypothesis generation.

### Limitations

- Dataset is currently limited to elemental solids; extension to multicomponent systems will require further validation.
- ML corrections are only as reliable as the underlying DFT reference data and their coverage of relevant phase space.
- Physical interpretability of ML corrections may be limited for highly nonlinear or outlier cases.

---

## Usage Instructions

### Dependencies

- Python 3.9+
- numpy, pandas, matplotlib, scikit-learn, pymatgen, matten

Install dependencies:
```bash
pip install -r requirements.txt
