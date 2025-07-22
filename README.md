# Bulk Modulus Prediction Using MatTen GNN with Machine Learning Corrections

---

This repository provides a reproducible workflow for predicting the pressure-dependent bulk modulus of elemental crystalline solids. It combines **MatTen**, an equivariant Graph Neural Network (GNN), with a suite of regression-based correction models to enhance its predictions. The primary goal is to offer a high-throughput, accurate, and physically meaningful approach for elastic property prediction across varying pressure conditions, directly supporting computational materials science and informatics.

---

## Table of Contents

* [Overview](#overview)
* [Scientific Motivation](#scientific-motivation)
* [Dataset Description](#dataset-description)
* [Methodology](#methodology)
    1.  [MatTen GNN Elastic Tensor Prediction](#1-marten-gnn-elastic-tensor-prediction)
    2.  [Bulk Modulus Calculation](#2-bulk-modulus-calculation)
    3.  [Regression-Based Correction Models](#3-regression-based-correction-models)
* [Results and Analysis](#results-and-analysis)
    * [Prediction Accuracy](#prediction-accuracy)
    * [Extrapolation, Robustness, and Trends](#extrapolation-robustness-and-trends)
    * [Physical Interpretation](#physical-interpretation)
* [Limitations](#limitations)
* [Usage Instructions](#usage-instructions)
* [Future Directions](#future-directions)
* [Contact](#contact)
* [References](#references)
* [Acknowledgments](#acknowledgments)

---

## Overview

This project presents a data-driven approach to predict the **bulk modulus**—a fundamental mechanical property—of elemental crystals as a function of applied pressure. By integrating **MatTen**, an equivariant GNN designed for tensorial properties, with regression-based corrections, this workflow significantly reduces the computational cost compared to direct DFT calculations, while maintaining high accuracy and physical interpretability.

---

## Scientific Motivation

Elastic moduli, particularly the bulk modulus, are crucial for understanding the mechanical behavior and phase stability of solids under extreme conditions. While traditional quantum-mechanical simulations (e.g., Density Functional Theory - DFT) offer reliable results, they are computationally prohibitive for systematic studies across a wide range of compositions and pressure conditions.

Recent advancements in machine learning, especially physically-informed Graph Neural Networks, offer a promising avenue for rapid property prediction. However, pre-trained GNNs may exhibit systematic biases when applied to new domains or under extrapolation. This work addresses these challenges by:

* Quantifying the performance of the **MatTen** GNN for elemental systems under pressure.
* Applying machine learning regression models to correct systematic errors and calibrate predictions against DFT data.
* Evaluating the model's performance for both **interpolation** (within DFT-sampled pressure ranges) and **extrapolation** (beyond those ranges).

---

## Dataset Description

This project utilizes a custom dataset of elemental crystal structures evaluated at various pressures. This dataset is specifically designed for benchmarking and developing machine learning models for elastic property prediction, such as the bulk modulus.

* **Systems:** A diverse range of elemental crystals (e.g., Fe, Cr, Mn, Mo, Nb, Rh, Ru, Sr, Tc, Y, Zr, Rb), each evaluated under hydrostatic pressure.
* **Pressure Range:** Typically spans from 0 to 200 GPa, with data sampled at discrete intervals for each system.
* **Properties per Entry:** Each entry includes:
    * **POSCAR:** Atomic structure in standard VASP format.
    * **target.json:** DFT-computed physical properties, notably:
        * `"bulk_modulus"` (in GPa, the primary target property for ML).
        * `"elastic_tensor"` (6x6 Voigt matrix).
        * Additional fields such as pressure, free energy, ELF descriptors, and lattice constants.
* **MatTen Model Pre-training:** The **MatTen** GNN model is pre-trained on a comprehensive dataset of 10,276 elasticity tensors from the Materials Project. This pre-training utilizes an 8:1:1 train/validation/test split, ensuring robust generalization.

This integrated dataset, combining quantum-computed properties with standardized structure files, forms a robust foundation for data-driven elastic property prediction and model evaluation.

### Data Organization and Format

The dataset should be organized into the following directory structure:
gnn_dataset/
├── Mo_0.0GPa/
│   ├── POSCAR
│   └── target.json
├── Mo_10.0GPa/
│   ├── POSCAR
│   └── target.json
├── Nb_0.0GPa/
│   ├── POSCAR
│   └── target.json
...
