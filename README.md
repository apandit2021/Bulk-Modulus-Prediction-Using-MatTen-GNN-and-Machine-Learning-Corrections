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

This project uses a dataset of elemental crystal structures evaluated at various pressures, suitable for benchmarking and developing machine learning models for elastic property prediction.

- **Systems:**  
  A range of elemental crystals (e.g., Fe, Cr, Mn, Mo, Nb, Rh, Ru, Sr, Tc, Y, Zr, Rb) evaluated under hydrostatic pressure.
- **Pressure Range:**  
  Typically from 0 to 200 GPa, sampled at discrete intervals for each system.
- **Properties per Entry:**  
  - **`POSCAR`**: Atomic structure in standard VASP format.
  - **`target.json`**: DFT-computed physical properties, notably:
    - `"bulk_modulus"` (in GPa, target property for ML)
    - `"elastic_tensor"` (6x6 Voigt matrix)
    - Additional fields: pressure, free energy, ELF descriptors, lattice constants, etc.
- **MatTen Pre-training:**  
  The MatTen GNN model is pre-trained on 10,276 elasticity tensors from the Materials Project, with an 8:1:1 train/validation/test split, ensuring robust generalization.

The dataset thus integrates quantum-computed properties with standardized structure files, forming a robust basis for data-driven elastic property prediction and model evaluation.

---

### Data Organization and Format

The dataset should be organized in the following directory structure:

gnn_dataset/
├── Mo_0.0GPa/
│ ├── POSCAR
│ └── target.json
├── Mo_10.0GPa/
│ ├── POSCAR
│ └── target.json
├── Nb_0.0GPa/
│ ├── POSCAR
│ └── target.json
...

- **Naming Convention:**  
  Each subdirectory is named as `{ElementSymbol}_{Pressure}GPa` (e.g., `Mo_0.0GPa` for molybdenum at 0 GPa).

- **File Contents:**  
  - **POSCAR:** VASP format crystal structure.
  - **target.json:** DFT-computed properties and physical descriptors.

#### Example: `gnn_dataset/Mo_0.0GPa/`

**POSCAR**
<details>
<summary>Click to expand</summary>

```text
BCC2                                    
   1.00000000000000     
     3.1592668255831322   -0.0000000000000000    0.0000000000000000
     0.0000000000000000    3.1592668255831322    0.0000000000000000
    -0.0000000000000000   -0.0000000000000000    3.1592668255831322
   Mo
     2
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.5000000000000000  0.5000000000000000  0.5000000000000000

  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00

</details>
target.json

<details> <summary>Click to expand</summary>
{
  "system": "Mo",
  "pressure": 0.0,
  "atomic_numbers": [42, 42],
  "avg_elf": 0.3212510371425438,
  "octa_elf": 0.3609,
  "tetra_elf": 0.48302,
  "lattice_constant": 3.159266825583132,
  "lattice_volume": 31.53253753615866,
  "free_energy": -21.8470852,
  "bulk_modulus": 262.63581666666664,
  "elastic_tensor": [
    [470.09325, 158.9071, 158.9071, -0.0, -0.0, 0.0],
    [158.9071, 470.09325, 158.9071, -0.0, 0.0, 0.0],
    [158.9071, 158.9071, 470.09325, -0.0, -0.0, -0.0],
    [-0.0, -0.0, -0.0, 103.80556, 0.0, 0.0],
    [-0.0, 0.0, -0.0, 0.0, 103.80556, 0.0],
    [0.0, 0.0, -0.0, 0.0, 0.0, 103.80556]
  ],
  "avg_electronegativity": 2.16,
  "avg_valence_electrons": 12.0,
  "avg_rwigs_angstrom": 1.455
}
</details>

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
