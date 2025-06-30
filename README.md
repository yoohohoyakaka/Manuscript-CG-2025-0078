

# CG-2025-0078 — Gas-Hydrate ML Workflows

Scripts & data for the *China Geology* submission:  
**“Machine-learning supervised algorithms for gas-hydrate identification and saturation estimation in marine reservoirs: a case study of NGHP-01-19B”**

> ⚠️ **Purpose:** This private repository is created **solely for peer review** of manuscript **CG-2025-0078**.  
> All files are provided to reproduce the main experiments; data is **not intended for redistribution**.

---

## Repository Structure

scripts/
├── Archie-based label/ # 12-model experiments using Archie resistivity-based labels
│ ├── classification/ # Includes 5-fold CV & group-CV runs
│ ├── regression/
│ └── data/ # Pre-processed XLSX for 1,350 samples
│
├── Gridsearch/ # Hyperparameter tuning for all 12 models
│ ├── *.py # Grid search runners
│ └── grid_data/ # Parameter logs and CV scores
│
└── Three-phase velocity-derived label/
├── classification/ # 12 models, velocity-derived labels
├── regression/
└── data/ # 890-sample XLSX set

README.md # This file

---

## Usage Instructions

This repository includes all scripts required to reproduce the classification and regression experiments under both labeling strategies.

### 1. Environment

Python 3.8+ is recommended. Install dependencies with:

```bash
pip install -r requirements.txt

---

## Usage Instructions

This repository includes all scripts required to reproduce the classification and regression experiments under both labeling strategies.

### 1. Environment

Python 3.8+ is recommended. Install dependencies with:

```bash
pip install -r requirements.txt

pandas
numpy
scikit-learn
catboost
lightgbm
xgboost
openpyxl


2. Running Experiments
Each subfolder contains classification/ and regression/ scripts. Please run scripts in sequence depending on the label type:

scripts/Archie-based label/

scripts/Three-phase velocity-derived label/

scripts/Gridsearch/ (optional tuning)

Outputs will be saved in model-specific folders or generated automatically.


Data Notes
Input data is pre-processed and stored in .xlsx format under each data/ folder.

Labels are derived from:

Archie resistivity

Three-phase velocity-based estimation

Raw well log data is not included due to size or licensing restrictions.

Contact
If you have any questions during the review process, please contact:

Wu Yifan
📧 Email: wyf0924@mail.ustc.edu.cn

License / Usage
This repository is made available only for peer review of manuscript CG-2025-0078.
Redistribution or reuse of the data or scripts is not permitted without the author's consent.