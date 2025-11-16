# Multi-Annotator Learning

This repository contains the implementation of our multi-annotator learning framework, providing core logic and essential components to support the reproducibility of our research findings.
The organized formal codes will be released upon acceptance.

## 📁 Structure

```
├── AMER_dataset.py           # Core training/testing/validation code for AMER dataset
├── STREET_dataset.py         # Core training/testing/validation code for STREET dataset  
├── process_dataset_mer.py    # AMER dataset splitting
├── process_dataset_street.py # STREET dataset splitting
├── environment.yml           # Conda environment configuration
└── README.md                # This file
```

## 🛠️ Environment Setup

# Prerequisites
- Anaconda or Miniconda
- Python 3.8+
# Installation
conda env create -f environment.yml

## 🚀 Usage

### Splitting AMER dataset
python process_dataset_mer.py
### Splitting STREET dataset
python process_dataset_street.py

### Run AMER dataset experiments under multi-gpu distributed mode
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29501 AMER_dataset.py
### Run STREET dataset experiments under multi-gpu distributed mode 
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29501 STREET_dataset.py

# 🔬 Reproducibility

1. Use the provided environment: Always run experiments using the `environment.yml` configuration
2. Fixed random seeds: All random operations use predefined seeds
3. Standardized data splits: Consistent train/validation/test partitioning
