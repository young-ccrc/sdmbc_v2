# SDMBCv2

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Sub-Daily Multivariate Bias Correction Version 2 (SDMBCv2)** is a Python package for statistical bias correction of Global Climate Model (GCM) outputs prior to dynamical downscaling.

It addresses systematic biases in GCMs to enhance the physical realism of Regional Climate Model (RCM) simulations.

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start Guide](#quick-start-guide)
  - [1. Prepare Input Data (Interpolation)](#1-prepare-input-data-interpolation)
  - [2. Perform Bias Correction](#2-perform-bias-correction)
  - [3. Reformat Outputs](#3-reformat-outputs)
- [License](#license)
- [Citation](#citation)

---

## Key Features

- Supports bias correction for both historical and future simulations.
- Handles daily, monthly, seasonal, and sub-daily scales.
- Flexible interpolation tools for observational and model datasets.
- Parallel processing with Dask for efficient large-scale computations.
- Reformatting utilities to match original GCM NetCDF structures.

---

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your_username/sdmbc_v2.git
cd sdmbc_v2
python -m venv sdmbc_env
source sdmbc_env/bin/activate    # Linux/macOS
sdmbc_env\Scripts\activate       # Windows
pip install -r Requirements.txt
```

---

## Quick Start Guide

### 1. Prepare Input Data (Interpolation)

- **3D Atmospheric Fields** (e.g., humidity, temperature):

```bash
python src/interp_obs2gcm.py --yp configs/config_interp.yaml --var hus
```

- **2D Surface Fields** (e.g., sea surface temperature):

```bash
python src/interp_2d_obs2gcm_cdo.py --yp configs/config_interp.yaml --var tos
```

---

### 2. Perform Bias Correction

- **For 3D Atmospheric Variables**:

```bash
python src/sdmbc_hist_main_3d.py --config configs/config_3d.yaml
```

- **For 2D Surface Variables**:

```bash
python src/sdmbc_hist_main_3d.py --config configs/config_sst.yaml
```

---

### 3. Reformat Outputs

After bias correction, reformat outputs to match the original GCM structure:

- **3D Reformatting:**

```python
from reformat_gcm2origin import reformat_and_save_3d

reformat_and_save_3d(
    bc_path="/path/to/bias_corrected/",
    tlevel=17,
    startyear=1959,
    endyear=1989,
    input_vargcm=["hus", "ta", "ua", "va"],
    origin_vargcm=["hus", "ta", "ua", "va"]
)
```

- **2D Reformatting:**

```python
from reformat_gcm2origin import reformat_and_save_2d

reformat_and_save_2d(
    input_path="/path/to/bias_corrected_sst.nc",
    original_file="/path/to/original_sst.nc",
    output_file="/path/to/reformatted_sst_output.nc",
    remap_weights_file="/path/to/remap_weights_sst.nc"
)
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

If you use SDMBCv2 for your research, please cite:

> Kim, Y. (2023). * A software for correcting systematic biases in RCM input boundary conditions. Environmental Modelling & Software, p.105799. https://doi.org/10.1016/j.envsoft.2023.105799 *.

> Kim, Y. (2023). * Can sub-daily multivariate bias correction of regional climate model boundary conditions improve simulation of the diurnal precipitation cycle? Geophysical Research Letters, 50, e2023GL104442. https://doi.org/10.1029/2023GL104442 *.

> [SDMBCv2 Documentation](https://young-k.notion.site/SDMBCv2-486ef6f8c22e4c33ab696a2eaec06b9c?pvs=4)

---

