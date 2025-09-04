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
  - [2. Compile mrmbc with f2py](#2-compile-mrmbc-with-f2py)
  - [3. Perform Bias Correction](#3-perform-bias-correction)
  - [4. Reformat Outputs](#4-reformat-outputs)
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
python src/interp_obs2gcm.py --yp path/to/config/config.yaml --var hus --ncpus N --mem M
```

- **2D Surface Fields** (e.g., sea surface temperature):

```bash
python src/interp_2d_obs2gcm_cdo.py --yp path/to/config/config.yaml --var tos --ncpus N --mem M
```

---
### 2. Compile mrmbc with f2py

Compile the Fortran subroutines into a Python module (requires gfortran). Run these in the directory containing mbc_subroutines.f90:

```bash
python -m numpy.f2py -m mrmbc -h mbc_subroutines.pyf mbc_subroutines.f90 --overwrite-signature
python -m numpy.f2py -c mbc_subroutines.pyf mbc_subroutines.f90 --fcompiler=gnu95 --f90flags="-O3 -frecursive"
```

This produces a mrmbc module that can be imported from Python.

### 3. Perform Bias Correction

- **For 3D Atmospheric Variables**:

```bash
python src/sdmbc_main_3d.py --yp /path/to/config/config.yaml --ncpus N --mem M
```

- **For 2D Surface Variables**:

```bash
python src/sdmbc_main_surface.py --config /path/to/config/config.yaml --ncpus N --mem M
```

---

### 4. Reformat Outputs

After bias correction, reformat outputs to match the original GCM structure:

- **3D Reformatting:**

```python
from reformat_gcm2origin import reformat_and_save_3d

reformat_and_save_3d(
    bc_path="/path/to/bias_corrected/",
    tlevel={level},
    startyear={year},
    endyear={year},
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

> [SDMBCv2 Documentation](https://young-k.notion.site/SDMBCv2-48a0670426514461a778848322178882?pvs=4)

---

