# Global Reconstructions of 3D Clouds & Climate Extremes

Use geostationary satellite observations and deep learning to reconstruct 3D microphysical properties of clouds and tropical cyclones.

## Table of Contents
1. [Installation](#1-installation)
   1. [Conda environment](#11-conda-environment)
   2. [Miscellaneous tools (optional)](#12-miscellaneous-tools-optional)
2. [Usage](#2-usage)
   1. [Download](#21-download)
      1. [Datasets](#211-datasets)
      2. [Trained models](#212-trained-models)
   2. [Pre-training](#22-pre-training)
      1. [Training](#221-training)
      2. [Inference](#222-inference)
      3. [Evaluation](#223-evaluation)
   3. [Fine-tuning](#23-fine-tuning)
      1. [Training](#231-training)
      2. [Inference](#232-inference)
      3. [Evaluation](#233-evaluation)
3. [References](#3-references)
   1. [Publications](#31-publications)
   2. [Acknowledgements](#32-acknowledgements)

## 1. Installation

Clone the repository:

```bash
git clone https://github.com/spaceml-org/3DClouds.git
```
3DClouds is built with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [Hydra](https://hydra.cc/docs/intro/). 

### 1.1 Conda environment

Create a new conda environment and install pre-requisites by executing the script [`environment.sh`](environment.sh):
```bash
./environment.sh
conda activate 3DClouds
```
Note: Allow enough time for Conda to resolve the environment, as it may take a while to find compatible versions of all packages.

### 1.2 Miscellaneous tools (optional)

Install [pre-commit](https://pre-commit.com/) to automatically run code formatting and linting before each commit:

```bash
pre-commit install
```

Each time you commit, your code will be formatted, linted, checked for imports, merge conflicts, and more. If any of these checks fail, the commit will be aborted.

## 2. Usage

### 2.1 Download

#### 2.1.1 Datasets

To be completed.

#### 2.1.2 Trained models

To be completed.

### 2.2 Pre-training

#### 2.2.1 Training

To be completed.

#### 2.2.2 Inference

To be completed.

#### 2.2.3 Evaluation

To be completed.

### 2.3 Fine-tuning

#### 2.3.1 Training

To be completed.

#### 2.3.2 Inference

To be completed.

For example, to run the multivariable, multisatellite U-Net model, execute the script:
```bash
./scripts/predict_unet-baseline-multivar-multisat-63it0bw9.sh
```

#### 2.3.3 Evaluation

To be completed.

## 3. References

### 3.1 Publications
- Girtsou et al. (2024): https://arxiv.org/abs/2501.02035.
- Ermis et al. (2025): https://www.climatechange.ai/papers/neurips2025/63

### 3.2 Acknowledgements
This work has been enabled by Frontier Development Lab Earth Systems Lab (https://eslab.ai/)—a
public / private partnership between the European Space Agency (ESA), Trillium Technologies, the
University of Oxford and leaders in commercial AI supported by Google Cloud, Scan Computers,
Nvidia Corporation and Pasteur Labs.