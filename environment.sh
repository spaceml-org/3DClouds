#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Creating conda environment '3DClouds' with Python 3.11..."
conda create -n 3DClouds python=3.11 -y

# Activate the environment
# Note: In a shell script, 'conda activate' requires initializing conda first
# Using the source path is the most reliable way inside a script
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate 3DClouds

echo "⚙️ Configuring Conda channels..."
conda config --env --add channels conda-forge
conda config --env --add channels bioconda
conda config --env --add channels nvidia
conda config --env --set channel_priority flexible

echo "📦 Installing data science, geospatial, and system dependencies via Conda..."
# Grouping these allows Conda to resolve all dependencies (like OpenVINO) safely upfront
conda install -y \
    scipy satpy opencv scikit-learn scikit-image \
    netcdf4 hdf4 pyhdf hdf5 h5py h5netcdf \
    boto3 goes2go s3fs earthaccess eumdac google-cloud-storage gcsfs \
    geopandas rasterio rioxarray \
    tqdm wandb mlflow albumentations \
    matplotlib seaborn geoviews hvplot datashader progressbar2 \
    loguru python-dotenv timm hydra-core hydra-colorlog \
    sphinx sphinx_rtd_theme black einops

echo "🧬 Installing Snakemake from Bioconda..."
conda install -y snakemake -c bioconda

echo "🐍 Installing Python tools and utilities via Pip..."
pip install lightning torcheval autoroot jupyterlab ipywidgets jupyter monai jupyter_bokeh

echo "🔥 The Grand Finale: Installing GPU-enabled PyTorch (CUDA 12.6)..."
# Running this LAST guarantees that Pip overwrites any dummy CPU versions
# pulled in by the geospatial/data-science packages above.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "✅ Environment setup complete!"
echo "🔍 Verification:"
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"