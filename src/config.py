"""File where constant variables are kept.

# TODO:
  - Will: This file is getting a little too large to manage easily. I'd like to
    shift the config paths etc. to a set of yaml files, and then load these here
    using a set of Enum objects (e.g. a local path enum, a GCP path enum etc.)
    that can be imported wherever they are needed. I'd like to also shift to
    using Pathlib/Universal pathlib for handling directories and filenames, but
    this would require work throughout the codebase to implement.
  - Make sure local data file structure matches GCP for ease of use.
"""
import os
import pathlib

import autoroot

"""
data file structure:

/2024-ESL-3DClouds
    /src
    ...
/data
    /raw
        /cloudsat
            /2b-geoprof
            /2b-cldclass-lidar
        /modis
        /msg
    /geoprocessed
        /cloudsat
        /modis
        /msg
    /preprocessed
        /cloudsat_aligned_modis
        /cloudsat_aligned_msg
    /ml_ready
        /modis_cloudsat
    ...
"""

# ===== LOCAL DATA PATHS =====
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH = os.path.join(CURRENT_DIR, "..")
DATA_PATH = os.path.join(PROJECT_PATH, "..", "data")

# Raw data
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, "preprocessed")
CLOUDSAT_RAW_DATA_PATH = os.path.join(RAW_DATA_PATH, "cloudsat", "2b-geoprof")
CLOUDCLASS_RAW_DATA_PATH = os.path.join(RAW_DATA_PATH, "cloudsat", "2b-cldclass-lidar")

# Geoprocessed data
GEOPROCESSED_DATA_PATH = os.path.join(DATA_PATH, "geoprocessed")
GEOPROCESSED_CLOUDSAT_DATA_PATH = os.path.join(
    "/mnt", "disks", "msg-patches", "cloudsat-geoprocessed"
)
GEOPROCESSED_MODIS_DATA_PATH = os.path.join(GEOPROCESSED_DATA_PATH, "modis")
GEOPROCESSED_MSG_DATA_PATH = os.path.join(
    "/mnt", "disks", "msg-geoprocessed", "geoprocessed", "msg"
)

# Aligned data
CLOUDSAT_ALIGNED_MODIS_PATH = os.path.join(
    PREPROCESSED_DATA_PATH, "cloudsat_aligned_modis"
)
CLOUDSAT_ALIGNED_MSG_PATH = os.path.join(PREPROCESSED_DATA_PATH, "cloudsat_aligned_msg")

ALIGNED_CLOUDSAT_MSG_CHUNKS_PATH = str(
    pathlib.Path("/mnt/disks/msg-patches/msg-cloudsat-matched")
)
ALIGNED_CLOUDSAT_MODIS_CHUNKS_PATH = str(
    pathlib.Path("/mnt/disks/modis-patches/modis-cloudsat-matched")
)
MSG_CLOUDSAT_OUT_DIR = "/mnt/disks/msg-patches/patches/"
MODIS_CLOUDSAT_OUT_DIR = "/mnt/disks/modis-patches/patches/"

# ML ready/patched data
ML_READY_DATA_PATH = os.path.join(DATA_PATH, "ml_ready")
MODIS_ML_PATCHES_DIRECTORY = os.path.join(ML_READY_DATA_PATH, "modis_cloudsat")
# CLOUDSAT_MODIS_ML_PATCHES_DIRECTORY = os.path.join(ML_READY_DATA_PATH, "modis_cloudsat")
# CLOUDSAT_MSG_ML_PATCHES_DIRECTORY = os.path.join(ML_READY_DATA_PATH, "msg_cloudsat")
CLOUDSAT_MSG_ML_PATCHES_DIRECTORY = "/mnt/disks/msg-patches/patches/"
CLOUDSAT_MODIS_ML_PATCHES_DIRECTORY = "/mnt/disks/modis-patches/patches/"

# Model file paths
MODEL_DIR = os.path.join(PROJECT_PATH, "..", "models")
PRETRAINED_MAE_CHECKPOINT_DIR = os.path.join(MODEL_DIR, "pretrained_mae_models")


# ===== PREPROCESSING PARAMETERS =====

MODIS_PIXEL_SIZE = 1
KM_TO_DEGREES = 1 / 111  # 1 degree lat ~111 km
MODIS_CUTOFF_SIZE_DEGREES = MODIS_PIXEL_SIZE * KM_TO_DEGREES
MSG_PIXEL_SIZE = 3
MSG_CUTOFF_SIZE_DEGREES = MSG_PIXEL_SIZE * KM_TO_DEGREES

MSG_GEOJSON_PATH = os.path.join(PROJECT_PATH, "aux_data", "msg_fov.geojson")
MSG_INVARIANTS_PATH = str(autoroot.root / "aux_data/msg/msg_invariants.nc")

TEMPORAL_BUFFER_GEOSTATIONARY = 10
TEMPORAL_BUFFER_POLAR = 0

CLOUDSAT_GEOSTATIONARY_VARIABLES_FOR_ML = [
    "Height",
    "Radar_Reflectivity",
    "cloud_type_mask",
    "input_indices",
]

# Variables relating to MODIS alignment
MODIS_PATCH_SIZE = 256
MODIS_PATCH_DENSITY: float = 4
MODIS_MAX_OFFSET: int = 96
MODIS_MIN_CLOUD_COVERAGE: float = 0.2
MODIS_MAX_NAN_COVERAGE: float = 0.0

# Variables related to MSG alignment
MSG_PATCH_SIZE = 128 # 256 # Updated variables for new temporal patches
MSG_MAX_OFFSET: int = 64 # MSG_PATCH_SIZE // 4 # Reverted back to previous offset (256//4) 
MSG_PATCH_DENSITY: float = 2 # 4 # Updated density for new temporal patches
MSG_MIN_CLOUD_COVERAGE: float = 0.2
MSG_MAX_NAN_COVERAGE: float = 0.01

# CURRENTLY NOT USED
MIN_NUMBER_CLOUDSAT_PIXELS_MSG = 1000
MSG_PATCHING_BOUNDS = (
    700  # How many pixels removed from the MSG edge the patch is allowed to be
)
MSG_PATCHING_STRATEGY = "random"


# ===== GCP DATA PATHS =====

# GCP folder structure
GCP_BASE_PATH = "gs://2024-esl-3dclouds"
GCP_DATASETS_PATH = f"{GCP_BASE_PATH}-datasets"
GCP_PREPROCESSED_DATA_PATH = f"{GCP_DATASETS_PATH}-preprocessed"
GCP_ML_READY_DATA_PATH = f"{GCP_DATASETS_PATH}-ml-ready"

GCP_GEOPROF_PATH = os.path.join(GCP_DATASETS_PATH, "cloudsat", "2b-geoprof")
GCP_CLDCLASS_PATH = os.path.join(GCP_DATASETS_PATH, "cloudsat", "2b-cldclass-lidar")


GCP_MSG_NATIVE_PATH = os.path.join(GCP_DATASETS_PATH, "msg", "2010", "L1b")

GCP_MODIS_GEOPROCESSED_PATH = "gs://2024-esl-3dclouds-datasets-geoprocessed/aqua/2010"

GCP_CLOUDSAT_ALIGNED_MODIS_PATH = f"{GCP_PREPROCESSED_DATA_PATH}/cloudsat_aligned_modis"
GCP_CLOUDSAT_ALIGNED_MODIS_CWC_RVOD_PATH = (
    f"{GCP_CLOUDSAT_ALIGNED_MODIS_PATH}/2b-cwc-rvod"
)
GCP_CLOUDSAT_ALIGNED_MODIS_GEOPROF_PATH = (
    f"{GCP_CLOUDSAT_ALIGNED_MODIS_PATH}/2b-geoprof"
)

GCP_CLOUDSAT_ALIGNED_MSG_PATH = f"{GCP_PREPROCESSED_DATA_PATH}/cloudsat_aligned_msg"
GCP_CLOUDSAT_ALIGNED_MSG_CWC_RVOD_PATH = f"{GCP_CLOUDSAT_ALIGNED_MSG_PATH}/2b-cwc-rvod"
GCP_CLOUDSAT_ALIGNED_MSG_GEOPROF_PATH = f"{GCP_CLOUDSAT_ALIGNED_MSG_PATH}/2b-geoprof"
GCP_CLOUDSAT_MSG_ML_PATCHES_DIRECTORY = ""

# Different wavelengths
MSG_WAVELENGTHS = {
    "IR_016": 1.64,
    "IR_039": 3.92,
    "IR_087": 8.70,
    "IR_097": 9.66,
    "IR_108": 10.80,
    "IR_120": 12.00,
    "IR_134": 13.40,
    "VIS006": 0.64,
    "VIS008": 0.81,
    "WV_062": 6.25,
    "WV_073": 7.35,
}
