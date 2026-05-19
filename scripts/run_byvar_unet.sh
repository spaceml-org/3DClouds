#!/bin/bash

set -o xtrace

config_name=configs/baselines/unet-byvar

variables=("IWC" "re" "Radar_Reflectivity")

for variable in "${variables[@]}"; do
    CUDA_VISIBLE_DEVICES=5,6 python train.py \
        --config-path $config_name \
        experiment_name=unet-baseline-$variable-byvar \
        ++dataloader.transforms_dict.msg_cloudsat.cloudsat_variables="[$variable]" \
        ++dataloader.transforms_dict.goes_cloudsat.cloudsat_variables="[$variable]" \
        ++dataloader.transforms_dict.himawari_cloudsat.cloudsat_variables="[$variable]" \
        ++dataloader.cloudsat_variables="[$variable]" \
        ++model.variable="$variable" \
        ++max_epochs=50 \

done

