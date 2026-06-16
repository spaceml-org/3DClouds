#!/bin/bash

set -o xtrace

splits=("test" "val" "train")
wandb_project=spaceml-3dclouds-issue-1

inferences_dir=/mnt/e/Data/3DClouds/models/finetune_outputs/NeurIPS2025/runs/unet-baseline-radar-reflectivity-msg-svgiqytm

num_workers=10
criterion_best_model=best
cyclone_prediction=True

wandb_runids=("svgiqytm")
finetuned_dir=/mnt/e/Data/3DClouds/models/finetune_outputs/NeurIPS2025/checkpoints/unet-baseline-radar-reflectivity-msg-svgiqytm

# Cyclone dataset
data_path=/mnt/e/Data/3DClouds/fine-tuning-reprocessed/cyclones

for wandb_runid in "${wandb_runids[@]}"; do
    for split in "${splits[@]}"; do
        python src/make_inference.py \
            --wandb_runid $wandb_runid \
            --wandb_project $wandb_project \
            --finetuned_dir $finetuned_dir \
            --inferences_dir $inferences_dir \
            --data_path $data_path \
            --split $split \
            --num_workers $num_workers \
            --criterion_best_model $criterion_best_model \
            --cyclone_prediction $cyclone_prediction
    done
done

# Clouds dataset
data_path=/mnt/e/Data/3DClouds/fine-tuning-reprocessed

for wandb_runid in "${wandb_runids[@]}"; do
    for split in "${splits[@]}"; do
        python src/make_inference.py \
            --wandb_runid $wandb_runid \
            --wandb_project $wandb_project \
            --finetuned_dir $finetuned_dir \
            --inferences_dir $inferences_dir \
            --data_path $data_path \
            --split $split \
            --num_workers $num_workers \
            --criterion_best_model $criterion_best_model
    done
done
