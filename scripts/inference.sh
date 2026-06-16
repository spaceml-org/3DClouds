#!/bin/bash

set -o xtrace

splits=("test" "val" "train")  # TODO: Fill in
wandb_project=inference-2026  # TODO: Fill in

inferences_dir=/home/user/inferences  # TODO: Fill in

num_workers=6  # TODO: Fill in
criterion_best_model=best
cyclone_prediction=True  # TODO: Fill in

wandb_runids=("h3158clx")  # TODO: Fill in
finetuned_dir=/data/2025-esl-3dclouds-extremes-models/2025-swinmae-finetuning  # TODO: Fill in

# Cyclone dataset
data_path=/data/2025-esl-3dclouds-extremes-datasets/fine-tuning-reprocessed/cyclones  # TODO: Fill in

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
data_path=/mnt/disks/data/2025-esl-3dclouds-extremes-datasets/fine-tuning-reprocessed  # TODO: Fill in

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
