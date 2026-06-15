#!/bin/bash

set -o xtrace

splits=("test" "val" "train")
# splits=("test")
wandb_project=spaceml-3dclouds-issue-1

inferences_dir=/mnt/e/Data/3DClouds/models/finetune_outputs/NeurIPS2025/runs/unet-baseline-radar-reflectivity-goes-kpyrz9qb

num_workers=10
criterion_best_model=best
cyclone_prediction=True

wandb_runids=("kpyrz9qb") # "s9hs4usz" "2l7ldorg") # mae-stack, satmae-msi, mae-stack-stack
finetuned_dir=/mnt/e/Data/3DClouds/models/finetune_outputs/NeurIPS2025/checkpoints/unet-baseline-radar-reflectivity-goes-kpyrz9qb
# finetuned_dir=/home/annajungbluth/models

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

# wandb_runids=("hg4k8ksj") # unet-stack
# finetuned_dir=/data/2025-esl-3dclouds-extremes-models/2025-unet

# # Cyclone dataset
# data_path=/data/2025-esl-3dclouds-extremes-datasets/fine-tuning-reprocessed/cyclones

# for wandb_runid in "${wandb_runids[@]}"; do
#     for split in "${splits[@]}"; do
#         python src/make_inference.py \
#             --wandb_runid $wandb_runid \
#             --wandb_project $wandb_project \
#             --finetuned_dir $finetuned_dir \
#             --inferences_dir $inferences_dir \
#             --data_path $data_path \
#             --split $split \
#             --num_workers $num_workers \
#             --criterion_best_model $criterion_best_model \
#             --cyclone_prediction $cyclone_prediction
#     done
# done

# # Clouds dataset
# data_path=/mnt/disks/data/2025-esl-3dclouds-extremes-datasets/fine-tuning-reprocessed

# for wandb_runid in "${wandb_runids[@]}"; do
#     for split in "${splits[@]}"; do
#         python src/make_inference.py \
#             --wandb_runid $wandb_runid \
#             --wandb_project $wandb_project \
#             --finetuned_dir $finetuned_dir \
#             --inferences_dir $inferences_dir \
#             --data_path $data_path \
#             --split $split \
#             --num_workers $num_workers \
#             --criterion_best_model $criterion_best_model
#     done
# done

