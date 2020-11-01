#!/bin/bash
source /itet-stor/apanwar/net_scratch/conda/etc/profile.d/conda.sh
conda activate .fas
python -u src/train_dg_final.py --config_path src/configs/train.yaml --out_path output --project_path fas_project