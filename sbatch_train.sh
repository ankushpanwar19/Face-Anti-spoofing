#!/bin/bash
source /itet-stor/apanwar/net_scratch/conda/etc/profile.d/conda.sh
conda activate .fas
# python -u src/train_dg_final.py --config_path src/configs/train.yaml --out_path output --project_path fas_project
# python -u src/ocda_fas/1compute_centroids.py
# python -u src/ocda_fas/2train_dg_mann_net.py
python -u src/ocda_fas/3train_dg_domainfactornet.py