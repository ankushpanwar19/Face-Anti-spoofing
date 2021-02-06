#!/bin/bash
source /itet-stor/apanwar/net_scratch/conda/etc/profile.d/conda.sh
conda activate .fas
# python -u src/train_dg_final.py --config_path src/configs/train.yaml --out_path output --project_path fas_project
# python -u src/ocda_fas/0train_src_net.py
# python -u src/ocda_fas/1compute_centroids.py
# python -u src/ocda_fas/2train_dg_mann_net.py
python -u src/ocda_fas/3train_dg_domainfactornet.py
# python -u src/ocda_fas/4dg_extract_domain_ftrs.py
# python -u src/ocda_fas/5train_dg_scheduled_mann.py
# python -u src/ocda_fas/eval.py
# python src/ocda_fas/DA_baseline.py
# python src/ocda_fas/test.py