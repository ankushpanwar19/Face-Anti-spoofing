import os

import numpy as np
import yaml
import torch
from os.path import join
from argparse import ArgumentParser

import sys
sys.path.append("src")
print(os.getcwd())
from utils import get_config, make_dir


from centroids import compute_source_centroids


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
    config_fname='src/configs/train.yaml'
    config= get_config(config_fname)
    config['net_type']=args.net_type
    config['checkpoint_file_path']=os.path.join(args.experiment_path,args.checkpoint_file)
    config['centroids_path']=os.path.join(args.experiment_path,args.centroids_path)
    config['device']=device

    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)


    if os.path.isfile(config['checkpoint_file_path']):

        if not os.path.isdir(config['centroids_path']):
            make_dir(config['centroids_path'])
        compute_source_centroids(config,configdl)
    else:
        print("Checkpoint files doesn't exist !")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--net_type', type=str, default='lstmmot')
    parser.add_argument('--experiment_path', type=str, default='output/fas_project/DG_exp/lstmmot_exp_013')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoints/net_00039439.pt')
    parser.add_argument('--centroids_path', type=str, default='ocda_fas_files')

    args = parser.parse_args()
    main(args)