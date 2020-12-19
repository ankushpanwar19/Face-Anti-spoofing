import os
from os.path import join
from argparse import ArgumentParser
import torch
import torch.optim as optim
import numpy as np
import sys
sys.path.append("src")

from utils import get_config, make_dir
from ocda_fas.data_utils import get_domain_list,domain_combined_data_loaders
from source.algorithms.baseline import DABaselineSrc


def baseline_train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    config_fname='src/configs/train.yaml'
    config= get_config(config_fname)
    config['device']=device
    config['debug']=args.debug
    config['net_type']=args.net_type
    config['da_baseline_path']=join(args.experiment_path,args.da_baseline)
    print(config_fname)
    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)
    print(configdl_fname)
    os.makedirs(config['da_baseline_path'], exist_ok=True)

    #******* Dataloader *************
    source_domain_list,target_domain_list= get_domain_list(config,'da_baseline')

    src_train_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='train',net='da_baseline',type='src')
    src_val_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='val',net='da_baseline',type='src')
    src_test_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='test',net='da_baseline',type='src')

    #******* src class *************
    src_baseline=DABaselineSrc(config,configdl)

    if config['debug']:
        num_epoch=1
    else:
        num_epoch= config['da_baseline']['epochs']

    #******* src train loop *************
    best_hter_val=1.0
    print("***** Src Training Start ******")
    for epoch in range(num_epoch):
        print(epoch)
        src_baseline.train_epoch(src_train_loader,epoch,num_epoch)
        hter_val,eer_thr=src_baseline.val(src_val_loader,type='val')
        hter_test,_=src_baseline.val(src_test_loader,type='test',eer_thr=eer_thr)
        
        src_baseline.save(join(config['da_baseline_path']),"src_baseline_checkpoint_last.pt")
        print("*****Checkpoint saved*****")
        if hter_val<= best_hter_val:
            src_baseline.save(join(config['da_baseline_path']),"src_baseline_checkpoint_best.pt")
            best_hter_val=hter_val
    print("*** End ***")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--net_type', type=str, default='lstmmot')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--experiment_path', type=str, default='output/fas_project/DG_exp/lstmmot_exp_013')
    parser.add_argument('--da_baseline', type=str, default='da_baseline/')

    args = parser.parse_args()
    baseline_train(args)





