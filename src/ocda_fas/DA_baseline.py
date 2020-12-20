import os
import yaml
from os.path import join
from argparse import ArgumentParser
import torch
import torch.optim as optim
import numpy as np
import sys
from torch.utils.tensorboard import SummaryWriter
sys.path.append("src")

from utils import get_config, make_dir
from data_utils import make_exp_dir
from ocda_fas.data_utils import get_domain_list,domain_combined_data_loaders
from source.algorithms.baseline import DABaselineSrc,DABaselineTgt


def baseline_src_train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    config_fname='src/configs/train.yaml'

    config= get_config(config_fname)
    config['device']=device
    config['debug']=args.debug
    config['net_type']=args.net_type
    config['da_baseline_path']=join(args.experiment_path,args.da_baseline)
    config['da_baseline_src_path']=join(config['da_baseline_path'],"src_baseline")
    os.makedirs(config['da_baseline_src_path'], exist_ok=True)
    config['da_src_exp_path']=make_exp_dir(config['da_baseline_src_path'],"src_net")
    config['da_baseline_tnsorboard']=join(config['da_src_exp_path'],"tensorbrd_files")

    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)

    os.makedirs(config['da_src_exp_path'], exist_ok=True)
    os.makedirs(config['da_baseline_tnsorboard'], exist_ok=True)

    #******* Dataloader *************
    source_domain_list,target_domain_list= get_domain_list(config,'da_baseline')

    src_train_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='train',net='da_baseline',type='src')
    src_val_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='val',net='da_baseline',type='src')
    src_test_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='test',net='da_baseline',type='src')

    #******* src class *************
    writer = SummaryWriter(config['da_baseline_tnsorboard'])
    src_baseline=DABaselineSrc(config,configdl,writer)

    if config['debug']:
        num_epoch=1
    else:
        num_epoch= config['da_baseline']['src']['epochs']

    #******* src train loop *************
    best_hter_val=1.0
    print("***** Src Training Start ******")
    for epoch in range(num_epoch):
        print(epoch)
        best_hter_val=src_baseline.train_epoch(src_train_loader,src_val_loader,src_test_loader,epoch,num_epoch,best_hter_val)
    print("*** End ***")

    config_write_loc=join(config['da_src_exp_path'],'config.yaml')
    with open(config_write_loc, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    writer.close()

def baseline_tgt_train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    config_fname='src/configs/train.yaml'
    config= get_config(config_fname)
    config['device']=device
    config['debug']=args.debug
    config['net_type']=args.net_type
    config['da_baseline_path']=join(args.experiment_path,args.da_baseline)
    config['da_baseline_tgt_path']=join(config['da_baseline_path'],"tgt_baseline")
    os.makedirs(config['da_baseline_tgt_path'], exist_ok=True)
    config['da_tgt_exp_path']=make_exp_dir(config['da_baseline_tgt_path'],"tgt_net")
    config['da_baseline_tnsorboard']=join(config['da_tgt_exp_path'],"tensorbrd_files")
    config['da_src_checkpoint']= join(config['da_baseline_path'],args.da_src_baseline_checkpoint)

    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)

    os.makedirs(config['da_tgt_exp_path'], exist_ok=True)
    os.makedirs(config['da_baseline_tnsorboard'], exist_ok=True)

    #******* Dataloader *************
    source_domain_list,target_domain_list= get_domain_list(config,'da_baseline')

    src_train_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='train',net='da_baseline',type='src')
    tgt_train_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='train',net='da_baseline',type='tgt')
    tgt_val_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='val',net='da_baseline',type='tgt')

    if (len(src_train_loader.dataset)/ len(tgt_train_loader.dataset)< 0.7):
        comb_dataset = torch.utils.data.ConcatDataset([src_train_loader.dataset,src_train_loader.dataset])
        src_train_loader = torch.utils.data.DataLoader(comb_dataset, batch_size=config['da_baseline']['batch_size_src'], shuffle=True,drop_last=True,num_workers=config['num_workers'])

    #******* src class *************
    writer = SummaryWriter(config['da_baseline_tnsorboard'])
    src_baseline=DABaselineTgt(config,configdl,writer)

    if config['debug']:
        num_epoch=1
    else:
        num_epoch= config['da_baseline']['tgt']['epochs']

    #******* src train loop *************
    best_loss=0.0
    print("***** Src Training Start ******")
    for epoch in range(num_epoch):
        print(epoch)
        best_loss=src_baseline.train_epoch(src_train_loader,tgt_train_loader,tgt_val_loader,epoch,num_epoch,best_loss)
    print("*** End ***")

    config_write_loc=join(config['da_tgt_exp_path'],'config.yaml')
    with open(config_write_loc, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    writer.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--net_type', type=str, default='lstmmot')
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--experiment_path', type=str, default='output/fas_project/DG_exp/lstmmot_exp_013')
    parser.add_argument('--da_baseline', type=str, default='da_baseline/')
    parser.add_argument('--da_src_baseline_checkpoint', type=str, default='src_baseline/src_net_exp_000/src_baseline_checkpoint_best.pt')

    args = parser.parse_args()
    # baseline_src_train(args)
    baseline_tgt_train(args)





