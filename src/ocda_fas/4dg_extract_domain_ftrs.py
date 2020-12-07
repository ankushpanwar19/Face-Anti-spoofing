import os
import numpy as np
from argparse import ArgumentParser
import torch

import sys
sys.path.append("src")
print(os.getcwd())
from utils import get_config, make_dir

from data_utils import get_domain_list,domain_combined_data_loaders
from source.models.dg_domain_factor_net import DgDomainFactorNet

import pdb


def extract_domain_factor_features(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    config_fname='src/configs/train.yaml'
    config= get_config(config_fname)
    config['device']=device
    config['net_type']=args.net_type
    config['out_path']=os.path.join(args.experiment_path,args.out_path)
    config['domain_checkpoint']=os.path.join(args.experiment_path,args.domain_checkpoint_file)

    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)
    
    source_domain_list,target_domain_list= get_domain_list(config,'domain_factor_net')

    src_data_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='train',net='domain_factor_net',type='src',shuffle=False,drop_last=False)

    tgt_train_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='train',net='domain_factor_net',type='tgt',shuffle=False,drop_last=False)

    net = DgDomainFactorNet(config,2,False)
    net.to(device)

    # load weights
    net.load(config['domain_checkpoint'])
    net.eval()

    src_ftrs = extract_dataset(src_data_loader, net,device)
    src_ftrs.tofile(os.path.join(config['out_path'], 'src_domain_factor_ftr.bin')) # N x 2048
    tgt_ftrs = extract_dataset(tgt_train_loader, net,device)
    tgt_ftrs.tofile(os.path.join(config['out_path'], 'tgt_domain_factor_ftr.bin'))


def extract_dataset(loader,net,device):
    ''' mean Domain feature factor of source data
    '''

    domain_ftrs=[]
    net.eval()
    for batch_idx, (data, _,_) in enumerate(loader):

        info_str = "[Extract] [{}/{} ({:.2f}%)]".format(
            batch_idx * len(data), len(loader.dataset), 100 * float(batch_idx) / len(loader))

        data=data.to(device)

        data.require_grad = False

        with torch.no_grad():
            domain_factor_ftr,_,_ = net.domain_factor_net(data.clone()) # Bx512

        domain_ftrs.append(domain_factor_ftr.detach().cpu().numpy())

        if batch_idx % 100 == 0:
            print(info_str)
        if batch_idx>2:
            break

    domain_ftrs_arr = np.concatenate(domain_ftrs, axis=0)
    # assert len(loader.dataset) == src_domain_ftrs.shape[0], "{} vs {}".format(len(loader.dataset), src_domain_ftrs.shape[0])
    # np.save(config['filepath'],domain_ftrs_arr)
    return domain_ftrs_arr

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--net_type', type=str, default='lstmmot')
    parser.add_argument('--experiment_path', type=str, default='output/fas_project/DG_exp/lstmmot_exp_013')
    parser.add_argument('--domain_checkpoint_file', type=str, default='ocda_fas_files/DomainFactorNet_MsCaOu_Ce.pt')
    parser.add_argument('--out_path', type=str, default='ocda_fas_files')

    args = parser.parse_args()
    extract_domain_factor_features(args)
    print('end')

   