import os
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import torch

import sys
sys.path.append("src")
print(os.getcwd())
from utils import get_config, make_dir

from ocda_fas.utils.data_utils import get_domain_list,domain_combined_data_loaders
from source.models.dg_domain_factor_net import DgDomainFactorNet

import pdb


def extract_domain_factor_features(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    config_fname='src/configs/train.yaml'
    config= get_config(config_fname)
    config['device']=device
    config['ocda_debug']=args.debug
    config['net_type']=args.net_type
    config['out_path']=os.path.join(args.experiment_path,args.domainfactor_exp_path)
    config['domain_checkpoint']=os.path.join(args.experiment_path,args.domainfactor_exp_path,args.checkpoint_file)

    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)
    
    source_domain_list,target_domain_list= get_domain_list(config,'domain_factor_net')

    src_data_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='train',net='domain_factor_net',type='src',shuffle=False,drop_last=False)

    tgt_train_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='train',net='domain_factor_net',type='tgt',shuffle=False,drop_last=False)

    net = DgDomainFactorNet(config,num_cls=2,use_init_weight=False)
    net.to(device)

    # load weights
    net.load(config['domain_checkpoint'])
    net.eval()

    src_centroid = domain_centroid(config,src_data_loader, net,device)
    sort_idx = scheduler_idx(config,tgt_train_loader,net,device,src_centroid)
    np.save(os.path.join(config['out_path'], 'sortidx_for_schedule'),sort_idx)


def domain_centroid(config,loader,net,device):
    ''' mean Domain feature factor of source data
    '''

    domain_dtr_sum=torch.zeros(2048)
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, _,_) in enumerate(tqdm(loader)):

            data=data.to(device)

            data.require_grad = False

            
            domain_factor_ftr= net.domain_factor_net.encoder(data.clone()) # Bx512

            norm_vec=torch.linalg.norm(domain_factor_ftr,dim=1).unsqueeze(dim=1)
            domain_factor_ftr_normlaized=domain_factor_ftr/norm_vec

            domain_dtr_sum+=torch.sum(domain_factor_ftr_normlaized,dim=0).cpu()

            if config['ocda_debug']:
                break
    src_centroid=domain_dtr_sum/len(loader.dataset)
    return src_centroid

def scheduler_idx(config,loader,net,device,centroid):
    ''' Returns: idx sorted based on distance from source centroid
    '''

    dist_list=[]
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, _,_) in enumerate(tqdm(loader)):

            data=data.to(device)

            data.require_grad = False

            
            domain_factor_ftr= net.domain_factor_net.encoder(data.clone()) # Bx512
            norm_vec=torch.linalg.norm(domain_factor_ftr,dim=1).unsqueeze(dim=1)
            domain_factor_ftr_normlaized=domain_factor_ftr/norm_vec

            dist=1.0-torch.mm(domain_factor_ftr_normlaized.cpu(),centroid.unsqueeze(dim=1).cpu())
            dist_list+=dist.squeeze(dim=1).tolist()

            if config['ocda_debug'] and batch_idx>1:
                break
    dist_list=np.array(dist_list)
    sort_idx=np.argsort(dist_list)
    return sort_idx

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--net_type', type=str, default='lstmmot')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--experiment_path', type=str, default='output/fas_project/DG_exp/lstmmot_exp_013')
    parser.add_argument('--domainfactor_exp_path', type=str, default='ocda_fas_files/domainfactor/domainfactor_net_exp_002/')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoints/DomainFactorNet_MsCaOu_Ce_7.pt')
    # parser.add_argument('--out_path', type=str, default='ocda_fas_files')

    args = parser.parse_args()
    extract_domain_factor_features(args)
    print('end')

   