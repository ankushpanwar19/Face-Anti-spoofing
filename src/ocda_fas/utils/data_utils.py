import os
from os.path import join
import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import Sampler

import sys
sys.path.append("src")
print(os.getcwd())
from utils_dg import get_data_loader as get_dataloader_train,get_part_labels
from data_loader_anet import get_dataset
from run_eval_msu import get_eval_part_labels as get_eval_part_labels_msu
from run_eval_replay_casia import get_eval_part_labels as get_eval_part_labels_re_ca
from run_eval_oulu_npu import get_eval_part_labels as get_eval_part_labels_oulu



def get_domain_list(config,net='mann_net'):
    dset_name_list = ['Ou', 'Ca', 'Ra', 'Ms','Ce']
    full_dset_name = ['oulu-npu', 'casia', 'replay-attack', 'msu','celebA']
    config['eval_on_oulu_devset'] = False 

    source_domain_list=[]
    for i in range(len(dset_name_list)):
        if dset_name_list[i] in config[net]['src_dataset']:
            source_domain_list.append(full_dset_name[i])

            print()

    target_domain_list=[]
    for i in range(len(dset_name_list)):
        if dset_name_list[i] in config[net]['tgt_dataset']:
            target_domain_list.append(full_dset_name[i])
    
    return source_domain_list,target_domain_list

def domain_data_loaders(config,configdl,domain_list,mode='train'):
    
    # source data loaders
    data_loaders = []
    epoch_sizes = []
    iters = []
    mode='train'
    for domain in domain_list:  
        dt_loader, epoch_size, _ = get_dataloader_train(config, configdl,domain, mode,small_trainset=False, drop_last=True)
        data_loaders += [dt_loader]
        epoch_sizes += [epoch_size]
        iters += [iter(dt_loader)]

    return data_loaders,epoch_sizes,iters

def domain_combined_data_loaders(config,configdl,domain_list,mode='train',net='mann_net',type='src',shuffle=True,drop_last=True,data_filter=''):
    
    if type=='src':
        if mode=='train':
            batch_size=config[net]['batch_size_src']
        else:
            batch_size=config[net]['batch_size_src_test'] 
    else:
        if mode=='train':
            batch_size=config[net]['batch_size_tgt']
        else:
            batch_size=config[net]['batch_size_tgt_test']

    print("batch_size",batch_size)
    
    comb_dataset=get_multi_domain_dataset(config,configdl,domain_list,mode,drop_last,data_filter)
    print("Workers:",config['num_workers'])
    combined_data_loader=torch.utils.data.DataLoader(comb_dataset, batch_size=batch_size, shuffle=shuffle,drop_last=drop_last,num_workers=batch_size,pin_memory=True)
    # print("next reach")

    return combined_data_loader 

def get_domain_dataset(config,configdl,dataset,mode='train',small_trainset=False,drop_last=True,data_filter=''):
    '''
    get torch dataset object for given data
    '''
    machine = config['machine']
    num_workers = config['num_workers']
    inp_dim = (config['inp_dim'], config['inp_dim'])

    mode_datafilter=mode+data_filter
    part_all, labels_all, num_exmps = get_part_labels(config, configdl, mode_datafilter, small_trainset = small_trainset, drop_last = drop_last, dataset_name=dataset)

    params = {'app_feats': config['app_feats'],
              'shuffle': True,
              'num_workers': num_workers,
              'res': inp_dim,
              'dataset_name': dataset,
              'net_type': config['net_type']
              }

    dataset= get_dataset(machine, configdl, part_all[mode], labels_all, mode, drop_last, **params)

    return dataset

def get_multi_domain_dataset(config,configdl,domain_list,mode='train',drop_last=True,data_filter=''):
    '''
    get multi dataset object for given data
    '''

    domain_list.sort()
    dataset_list=[]
    for domain in domain_list:

        if domain != 'celebA':
            if mode=='train':
                dataset=get_domain_dataset(config,configdl,domain,mode=mode,small_trainset=False,drop_last=drop_last,data_filter=data_filter)
            else:
                dataset=get_domain_eval_dataset(config,configdl,domain,mode=mode,small_trainset=False,drop_last=drop_last)
        else:
            dataset=get_domain_dataset(config,configdl,domain,mode=mode,small_trainset=False,drop_last=drop_last,data_filter=data_filter)


        dataset_list.append(dataset)

    if len(dataset_list)>1:
        comb_dataset = torch.utils.data.ConcatDataset(dataset_list)
    else:
        comb_dataset=dataset_list[0]

    return comb_dataset


def get_domain_eval_dataset(config,configdl,dataset,mode='train',small_trainset=False,drop_last=True):

    machine = config['machine']
    num_workers = config['num_workers']
    inp_dim = (config['inp_dim'], config['inp_dim'])

    if dataset == 'casia' or dataset == 'replay-attack':
        part_all, labels_all, num_exmps = get_eval_part_labels_re_ca(config, configdl, mode, dataset_name=dataset, drop_last = drop_last)
    elif dataset=='msu':
        part_all, labels_all, num_exmps = get_eval_part_labels_msu(config, configdl, mode, dataset_name=dataset, drop_last = drop_last)
    elif dataset=='oulu-npu':
        part_all, labels_all, num_exmps = get_eval_part_labels_oulu(config, configdl, mode, dataset_name=dataset, drop_last = drop_last)



    params = {'app_feats': config['app_feats'],
              'shuffle': True,
              'num_workers': num_workers,
              'res': inp_dim,
              'dataset_name': dataset,
              'net_type': config['net_type']
              }

    dataset= get_dataset(machine, configdl, part_all[mode], labels_all, mode, drop_last, **params)

    return dataset



class DomainScheduledSampler(Sampler):
    '''
    Sampling according to the score
    '''
    def __init__(self, dataset, sort_idx, ratio, init_ratio, strategy, seed=0):
        np.random.seed(seed)
        if strategy == 'expand':
            start = 0
        elif strategy == 'shift':
            start = int(len(dataset) * (ratio - init_ratio))
        else:
            raise Exception("No such schedule strategy: {}".format(strategy))
        end = int(len(dataset) * ratio)
        self.selected = sort_idx[start:end]
        np.random.shuffle(self.selected)
    
    def __iter__(self):
        return iter(self.selected)

    def __len__(self):
        return len(self.selected)


def make_exp_dir(sub_base_path,net_type):
    if not os.path.isdir(sub_base_path):
        os.mkdir(sub_base_path)
    exp_list = [f for f in os.listdir(sub_base_path) if 'exp' in f]
    if not exp_list:
        exp_path=join(sub_base_path, '{}_exp_000'.format(net_type))
    else:
        exp_path=join(sub_base_path, '{:s}_exp_{:03d}'.format(net_type,len(exp_list)))
    os.mkdir(exp_path)
    return exp_path


class MyDataParallel(nn.DataParallel):

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
