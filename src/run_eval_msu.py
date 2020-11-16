# from get_examples_labels_siw import get_examples_labels as get_siw_dataset
from get_examples_labels_oulu import get_examples_labels as get_oulu_dataset
# from get_examples_labels_pxlvision import  get_examples_labels as get_pxlvision_dataset
# from data_loader_dnet import get_loader as get_loader_dnet
from data_loader_anet import get_loader as get_loader_anet
import math
from utils import make_dir
from os.path import join
import statistics as stat
import torch
import sys
import os
import time

from get_examples_labels_msu import get_examples_labels as get_msu_dataset
from get_examples_labels_replayattack import get_examples_labels as get_replayattack_dataset
from utils import get_loader_all
from data_loader_anet import get_MOT_loader

def get_dataset_id(dataset):
    datasetID = ''
    if dataset == 'siw':
        datasetID = 'si'
    elif dataset == 'oulu-npu':
        datasetID = 'ou'
    elif dataset == 'replay-attack':
        datasetID = 'ra'
    elif dataset == 'replay-mobile':
        datasetID = 'rm'
    elif dataset == 'casia':
        datasetID = 'ca'
    elif dataset == 'msu':
        datasetID = 'ms'
    elif dataset == 'casia-maged':
        datasetID = 'ca-ma'
    elif dataset == 'replay-attack-maged':
        datasetID = 'ra-ma'
    else:
        pass
    return datasetID

def get_dataloader_test(config, configdl, debug, dataset_name = None,drop_last = False):
    '''
    Data loader for validation and test set for MSU
    ouput: 
    Dataloader videos [tensor,video path, label],
    num_exmps [number of frames], scores ([[],1],[[],-2]), 
    gtFlags [''live/real_client005_android_SD_scene01'':1,...], batch_size, num_test_batches, img_path = '/scratch_net/knuffi_third/susaha/apps/datasets/msu-mfsd/rgb_images_full/train'

    '''
    print(' --- run_eval.py --> get_test_dataloader() --> getting the test dataloader ---')


    # dataset_name = config['test_dataset']
    dataset_name = 'msu'
    machine = config['machine']
    num_workers = config['num_workers']
    inp_dim = (config['inp_dim'], config['inp_dim'])
    depth_map_size = (config['depth_map_size'], config['depth_map_size'])
    app_feats = config['app_feats']

    if config['net_type'] == 'lstmmot':
        batch_size = config['batch_size_lstm']
    else:
        batch_size = config['batch_size']
    protocol = config['test_dataset_conf'][dataset_name]['protocol']
    split = config['test_dataset_conf'][dataset_name]['split']
    sel_every = config['test_dataset_conf'][dataset_name]['sel_every']
    sel_these_many = config['test_dataset_conf'][dataset_name]['sel_these_many']
    dataset_path = config['test_dataset_conf'][dataset_name]['dataset_path_machine{}'.format(machine)]
    eval_type = config['eval_type']
    const = config['const_testset']

    config['batch_size'] = batch_size
    # config['test_dataset_conf'][test_dataset]['sel_these_many'] = sel_these_many

    if 'Val' in eval_type:
        mode = 'val'
    elif 'Test' in eval_type:
        mode = 'test'

    if mode == 'val':
        img_path = config['test_dataset_conf'][dataset_name]['full_img_path_machine{}'.format(machine)]
    if mode == 'test':
        img_path = config['test_dataset_conf'][dataset_name]['full_img_path_test_machine{}'.format(machine)]
    else:
        pass

    num_cls = configdl['num_cls']
    print(dataset_name)
    datasetID = get_dataset_id(dataset_name)

    net_type = config['net_type']

    print(' --- datasetID: {}'.format(datasetID))
    print(' --- sampling {} examples,labels for dataset {} --- '.format('test', dataset_name))
    print(' --- img read path {} ---'.format(img_path))

    part, labels, gtFlags, scores, num_exmps = \
        get_msu_dataset(dataset_path, mode, protocol, split, sel_every=sel_every,sel_these_many = sel_these_many, img_path = img_path, net_type = net_type, datasetID=datasetID, num_cls=num_cls)

    if config['net_type'] == 'lstmmot':
        batch_size = config['batch_size_lstm']

    num_test_batches = math.floor(num_exmps / batch_size)
    last_batch_size = num_exmps % batch_size
    if last_batch_size > 0:
        num_test_batches += 1
    print('mini-batch size [{}]; num_test_batches [{}]'.format(batch_size, num_test_batches))
    params = {'app_feats': app_feats,
              'batch_size': batch_size,
              'shuffle': True,
              'num_workers': num_workers,
              'res': inp_dim,
              'dataset_name': dataset_name,
              'depth_map_size': depth_map_size,
              'net_type': net_type, 
              'const_testset': const
              }
    if config['net_type'] == 'lstmmot' or 'cvpr2018dg' in config['net_type']:
        print('>>> Eval MSU lstmmot <<< ')
        my_dataloader = get_MOT_loader(machine, configdl, part[mode], labels, mode, drop_last, **params)
    else: 
        print('>>> Eval MSU regular <<< ')
        my_dataloader = get_loader_all(machine, configdl, part[mode], labels, mode, drop_last, **params)

    return my_dataloader, num_exmps, scores, gtFlags, batch_size, num_test_batches, img_path