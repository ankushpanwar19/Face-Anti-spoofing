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
import numpy as np

import copy

# --- For siw, oulu dataset
def get_dataloader_test(config, configdl, debug, eval_on_oulu_devset, drop_last = False):
    if debug:
        print(' --- run_eval.py --> get_test_dataloader() --> getting the test dataloader ---')
    dataset_name = config['test_dataset']
    machine = config['machine']
    num_workers = config['num_workers']
    inp_dim = (config['inp_dim'], config['inp_dim'])
    depth_map_size = (config['depth_map_size'], config['depth_map_size'])
    app_feats = config['app_feats']
    batch_size = config['batch_size']
    protocol = config['test_dataset_conf'][dataset_name]['protocol']
    split = config['test_dataset_conf'][dataset_name]['split']
    sel_every = config['test_dataset_conf'][dataset_name]['sel_every']
    sel_these_many = config['test_dataset_conf'][dataset_name]['sel_these_many']
    dataset_path = config['test_dataset_conf'][dataset_name]['dataset_path_machine{}'.format(machine)]
    if eval_on_oulu_devset:
        img_path = config['test_dataset_conf'][dataset_name]['img_path_dev_machine{}'.format(machine)]
    else:
        img_path = config['test_dataset_conf'][dataset_name]['img_path_machine{}'.format(machine)]
    depth_map_path = config['test_dataset_conf'][dataset_name]['depth_map_path_machine{}'.format(machine)]
    net_type = config['net_type']
    pickle_fname = config['pickle_fname']
    print(' --- sampling {} examples,labels for dataset {} --- '.format('test', dataset_name))
    print(' --- dataset read path {} ---'.format(dataset_path))
    num_exmps = None
    part = None
    labels = None
    scores = None
    gtFlags = None
    if dataset_name == 'siw':
        part, labels, gtFlags, scores, num_exmps = get_siw_dataset(dataset_path, 'test', protocol, split, sel_every, sel_these_many, pickle_fname, net_type, datasetID='si')  # gtFlags, scores are for test time
    elif dataset_name == 'oulu-npu':
        part, labels, gtFlags, scores, num_exmps = get_oulu_dataset(dataset_path, 'test', protocol, split, sel_every, sel_these_many, img_path, net_type, eval_on_oulu_devset, datasetID='ou')  # gtFlags, scores are for test time
    # elif dataset_name == 'pxlvision':
    #     part, labels, gtFlags, scores, num_exmps = get_pxlvision_dataset(dataset_path, 'test', protocol, split, sel_every, sel_these_many, img_path, net_type, eval_on_oulu_devset)  # gtFlags, scores are for test time
    # else:
        pass
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
              'net_type': net_type
              }
    if config['net_type'] == 'resnet' or config['net_type'] == 'anet' or 'dadpnet' in config['exp_name'] and config['dadpnet_net_arch'] == 3:
        if eval_on_oulu_devset:
            mode = 'val'
        else:
            mode = 'test'
        my_dataloader = get_loader_anet(machine, configdl, part['test'], labels, mode, False, **params)
    # else:
    #     my_dataloader = get_loader_dnet(part['test'], labels, img_path, 'test', depth_map_path, drop_last, **params)
    return my_dataloader, num_exmps, scores, gtFlags, batch_size, num_test_batches, img_path


def eval(config, trainer, dataset_name, iterations, epoc_cnt, eval_outpath, test_dataloader, num_exmps, scores, gtFlags, debug, min_l2norm_val_previous=None, max_l2norm_val_previous=None, anet_loss='softmax', tsne_feats = None):
    if debug:
        print('Running evaluation ... --> Iteration: {}; Epoch: {}'.format(iterations + 1, epoc_cnt + 1))
    trainer.set2eval()
    print('>>>>>> eval <<<<<')
    # fusion_net_arch = config['fusion_net_arch']
    if config['net_type'] == 'lstmmot':
        batch_size = config['batch_size_lstm']
    else:
        batch_size = config['batch_size']
    
    device = config['device']
    sel_these_many = config['test_dataset_conf'][dataset_name]['sel_these_many']
    testset_protocol = config['test_dataset_conf'][dataset_name]['protocol']
    testset_split = config['test_dataset_conf'][dataset_name]['split']
    # out_path = join(eval_outpath, dataset_name, 'Protocol_{}'.format(testset_protocol), 'Split_{}'.format(testset_split))
    out_path = eval_outpath
    num_test_batches = math.floor(num_exmps / batch_size)
    last_batch_size = num_exmps % batch_size
    if last_batch_size > 0:
        num_test_batches += 1
    save_file_path = join(out_path, dataset_name)
    # save_file_path_tsne = join(out_path, 'sel_these_many_{}'.format(sel_these_many), 'tsne_feats_{:08d}'.format(iterations + 1))
    fusiontype = ['sum_fusion', 'mean_fusion', 'mult_fusion']
    # if config['fusion_type']:
    #     if config['fusion_type'] > -1:
    #         save_file_path = join(save_file_path, fusiontype[int(config['fusion_type'])])
    make_dir(save_file_path)
    # make_dir(save_file_path_tsne)
    evalOutpath = save_file_path
    eval_out_file_dnet_minmax_values = None
    eval_out_file_vl_dnetsc = None
    eval_out_file_vl_anetsc = None
    eval_out_file_vl_fusedsc = None
    eval_out_file_vl_fusedsc2 = None
    fname_dnet_minmax_values = None
    fname_video_level_dnet_scores = None
    fname_video_level_anet_scores = None
    fname_video_level_fused_scores = None
    fname_video_level_fused_scores2 = None

    # --- added
    fname_video_level_lstm_scores = None
    fname_video_level_comb_scores = None
    # --- 

    # if 'dnet' in config['exp_name'] or 'dadpnet' in config['exp_name'] and config['dadpnet_net_arch'] == 0:
    #     fname_dnet_minmax_values = join(save_file_path, 'dnet_minmax_values_{:08d}.txt'.format(iterations + 1))
    #     fname_video_level_dnet_scores = join(save_file_path, 'dnet_scores_vl_{:08d}.txt'.format(iterations + 1))
    #     eval_out_file_dnet_minmax_values = open(fname_dnet_minmax_values, 'w')
    #     eval_out_file_vl_dnetsc = open(fname_video_level_dnet_scores, 'w')
    # elif 'anet' in config['exp_name'] or 'resnet' in config['exp_name'] or \
    #         'dadpnet' in config['exp_name'] and config['dadpnet_net_arch'] == 2 or\
    #         'dadpnet' in config['exp_name'] and config['dadpnet_net_arch'] == 3:
    #     fname_video_level_anet_scores = join(save_file_path, 'anet_scores_vl_{:08d}.txt'.format(iterations + 1))
    #     eval_out_file_vl_anetsc = open(fname_video_level_anet_scores, 'w')
    # elif 'fusion' in config['exp_name'] or 'dadpnet' in config['exp_name'] and config['dadpnet_net_arch'] == 1:
    #     fname_dnet_minmax_values = join(save_file_path, 'dnet_minmax_values_{:08d}.txt'.format(iterations + 1))
    #     fname_video_level_dnet_scores = join(save_file_path, 'dnet_scores_vl_{:08d}.txt'.format(iterations + 1))
    #     eval_out_file_dnet_minmax_values = open(fname_dnet_minmax_values, 'w')
    #     eval_out_file_vl_dnetsc = open(fname_video_level_dnet_scores, 'w')
    #     fname_video_level_anet_scores = join(save_file_path, 'anet_scores_vl_{:08d}.txt'.format(iterations + 1))
    #     eval_out_file_vl_anetsc = open(fname_video_level_anet_scores, 'w')
    #     fname_video_level_fused_scores = join(save_file_path, 'fused_scores_vl_val_{:08d}.txt'.format(iterations + 1))
    #     fname_video_level_fused_scores2 = join(save_file_path, 'fused_scores_vl_test_{:08d}.txt'.format(iterations + 1))
    #     eval_out_file_vl_fusedsc = open(fname_video_level_fused_scores, 'w')
    #     eval_out_file_vl_fusedsc2 = open(fname_video_level_fused_scores2, 'w')
    if 'lstmmot' in config['exp_name'] or 'cvpr2018dg' in config['exp_name']  or 'resnet18' in config['exp_name']:
        batch_size = config['batch_size_lstm']
        fname_video_level_anet_scores = join(save_file_path, 'anet_scores_vl_{:08d}.txt'.format(iterations + 1))
        eval_out_file_vl_anetsc = open(fname_video_level_anet_scores, 'w')
        fname_video_level_lstm_scores = join(save_file_path, 'lstm_scores_vl_{:08d}.txt'.format(iterations + 1))
        eval_out_file_vl_lstm_sc = open(fname_video_level_lstm_scores, 'w')
        fname_video_level_comb_scores = join(save_file_path, 'comb_scores_vl_{:08d}.txt'.format(iterations + 1))
        eval_out_file_vl_comb_sc = open(fname_video_level_comb_scores, 'w')

    bsize = batch_size
    print('>>>> Batch Size: {} <<<<'.format(bsize))
    dnet_mean_sc_test = None
    fused_sc_test = None
    print('{}  {}'.format(config['exp_name'], config['dadpnet_net_arch']))



    if 'lstmmot' in config['exp_name'] or 'cvpr2018dg' in config['exp_name']  or 'resnet18' in config['exp_name']:
        # --- At this point 'scores'(variable: scores) only have labels rather than classification scores
        
        lstm_scores = copy.deepcopy(scores)
        comb_scores = copy.deepcopy(scores)

        # print('scores: ',scores)
        # print('lstm scores: ',lstm_scores)
        with torch.no_grad():
            test_iter = iter(test_dataloader)            
            num_test_batches = len(test_dataloader)
            print('NUM_TEST_BATCHES: ',num_test_batches)
            
            for bid in range(0, num_test_batches):
                if config['const_testset']:
                    vid_batch, sample_id, gt_labels = next(test_iter)
                else:
                    vid_batch, sample_id, gt_labels = next(test_iter)
                vid_batch = vid_batch.to(device)
                cls_scores, lstm_cls_scores = trainer.forward_lstm(vid_batch)# --- output are classification scores
                cls_scores = torch.exp(cls_scores) # --- 32*2
                lstm_cls_scores = torch.exp(lstm_cls_scores) # --- 4*2
                cls_sc_as_lstm = cls_scores.view(bsize, -1, 2) # --- cls_scores as lstm 4*8*2

                # --- Sizes: 
                # print('cls_scores: {}'.format(cls_scores))
                # print('lstm_cls_scores: {}'.format(lstm_cls_scores))
                # print('cls_sc_as_lstm size: {}'.format(cls_sc_as_lstm))
                # --- Calculate combined classification scores
                comb_cls_scores = 0.5*torch.mean(cls_sc_as_lstm,1) + 0.5*lstm_cls_scores # --- 4*2

                for eid in range(bsize): # --- each eid stands for a video                                        
                    # str_sid = None
                    if dataset_name == 'siw': # eval_type == 'intra_test':
                        str_sid = sample_id[eid].split('/')
                        str_sid = '{}/{}/{}'.format(str_sid[1], str_sid[2], str_sid[3])
                    elif dataset_name == 'oulu-npu': # eval_type == 'cross_test':
                        str_sid = sample_id[eid].split('/')[1]
                    elif dataset_name == 'casia' or dataset_name == 'replay-attack' or dataset_name == 'casia-maged' or dataset_name == 'replay-attack-maged':
                        
                        str_sid = sample_id[eid].split('/')
                        str_sid = '{}/{}'.format(str_sid[1], str_sid[2])
                    elif dataset_name == 'msu':
                            str_sid = sample_id[eid].split('/')
                            str_sid = '{}/{}'.format(str_sid[1], str_sid[2])
                    else:
                        pass
                    
                    resnet_cls_scores = torch.mean(cls_sc_as_lstm,1)
                    scores[str_sid][0].append(resnet_cls_scores[eid, 0].item())
                    lstm_scores[str_sid][0].append(lstm_cls_scores[eid, 0].item())
                    comb_scores[str_sid][0].append(comb_cls_scores[eid, 0].item())
                    
                if bid % 80 == 0:
                    print('bid [{}]'.format(bid))

                # if bid >= 10:
                #     break
        for x in scores.keys():
            if len(scores[x][0]) > 0:
                eval_out_file_vl_anetsc.write('{:.8f}, {}, {}\n'.format(stat.mean(scores[x][0]), scores[x][1], x))
            else:
                pass

        for y in lstm_scores.keys():
            if len(lstm_scores[y][0]) > 0:                
                eval_out_file_vl_lstm_sc.write('{:.8f}, {}, {}\n'.format(stat.mean(lstm_scores[y][0]), lstm_scores[y][1], y))
            else:
                pass

        for z in comb_scores.keys():
            if len(comb_scores[z][0]) > 0:    
                eval_out_file_vl_comb_sc.write('{:.8f}, {}, {}\n'.format(stat.mean(comb_scores[z][0]), comb_scores[z][1], z))
            else:
                pass
                
        print()
        print('----------------------------------------------------------')
        print('ANet video-level scores written to {}'.format(fname_video_level_anet_scores))
        print('LSTM video-level scores written to {}'.format(fname_video_level_lstm_scores))
        print('COMB video-level scores written to {}'.format(fname_video_level_comb_scores))
        print('----------------------------------------------------------')
        print()
        eval_out_file_vl_anetsc.close()       
        eval_out_file_vl_lstm_sc.close()
        eval_out_file_vl_comb_sc.close()
        trainer.set2train()
        return fname_video_level_anet_scores,fname_video_level_lstm_scores,fname_video_level_comb_scores, evalOutpath
        # return eval_out_file_vl_anetsc, eval_out_file_vl_lstm_sc,eval_out_file_vl_comb_sc,evalOutpath, _
    else:
        print('run_eval.py --> eval() net_type is incorrect!!! exiting ...')
        sys.exit()
    trainer.set2train()

def eval_all(config, trainer, source_domain_list, iterations, epoc_cnt, eval_outpath, test_loaders, num_exmps, source_score_list, gtFlags, debug, min_l2norm_val_previous=None, max_l2norm_val_previous=None, anet_loss='softmax', tsne_feats = None):
    if debug:
        print('Running Eval_all ... --> Iteration: {}; Epoch: {}'.format(iterations + 1, epoc_cnt + 1))
    trainer.set2eval()
    print('>>>>>> eval <<<<<')
    # print('???? Source domain list: ', source_domain_list)
    # fusion_net_arch = config['fusion_net_arch']
    if config['net_type'] == 'lstmmot' or config['net_type'] == 'cvpr2018dg':
        batch_size = config['batch_size_lstm']
    else:
        batch_size = config['batch_size']
    
    device = config['device']
    # sel_these_many = config['test_dataset_conf'][dataset_name]['sel_these_many']
    # testset_protocol = config['test_dataset_conf'][dataset_name]['protocol']
    # testset_split = config['test_dataset_conf'][dataset_name]['split']
    # out_path = join(eval_outpath, dataset_name, 'Protocol_{}'.format(testset_protocol), 'Split_{}'.format(testset_split))
    num_test_batches = math.floor(num_exmps / batch_size)
    last_batch_size = num_exmps % batch_size
    if last_batch_size > 0:
        num_test_batches += 1
    # save_file_path = join(out_path, 'sel_these_many_{}'.format(sel_these_many))
    # save_file_path_tsne = join(out_path, 'sel_these_many_{}'.format(sel_these_many), 'tsne_feats_{:08d}'.format(iterations + 1))

    out_path = eval_outpath
    # save_file_path = join(out_path,'sel_these_many_{}'.format(sel_these_many))
    save_file_path = out_path
    fusiontype = ['sum_fusion', 'mean_fusion', 'mult_fusion']
    # if config['fusion_type']:
    #     if config['fusion_type'] > -1:
    #         save_file_path = join(save_file_path, fusiontype[int(config['fusion_type'])])
    make_dir(save_file_path)
    # make_dir(save_file_path_tsne)
    evalOutpath = save_file_path
    eval_out_file_dnet_minmax_values = None
    eval_out_file_vl_dnetsc = None
    eval_out_file_vl_anetsc = None
    eval_out_file_vl_fusedsc = None
    eval_out_file_vl_fusedsc2 = None
    fname_dnet_minmax_values = None
    fname_video_level_dnet_scores = None
    fname_video_level_anet_scores = None
    fname_video_level_fused_scores = None
    fname_video_level_fused_scores2 = None

    # --- added
    fname_video_level_lstm_scores = None
    fname_video_level_comb_scores = None
    # --- 

    # if 'dnet' in config['exp_name'] or 'dadpnet' in config['exp_name'] and config['dadpnet_net_arch'] == 0:
    #     fname_dnet_minmax_values = join(save_file_path, 'dnet_minmax_values_{:08d}.txt'.format(iterations + 1))
    #     fname_video_level_dnet_scores = join(save_file_path, 'dnet_scores_vl_{:08d}.txt'.format(iterations + 1))
    #     eval_out_file_dnet_minmax_values = open(fname_dnet_minmax_values, 'w')
    #     eval_out_file_vl_dnetsc = open(fname_video_level_dnet_scores, 'w')
    # elif 'anet' in config['exp_name'] or 'resnet' in config['exp_name'] or \
    #         'dadpnet' in config['exp_name'] and config['dadpnet_net_arch'] == 2 or\
    #         'dadpnet' in config['exp_name'] and config['dadpnet_net_arch'] == 3:
    #     fname_video_level_anet_scores = join(save_file_path, 'anet_scores_vl_{:08d}.txt'.format(iterations + 1))
    #     eval_out_file_vl_anetsc = open(fname_video_level_anet_scores, 'w')


    if 'lstmmot' in config['exp_name'] or 'cvpr2018dg' in config['exp_name'] or 'resnet18' in config['exp_name']:
        batch_size = config['batch_size_lstm']
        fname_video_level_anet_scores = join(save_file_path, 'anet_scores_vl_{:08d}.txt'.format(iterations + 1))
        eval_out_file_vl_anetsc = open(fname_video_level_anet_scores, 'w')
        fname_video_level_lstm_scores = join(save_file_path, 'lstm_scores_vl_{:08d}.txt'.format(iterations + 1))
        eval_out_file_vl_lstm_sc = open(fname_video_level_lstm_scores, 'w')
        fname_video_level_comb_scores = join(save_file_path, 'comb_scores_vl_{:08d}.txt'.format(iterations + 1))
        eval_out_file_vl_comb_sc = open(fname_video_level_comb_scores, 'w')

    bsize = batch_size
    print('>>>> Batch Size: {} <<<<'.format(bsize))
    dnet_mean_sc_test = None
    fused_sc_test = None
    print('{}  {}'.format(config['exp_name'], config['dadpnet_net_arch']))



    if 'lstmmot' in config['exp_name'] or 'cvpr2018dg' in config['net_type'] or 'resnet18' in config['net_type']:
        # --- At this point 'scores'(variable: scores) only have labels rather than classification scores
        
        for i in range(len(test_loaders)):
            test_dataloader = test_loaders[i]
            scores = source_score_list[i]
            dom = source_domain_list[i]

            lstm_scores = copy.deepcopy(scores)
            comb_scores = copy.deepcopy(scores)
            with torch.no_grad():
                test_iter = iter(test_dataloader)
                num_test_batches = len(test_dataloader)
                print('NUM_TEST_BATCHES: ',num_test_batches)
                
                for bid in range(0, num_test_batches):
                    if config['const_testset']:
                        vid_batch, sample_id, gt_labels = next(test_iter)
                    else:
                        vid_batch, sample_id, gt_labels = next(test_iter)
                    vid_batch = vid_batch.to(device)

                    cls_scores, lstm_cls_scores = trainer.forward_lstm(vid_batch) # --- output are classification scores
                    cls_scores = torch.exp(cls_scores) # --- 32*2
                    lstm_cls_scores = torch.exp(lstm_cls_scores) # --- 4*2

                    cls_sc_as_lstm = cls_scores.view(bsize, -1, 2) # --- cls_scores as lstm 4*8*2

                    comb_cls_scores = 0.5*torch.mean(cls_sc_as_lstm,1) + 0.5*lstm_cls_scores # --- 4*2

                    for eid in range(bsize): # --- each eid stands for a video                                        

                        # str_sid = None
                        if dom == 'siw': # eval_type == 'intra_test':
                            str_sid = sample_id[eid].split('/')
                            str_sid = '{}/{}/{}'.format(str_sid[1], str_sid[2], str_sid[3])
                        elif dom == 'oulu-npu': # eval_type == 'cross_test':
                            str_sid = sample_id[eid].split('/')[1]
                        elif dom == 'casia' or dom == 'replay-attack' or dom == 'casia-maged' or dom == 'replay-attack-maged':
                            str_sid = sample_id[eid].split('/')
                            str_sid = '{}/{}'.format(str_sid[1], str_sid[2])
                        elif dom == 'msu':
                                str_sid = sample_id[eid].split('/')
                                str_sid = '{}/{}'.format(str_sid[1], str_sid[2])
                        else:
                            pass
                        
                        resnet_cls_scores = torch.mean(cls_sc_as_lstm,1)

                        scores[str_sid][0].append(resnet_cls_scores[eid, 0].item())
                        lstm_scores[str_sid][0].append(lstm_cls_scores[eid, 0].item())
                        comb_scores[str_sid][0].append(comb_cls_scores[eid, 0].item())
                        
                    if bid % 80 == 0:
                        print('bid [{}]'.format(bid))

                    # if bid >= 10:
                    #     break
            
            for x in scores.keys():
                if len(scores[x][0]) > 0:
                    eval_out_file_vl_anetsc.write('{:.8f}, {}, {}_{}\n'.format(stat.mean(scores[x][0]), scores[x][1],dom ,x))
                else:
                    pass

            for y in lstm_scores.keys():
                if len(lstm_scores[y][0]) > 0:                
                    eval_out_file_vl_lstm_sc.write('{:.8f}, {}, {}_{}\n'.format(stat.mean(lstm_scores[y][0]), lstm_scores[y][1],dom,y))
                else:
                    pass

            for z in comb_scores.keys():
                if len(comb_scores[z][0]) > 0:    
                    eval_out_file_vl_comb_sc.write('{:.8f}, {}, {}_{}\n'.format(stat.mean(comb_scores[z][0]), comb_scores[z][1],dom, z))
                else:
                    pass
                
        print()
        print('----------------------------------------------------------')
        print('ANet video-level scores written to {}'.format(fname_video_level_anet_scores))
        print('LSTM video-level scores written to {}'.format(fname_video_level_lstm_scores))
        print('COMB video-level scores written to {}'.format(fname_video_level_comb_scores))
        print('----------------------------------------------------------')
        print()
        eval_out_file_vl_anetsc.close()       
        eval_out_file_vl_lstm_sc.close()
        eval_out_file_vl_comb_sc.close()
        trainer.set2train()
        return fname_video_level_anet_scores,fname_video_level_lstm_scores,fname_video_level_comb_scores, evalOutpath
        # return eval_out_file_vl_anetsc, eval_out_file_vl_lstm_sc,eval_out_file_vl_comb_sc,evalOutpath, _
    else:
        print('run_eval.py --> eval() net_type is incorrect!!! exiting ...')
        sys.exit()
    trainer.set2train()

