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
from ocda_fas.utils.data_utils import get_domain_list,domain_combined_data_loaders,make_exp_dir
from source.algorithms.baseline import DABaselineSrc,DABaselineTgt_ADDA,DABaselineTgt_GRL,SrcTgtDist

def baseline_src_eval(args):
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
    config['da_src_checkpoint']= join(config['da_baseline_path'],args.da_src_baseline_checkpoint)

    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)

    os.makedirs(config['da_src_exp_path'], exist_ok=True)
    os.makedirs(config['da_baseline_tnsorboard'], exist_ok=True)

    #******* Dataloader *************
    source_domain_list,target_domain_list= get_domain_list(config,'da_baseline')

    src_val_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='val',net='da_baseline',type='src')
    src_test_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='test',net='da_baseline',type='src')
    
    # Model 
    src_baseline=DABaselineSrc(config,configdl)
    src_baseline.load(config['da_src_checkpoint'])
    #******* src eval loop *************

    print("***** Src Eval Start ******")
    loss_val,acc_val,hter_val,eer_thr=src_baseline.val(src_val_loader,type="val",eer_thr=None)
    loss_test,acc_test,hter_test,eer_thr=src_baseline.val(src_test_loader,type="test",eer_thr=eer_thr)
    print("*** End ***")


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

def baseline_tgt_train_ADDA(args):

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
    tgt_baseline=DABaselineTgt_ADDA(config,configdl,writer)

    if config['debug']:
        num_epoch=1
    else:
        num_epoch= config['da_baseline']['tgt']['epochs']

    #******* src train loop *************
    best_loss=0.0
    print("***** Src Training Start ******")
    for epoch in range(num_epoch):
        print("Epoch:",epoch)
        best_loss=tgt_baseline.train_epoch(src_train_loader,tgt_train_loader,tgt_val_loader,epoch,num_epoch,best_loss)

    print("*** End ***")

    config_write_loc=join(config['da_tgt_exp_path'],'config.yaml')
    with open(config_write_loc, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    writer.close()


def baseline_tgt_train_GRL(args):

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
    print("Exp path: ",config['da_tgt_exp_path'] )
    os.makedirs(config['da_baseline_tnsorboard'], exist_ok=True)

    #******* Dataloader *************
    source_domain_list,target_domain_list= get_domain_list(config,'da_baseline')

    src_train_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='train',net='da_baseline',type='src')
    tgt_train_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='train',net='da_baseline',type='tgt')

    src_val_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='val',net='da_baseline',type='src')
    src_test_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='test',net='da_baseline',type='src')

    tgt_val_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='val',net='da_baseline',type='tgt')
    tgt_test_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='test',net='da_baseline',type='tgt')

    if (len(src_train_loader.dataset)<len(tgt_train_loader.dataset)):
        comb_dataset = torch.utils.data.ConcatDataset([src_train_loader.dataset,src_train_loader.dataset])
        src_train_loader = torch.utils.data.DataLoader(comb_dataset, batch_size=config['da_baseline']['batch_size_src'], shuffle=True,drop_last=True,num_workers=config['num_workers'])

    #******* src class *************
    writer = SummaryWriter(config['da_baseline_tnsorboard'])
    tgt_baseline=DABaselineTgt_GRL(config,configdl,writer)
    tgt_baseline.load("output/fas_project/ocda_exp/ocda_expand_improve/baseline_grl/tgt_net_exp_000/GRL_checkpoint_8.pt")
    if config['debug']:
        num_epoch=2
    else:
        num_epoch= config['da_baseline']['tgt']['epochs']

    #******* src train loop *************
    config_write_loc=join(config['da_tgt_exp_path'],'config.yaml')
    with open(config_write_loc, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # inital Evaluation
    # print("***** Initial Evaluation Start ******")
    # _,_,_,tgt_eer_thr=tgt_baseline.val(tgt_val_loader,domain='tgt',epoch=-1,type="val")
    # tgt_baseline.val(tgt_test_loader,domain='tgt',epoch=-1,type="test",eer_thr=tgt_eer_thr)

    print("***** Initial Evaluation Start ******")
    _,_,_,src_eer_thr=tgt_baseline.val(src_val_loader,domain='tgt',epoch=-1,type="val")
    tgt_baseline.val(src_test_loader,domain='tgt',epoch=-1,type="test",eer_thr=src_eer_thr)

    print("***** Src Training Start ******")
    for epoch in range(num_epoch):
        print("Epoch:",epoch)

        tgt_baseline.train_epoch(src_train_loader,tgt_train_loader,tgt_val_loader,epoch,num_epoch)

        # _,_,_,src_eer_thr=tgt_baseline.val(src_val_loader,domain='src',epoch=epoch,type="val")
        # tgt_baseline.val(src_test_loader,domain='src',epoch=epoch,type="test",eer_thr=src_eer_thr)

        _,_,_,tgt_eer_thr=tgt_baseline.val(tgt_val_loader,domain='tgt',epoch=epoch,type="val")
        tgt_baseline.val(tgt_test_loader,domain='tgt',epoch=epoch,type="test",eer_thr=tgt_eer_thr)
        
        tgt_baseline.scheduler_net.step()
        tgt_baseline.scheduler_discrim.step()
        
        chk_name="GRL_checkpoint_" +str(epoch+1)+".pt"
        tgt_baseline.save(join(config['da_tgt_exp_path'],chk_name))
        print("*****Checkpoint saved*****")

    print("*** End ***")

    
    writer.close()

def src_tgt_distribution(args):

    ''' To bring src and tgt distribution closer
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    config_fname='src/configs/train.yaml'
    config= get_config(config_fname)
    config['device']=device
    config['debug']=args.debug
    config['net_type']=args.net_type
    config['da_baseline_path']=join(args.experiment_path,args.da_baseline)
    config['da_baseline_tgt_path']=config['da_baseline_path']
    os.makedirs(config['da_baseline_tgt_path'], exist_ok=True)
    config['da_tgt_exp_path']=make_exp_dir(config['da_baseline_tgt_path'],"Dist_net")
    config['da_baseline_tnsorboard']=join(config['da_tgt_exp_path'],"tensorbrd_files")
    config['da_src_checkpoint']= join(config['da_baseline_path'],args.da_src_baseline_checkpoint)

    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)

    os.makedirs(config['da_tgt_exp_path'], exist_ok=True)
    print("Exp path: ",config['da_tgt_exp_path'] )
    os.makedirs(config['da_baseline_tnsorboard'], exist_ok=True)

    #******* Dataloader *************
    source_domain_list,target_domain_list= get_domain_list(config,'da_baseline')

    src_train_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='train',net='da_baseline',type='src')
    tgt_train_loader_live=domain_combined_data_loaders(config,configdl,target_domain_list,mode='val',net='da_baseline',type='tgt',data_filter='live')

    src_val_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='val',net='da_baseline',type='src')


    #******* src class *************
    writer = SummaryWriter(config['da_baseline_tnsorboard'])
    dist_baseline=SrcTgtDist(config,configdl,writer)

    if config['debug']:
        num_epoch=2
    else:
        num_epoch= config['da_baseline']['tgt']['epochs']

    # saving the config
    config_write_loc=join(config['da_tgt_exp_path'],'config.yaml')
    with open(config_write_loc, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    #******* src train loop *************

    #inital Evaluation
    print("***** Initial Evaluation Start ******")
    src_val_mean=dist_baseline.val(src_val_loader,domain='src',epoch=-1,type="val")
    tgt_mean=dist_baseline.val(tgt_train_loader_live,domain='tgt',epoch=-1,type="test")

    mean_dist=dist_baseline.vec_dist(src_val_mean,tgt_mean,epoch=-1,norm_type=2.0)

    writer.add_scalar('Eval/Distance', mean_dist,0)

    print("***** Src Training Start ******")
    for epoch in range(num_epoch):
        print("Epoch:",epoch)

        dist_baseline.train_epoch(src_train_loader,tgt_train_loader_live,epoch,num_epoch)


        src_val_mean=dist_baseline.val(src_val_loader,domain='src',epoch=epoch,type="val")
        tgt_mean=dist_baseline.val(tgt_train_loader_live,domain='tgt',epoch=epoch,type="test")

        mean_dist=dist_baseline.vec_dist(src_val_mean,tgt_mean,epoch,norm_type=2)

        print("\nVal: Epoch:{:d}/{:d} Mean_Dist_src_tgt:{:.4f}".format(epoch,num_epoch,mean_dist),flush=True)

        dist_baseline.scheduler_net.step()
        dist_baseline.scheduler_discrim.step()
        
        chk_name="Dist_similar_checkpoint_" +str(epoch+1)+".pt"
        dist_baseline.save(join(config['da_tgt_exp_path'],chk_name))
        print("*****Checkpoint saved*****")

    print("*** End ***")

    
    writer.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--net_type', type=str, default='lstmmot')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--experiment_path', type=str, default='output/fas_project/DG_exp/lstmmot_exp_013')
    parser.add_argument('--da_baseline', type=str, default='da_baseline/')
    parser.add_argument('--da_src_baseline_checkpoint', type=str, default='src_baseline/src_net_exp_000/src_baseline_checkpoint_best.pt')

    args = parser.parse_args()
    # baseline_src_train(args)
    # baseline_src_eval(args)
    # baseline_tgt_train_ADDA(args)
    baseline_tgt_train_GRL(args)
    # src_tgt_distribution(args)





