import os
from os.path import join
from copy import deepcopy
import numpy as np
from argparse import ArgumentParser

# Import from torch
import torch
import torch.optim as optim

import sys
sys.path.append("src")
from utils import get_config, make_dir
from source.models.dg_resnet import DgEncoder
from source.models.dg_mann_net import DgMannNet
from data_utils import get_domain_list,domain_combined_data_loaders,get_multi_domain_dataset,DomainScheduledSampler
from eval import eval_tgt

import pdb


def train_epoch(loader_src, loader_tgt, net, domain_factor_net, opt_net, opt_dis,
                opt_selector_content, epoch, device,the=0.6,domain_factor_cond=1):
   
    log_interval = 10  # specifies how often to display
  
    N = len(loader_tgt.dataset)
    joint_loader = zip(loader_src, loader_tgt)

    net.train()
    domain_factor_net.eval()
   
    last_update = -1

    for batch_idx, ((data_s, _,_), (data_t, _,_)) in enumerate(joint_loader):
        
        if len(data_s) == 1 or len(data_t) == 1:  # BN protection
            continue

        # log basic mann train info
        info_str = "[Train Schedule Mann] Epoch: {} [{}/{} ({:.2f}%)]".format(epoch, batch_idx * len(data_t),
                                                                             N, 100 * batch_idx * len(data_t) / N)
   
        ########################
        # Setup data variables #
        ########################
        
        data_s = data_s.to(device)
        data_t = data_t.to(device)

        data_s.require_grad = False
        data_t.require_grad = False

        ##########################
        # Optimize discriminator #
        ##########################

        # extract and concat features
        x_s,score_s, _ = net.src_net(data_s.clone())
        x_t,score_t,_ = net.tgt_net(data_t.clone())

        ###########################
        # storing direct feature
        direct_feature = x_t.clone()

        # set up visual memory
        keys_memory = net.centroids.detach().clone()

        # computing memory feature by querying and associating visual memory
        values_memory = score_t.clone()
        values_memory = values_memory.softmax(dim=1)
        memory_feature = torch.matmul(values_memory, keys_memory)

        if domain_factor_cond == 0:
            # computing concept selector
            concept_selector = net.fc_selector(x_t.clone()).tanh()
            x_t = direct_feature + concept_selector * memory_feature
        elif domain_factor_cond == 1:
            with torch.no_grad():
                domain_factor_ftr = domain_factor_net(data_t).detach()
            domain_factor_selector = net.domain_factor_selector(x_t).tanh()
            x_t = direct_feature + domain_factor_selector * domain_factor_ftr
        # apply cosine norm classifier
        score_t = net.tgt_net.gen.fc(x_t.clone())
        ###########################

        f = torch.cat((score_s, score_t), 0)
        
        # predict with discriminator
        pred_concat = net.discriminator(f.clone())

        # prepare real and fake labels: source=1, target=0
        target_dom_s = torch.ones(len(data_s), requires_grad=False).long()
        target_dom_t = torch.zeros(len(data_t), requires_grad=False).long()
        label_concat = torch.cat((target_dom_s, target_dom_t), 0).to(device)

        # compute loss for disciminator
        loss_dis = net.gan_criterion(pred_concat.clone(), label_concat)

        # zero gradients for optimizer
        opt_dis.zero_grad()

        # loss backprop
        loss_dis.backward()

        # optimize discriminator
        opt_dis.step()

        # compute discriminator acc
        pred_dis = torch.squeeze(pred_concat.max(1)[1])
        acc = (pred_dis == label_concat).float().mean()
        
        # log discriminator update info
        info_str += " acc: {:0.1f} D: {:.3f}".format(acc.item()*100, loss_dis.item())

        ###########################
        # Optimize target network #
        ###########################

        # only update net if discriminator is strong
        if acc.item() > the:
            
            last_update = batch_idx
        
            # extract target features
            x_t,score_t,_  = net.tgt_net(data_t.clone())

            ###########################
            # storing direct feature
            direct_feature = x_t.clone()

            # set up visual memory
            keys_memory = net.centroids.detach().clone()

            # computing memory feature by querying and associating visual memory
            values_memory = score_t.clone()
            values_memory = values_memory.softmax(dim=1)
            memory_feature = torch.matmul(values_memory, keys_memory)

            
            if domain_factor_cond == 0:
            # computing concept selector
                concept_selector = net.fc_selector(x_t.clone()).tanh()
                x_t = direct_feature + concept_selector * memory_feature
            elif domain_factor_cond == 1:
                with torch.no_grad():
                    domain_factor_ftr = domain_factor_net(data_t).detach()
                domain_factor_selector = net.domain_factor_selector(x_t).tanh()
                x_t = direct_feature + domain_factor_selector * domain_factor_ftr
                
            # apply cosine norm classifier
            score_t = net.tgt_net.gen.fc(x_t.clone())

            ###########################
            # predict with discriinator
            ###########################
            pred_tgt = net.discriminator(score_t)
            
            # create fake label
            label_tgt = torch.ones(pred_tgt.size(0), requires_grad=False).long().to(device)
            
            # compute loss for target network
            loss_gan_t = net.gan_criterion(pred_tgt, label_tgt)

            # zero out optimizer gradients
            opt_dis.zero_grad()
            opt_net.zero_grad()

            opt_selector_content.zero_grad()

            # loss backprop
            loss_gan_t.backward()

            # optimize tgt network
            opt_net.step()
            opt_selector_content.step()

            # log net update info
            info_str += " G: {:.3f}".format(loss_gan_t.item()) 

        ###########
        # Logging #
        ###########
        if batch_idx % log_interval == 0:
            print(info_str)

    return last_update


def train_scheduled_mann_multi(args):

    """Main function for training mann."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
    config_fname='src/configs/train.yaml'
    config= get_config(config_fname)
    config['device']=device
    config['net_type']=args.net_type
    config['mann_checkpoint_file']=os.path.join(args.experiment_path,args.mann_checkpoint_file)
    config['domainfactor_checkpoint_file']=os.path.join(args.experiment_path,args.domain_checkpoint_file)
    config['centroids_file']=os.path.join(args.experiment_path,args.centroids_path,config['scheduled_mann_net']['centroid_fname'])
    config['scheduled_outdir']=os.path.join(args.experiment_path,args.scheduled_outdir)
    # config['centroids_file']=os.path.join(args.experiment_path,args.centroids_path,config['mann_net']['centroid_fname'])

    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)
    
    lr= config['scheduled_mann_net']['lr']
    weight_decay= config['scheduled_mann_net']['weight_decay']
    beta1= config['scheduled_mann_net']['beta1']
    beta2= config['scheduled_mann_net']['beta2']
    num_epoch= config['scheduled_mann_net']['epochs']
    betas=(beta1,beta2)

    sort_idx = args.sort_idx
    initial_ratio=0.5
    schedule_strategy= config['scheduled_mann_net']['schedule_strategy']
    power=config['scheduled_mann_net']['schedule_power']
    batch_tgt=config['scheduled_mann_net']['batch_size_tgt']

    num_epochs=config['scheduled_mann_net']['epochs']

    ###########################
    # Setup cuda and networks #
    ###########################
    num_cls=2
    # setup network 
    net = DgMannNet(config,num_cls,use_init_weights=False)
    #load weights
    net.load(config['mann_checkpoint_file'])

    domain_factor_net = DgEncoder(config)
    #load weights
    domain_factor_net.load(config['domainfactor_checkpoint_file'],'domain_gen')
    # set to eval
    domain_factor_net.eval()

    # print network and arguments
    # print(net)
    print('Training Scheduled Mann model for {}->{}'.format(config['scheduled_mann_net']['src_dataset'], config['scheduled_mann_net']['tgt_dataset']))

    #######################################
    # Setup data for training and testing #
    #######################################
    source_domain_list,target_domain_list= get_domain_list(config,'scheduled_mann_net')

    #  source and target Data loaders
    
    src_data_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='train',net='scheduled_mann_net',type='src')

    tgt_train_dataset=get_multi_domain_dataset(config,configdl,target_domain_list,mode='train',drop_last=True)

    ######################
    # Optimization setup #
    ######################

    opt_net = optim.Adam(net.tgt_net.parameters(), lr=lr, 
                         weight_decay=weight_decay, betas=betas)
    opt_dis = optim.Adam(net.discriminator.parameters(), lr=lr, 
                         weight_decay=weight_decay, betas=betas)
    opt_selector_content = optim.Adam(net.fc_selector.parameters(), lr=lr*0.1, 
                                      weight_decay=weight_decay, betas=betas)
    
    # opt_classifier = optim.Adam(net.classifier.parameters(), lr=lr*0.1,
                                # weight_decay=weight_decay, betas=betas)
    # if domain_factor_cond != 0:
    #     opt_selector_domain_factor = optim.Adam(net.domain_factor_selector.parameters(), lr=lr*0.1,
    #                                     weight_decay=weight_decay, betas=betas)
    # else:
    #     opt_selector_domain_factor = None

    #########
    # Train #
    #########

    scheduled_ratio = lambda ep: (1. - initial_ratio) / ((num_epoch - 3) ** power) * (ep ** power) + initial_ratio
    # here 30 means ratio will be greater than 1 for last 30 sepchs

    # train_tgt_loader = load_data_multi(tgt_list, 'train', batch=batch,
    #                                    rootdir=datadir, num_channels=net.num_channels,
    #                                    image_size=net.image_size, download=True, kwargs=kwargs)

    hter,acc=eval_tgt(config,configdl,tgt_test_loader,net)
    print("Start: HTER {}  acc {}".format(hter,acc))

    for epoch in range(num_epoch):

        # Calculate current domain ratio
        ratio = float(scheduled_ratio(epoch))

        actual_lr = ratio * lr

        for param_group in opt_net.param_groups:
            param_group['lr'] = actual_lr
        for param_group in opt_dis.param_groups:
            param_group['lr'] = actual_lr
        for param_group in opt_selector_content.param_groups:
            param_group['lr'] = actual_lr * 0.1
        # for param_group in opt_classifier.param_groups:
        #     param_group['lr'] = actual_lr * 0.1
        # if domain_factor_cond != 0:
        #     for param_group in opt_net.param_groups:
        #         param_group['lr'] = actual_lr * 0.1

        if ratio < 1:
            # Use sampler for data loading
            print('Epoch: {}, using sampler'.format(epoch))
            sampler = DomainScheduledSampler(tgt_train_dataset, sort_idx, ratio,
                                             initial_ratio, 'expand', seed=epoch)
            train_tgt_loader = torch.utils.data.DataLoader(tgt_train_dataset, batch_size=batch_tgt,shuffle=False, sampler=sampler,drop_last=True,num_workers=config['num_workers'])
        else:
            print('Epoch: {}, using default'.format(epoch))
            train_tgt_loader = torch.utils.data.DataLoader(tgt_train_dataset, batch_size=batch_tgt, shuffle=True,drop_last=True,num_workers=config['num_workers'])

        err = train_epoch(src_data_loader, train_tgt_loader, net, domain_factor_net, opt_net, opt_dis, opt_selector_content,epoch,device)

        hter,acc=eval_tgt(config,configdl,tgt_test_loader,net)
        print("Epoch {} HTER {}  acc {}".format(epoch,hter,acc))
        if err == -1:
            print("No suitable discriminator")
            break
                
    ##############
    # Save Model #
    ##############
    outdir=config['scheduled_outdir']
    os.makedirs(outdir, exist_ok=True)
    outfile = join(outdir, 'scheduled_{:s}_net_{:s}_{:s}.pth'.format(config['scheduled_mann_net']['src_dataset'], config['scheduled_mann_net']['tgt_dataset']))
    print('Saving to', outfile)
    net.save(outfile)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--net_type', type=str, default='lstmmot')
    parser.add_argument('--experiment_path', type=str, default='output/fas_project/DG_exp/lstmmot_exp_013')
    parser.add_argument('--mann_checkpoint_file', type=str, default='ocda_fas_files/mann_net_MsCaOu_Ce.pt')
    parser.add_argument('--domain_checkpoint_file', type=str, default='ocda_fas_files/DomainFactorNet_MsCaOu_Ce.pt')
    parser.add_argument('--src_ftr_file', type=str, default='ocda_fas_files/src_domain_factor_ftr.bin')
    parser.add_argument('--tgt_ftr_file', type=str, default='ocda_fas_files/tgt_domain_factor_ftr.bin')
    parser.add_argument('--centroids_path', type=str, default='ocda_fas_files')
    parser.add_argument('--scheduled_outdir', type=str, default='ocda_fas_files')
    parser.add_argument('--norm_domain_factor', type=bool, default=True)
    args = parser.parse_args()

    src_ftr_file=os.path.join(args.experiment_path,args.src_ftr_file)
    tgt_ftr_file=os.path.join(args.experiment_path,args.tgt_ftr_file)
    src_ftr = np.fromfile(src_ftr_file, dtype=np.float32).reshape(-1, 2048)
    tgt_ftr = np.fromfile(tgt_ftr_file, dtype=np.float32).reshape(-1, 2048)

    # Calculate domain_factor feature centroids from source
    # And calculate distances of target feature to the centroids
    if args.norm_domain_factor:
        src_ftr /= np.linalg.norm(src_ftr, axis=1, keepdims=True)
        tgt_ftr /= np.linalg.norm(tgt_ftr, axis=1, keepdims=True)
        src_center = src_ftr.mean(axis=0)[:, np.newaxis]
        dist = 1. - tgt_ftr.dot(src_center).squeeze()
    else:
        src_center = src_ftr.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(tgt_ftr - src_center, axis=1)

    # Based on domain_factor feature distances to the target, calculate data order
    setattr(args, 'sort_idx', np.argsort(dist))

    train_scheduled_mann_multi(args)

    print("end")