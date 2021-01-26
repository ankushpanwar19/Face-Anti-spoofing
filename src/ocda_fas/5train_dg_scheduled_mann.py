import os
from os.path import join
from copy import deepcopy
import yaml
import numpy as np
from argparse import ArgumentParser

# Import from torch
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("src")
from utils import get_config, make_dir
from source.models.dg_resnet2 import DgEncoder
from source.models.dg_mann_net import DgMannNet
from ocda_fas.utils.data_utils import make_exp_dir,get_domain_list,domain_combined_data_loaders,get_multi_domain_dataset,DomainScheduledSampler
from source.algorithms.eval import eval2,eval_scheduled

import pdb


def train_epoch(config,loader_src, loader_tgt, net, domain_factor_net, opt_net, opt_dis, opt_selector,opt_classifier,opt_selector_domain_factor,epoch,writer,device,the=0.6):
   
    print_interval = 10 # specifies how often to display
    tnsorboard_logging_interval = 1000   # specifies how often to display
    
    domain_factor_cond=config['scheduled_mann_net']['domain_factor_cond']

    N = min(len(loader_src.dataset), len(loader_tgt.dataset)) 
    dataloader_len = min(len(loader_src), len(loader_tgt)) 
    joint_loader = zip(loader_src, loader_tgt)

    net.train()
    domain_factor_net.eval()
    last_update = -1

    running_loss_discrim=0.0
    running_loss_gan=0.0
    running_acc_discrim=0.0
    count_discrim_update=0
    count_tgtnet_update=0

    training_set="Training_"+config['scheduled_mann_net']['src_dataset'] +"_"+config['scheduled_mann_net']['tgt_dataset']
    print (training_set)
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
        keys_memory=keys_memory.to(device)
        # computing memory feature by querying and associating visual memory
        values_memory = score_t.clone()
        values_memory = values_memory.softmax(dim=1).to(device)
        memory_feature = torch.matmul(values_memory, keys_memory)

        if domain_factor_cond == 0:
            # computing concept selector
            concept_selector = net.fc_selector(x_t.clone()).tanh()
            x_t = direct_feature + concept_selector * memory_feature
        elif domain_factor_cond == 1:
            with torch.no_grad():
                domain_factor_ftr = domain_factor_net.encoder(data_t).detach()
            domain_factor_selector = net.domain_factor_selector(x_t).tanh()
            x_t = direct_feature + domain_factor_selector * domain_factor_ftr

        # apply cosine norm classifier
        score_t = net.tgt_net.classifier(x_t.clone())
        ###########################

        if config['scheduled_mann_net']['discrim_feat']:
            f = torch.cat((x_s, x_t), 0)
        else:
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

        #running loss and acc for disriminator
        running_loss_discrim+=loss_dis
        running_acc_discrim+=acc
        count_discrim_update+=1

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
            keys_memory=keys_memory.to(device)
            # computing memory feature by querying and associating visual memory
            values_memory = score_t.clone()
            values_memory = values_memory.softmax(dim=1).to(device)
            memory_feature = torch.matmul(values_memory, keys_memory)

            
            if domain_factor_cond == 0:
            # computing concept selector
                concept_selector = net.fc_selector(x_t.clone()).tanh()
                x_t = direct_feature + concept_selector * memory_feature
            elif domain_factor_cond == 1:
                with torch.no_grad():
                    domain_factor_ftr = domain_factor_net.encoder(data_t).detach()
                domain_factor_selector = net.domain_factor_selector(x_t).tanh()
                x_t = direct_feature + domain_factor_selector * domain_factor_ftr
                
            # apply cosine norm classifier
            score_t = net.tgt_net.classifier(x_t.clone())

            ###########################
            # predict with discriinator
            ###########################
            if config['scheduled_mann_net']['discrim_feat']:
                f = x_t
            else:
                f = score_t
            pred_tgt = net.discriminator(score_t)
            
            # create fake label
            label_tgt = torch.ones(pred_tgt.size(0), requires_grad=False).long().to(device)
            
            # compute loss for target network
            loss_gan_t = net.gan_criterion(pred_tgt, label_tgt)

            # zero out optimizer gradients
            opt_dis.zero_grad()
            opt_net.zero_grad()
            opt_selector.zero_grad()
            opt_classifier.zero_grad()
            if opt_selector_domain_factor:
                opt_selector_domain_factor.zero_grad()

            # loss backprop
            loss_gan_t.backward()

            # optimize tgt network
            opt_net.step()
            opt_selector.step()
            opt_classifier.step()
            if opt_selector_domain_factor:
                opt_selector_domain_factor.step()
            # log net update info
            info_str += " G: {:.3f}".format(loss_gan_t.item()) 

            #running loss for TGT NET with gan loss
            running_loss_gan+=loss_dis
            count_tgtnet_update+=1

        ###########
        # Logging #
        ###########
        if (batch_idx+1) % print_interval == 1:
            print(info_str)

        if (batch_idx+1) % tnsorboard_logging_interval == 1:
            disc_loss=running_loss_discrim/count_discrim_update
            disc_acc=running_acc_discrim/count_discrim_update
            gan_loss=0.0
            if count_tgtnet_update>0:
                gan_loss=running_loss_gan/(count_tgtnet_update)
            writer.add_scalar(training_set+'/Discrim_loss', disc_loss, (epoch*dataloader_len)+batch_idx)
            writer.add_scalar(training_set+'/Gan_loss', gan_loss, (epoch*dataloader_len)+batch_idx)
            writer.add_scalar(training_set+'/Acc_domain', disc_acc, (epoch*dataloader_len)+batch_idx)

            running_loss_discrim=0.0
            running_loss_gan=0.0
            running_acc_discrim=0.0
    
        if config['ocda_debug']:
            break

    return last_update


def train_scheduled_mann_multi(args):

    """Main function for training mann."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
    config_fname='src/configs/train.yaml'
    config= get_config(config_fname)
    config['device']=device
    config['ocda_debug']=args.debug
    config['net_type']=args.net_type
    config['mann_checkpoint_file']=os.path.join(args.experiment_path,args.mann_checkpoint_file)
    config['domainfactor_checkpoint_file']=os.path.join(args.experiment_path,args.domain_checkpoint_file)
    config['centroids_file']=os.path.join(args.experiment_path,args.centroids_path,config['scheduled_mann_net']['centroid_fname'])
    config['schmannnet_outpath']=os.path.join(args.experiment_path,args.schmannnet_outpath)
    config['scheduledmannnet_exp_path']=make_exp_dir(config['schmannnet_outpath'],"scheduled_mann_net")
    # config['centroids_file']=os.path.join(args.experiment_path,args.centroids_path,config['mann_net']['centroid_fname'])

    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)
    
    #######################################
    # Setup data for training and testing #
    #######################################
    source_domain_list,target_domain_list= get_domain_list(config,'scheduled_mann_net')

    #  source and target Data loaders
    
    src_data_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='train',net='scheduled_mann_net',type='src')

    tgt_train_dataset=get_multi_domain_dataset(config,configdl,target_domain_list,mode='train',drop_last=False)

    tgt_val_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='val',net='scheduled_mann_net',type='tgt')

    tgt_test_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='test',net='scheduled_mann_net',type='tgt')

    ###########################
    # Setup cuda and networks #
    ###########################
    num_cls=2
    # setup network 
    net = DgMannNet(config, num_cls, use_init_weights=False,feat_dim=2048,discrim_feat=False,use_domain_factor_selector=True)
    #load weights
    net.load(config['mann_checkpoint_file'])
    net.to(device)

    domain_factor_net = DgEncoder(config)
    #load weights
    domain_factor_net.load(config['domainfactor_checkpoint_file'],'domain_encoder','classifier')
    # set to eval
    domain_factor_net.to(device)
    domain_factor_net.eval()
    

    # print network and arguments
    # print(net)
    print('Training Scheduled Mann model for {}->{}'.format(config['scheduled_mann_net']['src_dataset'], config['scheduled_mann_net']['tgt_dataset']))


    ######################
    # Optimization setup #
    ######################
    sort_idx_path=os.path.join(args.experiment_path,args.sortidx_for_schedule)
    sort_idx = np.load(sort_idx_path)

    checkpoint_path=os.path.join(config['scheduledmannnet_exp_path'],'checkpoints')

    initial_ratio=config['scheduled_mann_net']['initial_ratio']
    schedule_strategy= config['scheduled_mann_net']['schedule_strategy']
    power=config['scheduled_mann_net']['schedule_power']
    batch_tgt=config['scheduled_mann_net']['batch_size_tgt']

    if config['ocda_debug']:
        num_epoch=1
    else:
        num_epoch= config['scheduled_mann_net']['epochs']

    lr= config['scheduled_mann_net']['lr']
    weight_decay= config['scheduled_mann_net']['weight_decay']
    beta1= config['scheduled_mann_net']['beta1']
    beta2= config['scheduled_mann_net']['beta2']
    num_epoch= config['scheduled_mann_net']['epochs']
    betas=(beta1,beta2)

    opt_net = optim.Adam(net.tgt_net.encoder.parameters(), lr=lr, 
                         weight_decay=weight_decay, betas=betas)
    opt_dis = optim.Adam(net.discriminator.parameters(), lr=lr, 
                         weight_decay=weight_decay, betas=betas)
    opt_selector = optim.Adam(net.fc_selector.parameters(), lr=lr*0.1, 
                              weight_decay=weight_decay, betas=betas)
    opt_classifier = optim.Adam(net.tgt_net.classifier.parameters(), lr=lr*0.1, 
                                weight_decay=weight_decay, betas=betas)

    if config['scheduled_mann_net']['domain_factor_cond'] != 0:
        opt_selector_domain_factor = optim.Adam(net.domain_factor_selector.parameters(), lr=lr*0.1,
                                        weight_decay=weight_decay, betas=betas)
    else:
        opt_selector_domain_factor = None

    #########
    # Train #
    #########
    #folder creation
    checkpoint_path=os.path.join(config['scheduledmannnet_exp_path'],'checkpoints')
    score_files=os.path.join(config['scheduledmannnet_exp_path'],'score_files')
    os.mkdir(checkpoint_path)
    os.mkdir(score_files)
    config['schmannnet_score_files']=score_files

    f_summary_file=os.path.join(config['scheduledmannnet_exp_path'],"summary.txt")
    config["f_summary_file"]=f_summary_file

    config_write_loc=join(config['scheduledmannnet_exp_path'],'config.yaml')
    with open(config_write_loc, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    tnsrboard_path=os.path.join(config['scheduledmannnet_exp_path'],'tensorboardfiles')
    writer = SummaryWriter(tnsrboard_path)



    scheduled_ratio = lambda ep: (1. - initial_ratio) / ((num_epoch - 5) ** power) * (ep ** power) + initial_ratio
    # here 30 means ratio will be greater than 1 for last 30 sepchs

    # train_tgt_loader = load_data_multi(tgt_list, 'train', batch=batch,
    #                                    rootdir=datadir, num_channels=net.num_channels,
    #                                    image_size=net.image_size, download=True, kwargs=kwargs)


    # hter,acc=eval2(config,tgt_val_loader,tgt_test_loader,net,-1,writer)
    hter,acc=eval_scheduled(config,tgt_val_loader,tgt_test_loader,net,domain_factor_net,-1,writer)
    print("Epoch {} HTER {}  acc {}".format(-1,hter,acc))

    schedule_strategy=config['scheduled_mann_net']['schedule_strategy']
    for epoch in range(num_epoch):

        # Calculate current domain ratio
        ratio = float(scheduled_ratio(epoch))

        # actual_lr = ratio * lr

        # for param_group in opt_net.param_groups:
        #     param_group['lr'] = actual_lr
        # for param_group in opt_dis.param_groups:
        #     param_group['lr'] = actual_lr
        # for param_group in opt_selector.param_groups:
        #     param_group['lr'] = actual_lr * 0.1
        # for param_group in opt_classifier.param_groups:
        #     param_group['lr'] = actual_lr * 0.1

        # if config['scheduled_mann_net']['domain_factor_cond'] != 0:
        #     for param_group in opt_net.param_groups:
        #         param_group['lr'] = actual_lr * 0.1

        if ratio < 1:
            # Use sampler for data loading
            print('Epoch: {}, using sampler'.format(epoch))
            sampler = DomainScheduledSampler(tgt_train_dataset, sort_idx, ratio,
                                             initial_ratio, schedule_strategy, seed=epoch)
            train_tgt_loader = torch.utils.data.DataLoader(tgt_train_dataset, batch_size=batch_tgt,shuffle=False, sampler=sampler,drop_last=True,num_workers=config['num_workers'])
        else:
            print('Epoch: {}, using default'.format(epoch))
            train_tgt_loader = torch.utils.data.DataLoader(tgt_train_dataset, batch_size=batch_tgt, shuffle=True,drop_last=True,num_workers=config['num_workers'])

        err = train_epoch(config,src_data_loader, train_tgt_loader, net, domain_factor_net, opt_net, opt_dis, opt_selector,opt_classifier,opt_selector_domain_factor,epoch,writer,device)

        hter,acc=eval_scheduled(config,tgt_val_loader,tgt_test_loader,net,domain_factor_net,epoch,writer)
        print("Epoch {} HTER {}  acc {}".format(epoch,hter,acc))

        if err == -1:
            print("No suitable discriminator")
        
        outfile = join(checkpoint_path, 'schmann_net_{:s}_{:s}_epoch{:02d}.pt'.format(config['scheduled_mann_net']['src_dataset'], config['scheduled_mann_net']['tgt_dataset'],epoch+1))
        print('Saving to', outfile)
        net.save(outfile)
    writer.close()

    print("end")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--net_type', type=str, default='lstmmot')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--experiment_path', type=str, default='output/fas_project/DG_exp/lstmmot_exp_013')
    parser.add_argument('--schmannnet_outpath', type=str, default='ocda_fas_files/scheduled_mann_net')
    parser.add_argument('--mann_checkpoint_file', type=str, default='ocda_fas_files/mann_net/mann_net_exp_009/checkpoints/mann_net_MsCaOu_Ce_epoch03.pt')
    parser.add_argument('--domain_checkpoint_file', type=str, default='ocda_fas_files/domainfactor/domainfactor_net_exp_000/checkpoints/DomainFactorNet_MsCaOu_Ce_5.pt')
    parser.add_argument('--sortidx_for_schedule', type=str, default='ocda_fas_files/domainfactor/domainfactor_net_exp_000/sortidx_for_schedule.npy')
    parser.add_argument('--centroids_path', type=str, default='ocda_fas_files')
    parser.add_argument('--norm_domain_factor', type=bool, default=True)
    args = parser.parse_args()

    train_scheduled_mann_multi(args)

    print("end")