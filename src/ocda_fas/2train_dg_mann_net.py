import os
from os.path import join
from argparse import ArgumentParser

# Import from torch
import torch
import torch.optim as optim

import sys
sys.path.append("src")
print(os.getcwd())
from data_utils import make_exp_dir
from utils import get_config, make_dir
from utils_dg import get_data_loader as get_dataloader_train,get_part_labels
from source.models.dg_mann_net import DgMannNet
from data_loader_anet import get_dataset
from data_utils import get_domain_list,domain_combined_data_loaders
from eval import eval2
# from torch.utils.tensorboard import SummaryWriter

import pdb


def train_epoch(config,loader_src, loader_tgt, net, opt_net, opt_dis, opt_selector,opt_classifier, epoch,device, the=0.6):
   
    log_interval = 10  # specifies how often to display
  
    N = min(len(loader_src.dataset), len(loader_tgt.dataset)) 
    joint_loader = zip(loader_src, loader_tgt)
      
    net.train()
   
    last_update = -1

    for batch_idx, ((data_s, _,_), (data_t,_, _)) in enumerate(joint_loader):
        
        # log basic mann train info
        info_str = "[Train Mann] Epoch: {} [{}/{} ({:.2f}%)]".format(epoch, batch_idx*len(data_t),
                                                                    N, 100 * batch_idx*len(data_t) / N)
   
        ########################
        # Setup data variables #
        ########################
        # if torch.cuda.is_available():
        data_s = data_s.to(device)
        data_t = data_t.to(device)

        data_s.require_grad = False
        data_t.require_grad = False
            
        ##########################
        # Optimize discriminator #
        ##########################

        # zero gradients for optimizer
        opt_dis.zero_grad()
        net.to(device)
        # extract and concat features
        x_s,score_s,prob_s = net.src_net(data_s.clone())
        x_t,score_t,prob_t = net.tgt_net(data_t.clone())

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

        # computing concept selector
        concept_selector = net.fc_selector(x_t.clone()).tanh()
        class_enhancer = concept_selector * memory_feature
        x_t = direct_feature + class_enhancer

        # apply cosine norm classifier
        score_t = net.tgt_net.classifier(x_t.clone())
        ###########################

        if config['mann_net']['discrim_feat']:
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
        
            # zero out optimizer gradients
            opt_dis.zero_grad()
            opt_net.zero_grad()

            opt_selector.zero_grad()
            # opt_classifier.zero_grad()

            # extract target features
            x_t,score_t,prob_t = net.tgt_net(data_t.clone())

            ###########################
            # storing direct feature
            ###########################

            direct_feature = x_t.clone()

            # set up visual memory
            keys_memory = net.centroids.detach().clone()
            keys_memory=keys_memory.to(device)
            # computing memory feature by querying and associating visual memory
            values_memory = score_t.clone()
            values_memory = values_memory.softmax(dim=1).to(device)
            memory_feature = torch.matmul(values_memory, keys_memory)

            # computing concept selector
            concept_selector = net.fc_selector(x_t.clone()).tanh()
            class_enhancer = concept_selector * memory_feature
            x_t = direct_feature + class_enhancer

            # apply cosine norm classifier
            score_t = net.tgt_net.classifier(x_t.clone())

            ###########################
            # predict with discriinator
            ###########################
            if config['mann_net']['discrim_feat']:
                f = x_t
            else:
                f = score_t
            pred_tgt = net.discriminator(f)
            
            # create fake label
            label_tgt = torch.ones(pred_tgt.size(0), requires_grad=False).long().to(device)
            
            # compute loss for target network
            loss_gan_t = net.gan_criterion(pred_tgt, label_tgt)
            loss_gan_t.backward()

            # optimize tgt network
            opt_net.step()

            opt_selector.step()
            # opt_classifier.step()

            # log net update info
            info_str += " G: {:.3f}".format(loss_gan_t.item()) 

        ###########
        # Logging #
        ###########
        if batch_idx % log_interval == 0:
            print(info_str)
        if config['ocda_debug']:
            break

    return last_update


def train_mann_multi(args):

    """Main function for training mann."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
    config_fname='src/configs/train.yaml'
    config= get_config(config_fname)
    config['device']=device
    config['ocda_debug']=args.debug
    config['net_type']=args.net_type
    config['src_checkpoint_file']=os.path.join(args.experiment_path,args.src_checkpoint_file)
    config['mannnet_outpath']=os.path.join(args.experiment_path,args.mannnet_outpath)
    config['centroids_file']=os.path.join(args.experiment_path,args.centroids_path,config['mann_net']['centroid_fname'])
    config['mannnet_exp_path']=make_exp_dir(config['mannnet_outpath'],"mann_net")
    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)

    # domain list
    source_domain_list,target_domain_list= get_domain_list(config,'mann_net')

    #  source and target Data loaders
    # print
    src_data_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='train',net='mann_net',type='src')

    tgt_train_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='train',net='mann_net',type='tgt')

    tgt_val_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='val',net='mann_net',type='tgt')

    tgt_test_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='test',net='mann_net',type='tgt')

    if (len(src_data_loader.dataset)/ len(tgt_train_loader.dataset)< 0.7):
        comb_dataset = torch.utils.data.ConcatDataset([src_data_loader.dataset,src_data_loader.dataset])
        src_data_loader = torch.utils.data.DataLoader(comb_dataset, batch_size=config['mann_net']['batch_size_src'], shuffle=True,drop_last=True,num_workers=config['num_workers'])

    # Target test Data loaders
    
    ###########################
    # Setup networks #
    ###########################
    num_cls=2
    net = DgMannNet(config,num_cls,use_init_weights=True,feat_dim=2048,discrim_feat=config['mann_net']['discrim_feat'])
    
    # print network and arguments
    # print(net)
    print('Training Mann {} model for {}->{}'.format('DgMannNet',config['mann_net']['src_dataset'], config['mann_net']['tgt_dataset']))

    ######################
    # Optimization setup #
    ######################
    lr= config['mann_net']['lr']
    weight_decay= config['mann_net']['weight_decay']
    beta1= config['mann_net']['beta1']
    beta2= config['mann_net']['beta2']
    betas=(beta1,beta2)

    if config['ocda_debug']:
        num_epoch=1
    else:
        num_epoch= config['mann_net']['epochs']

    
    opt_net = optim.Adam(net.tgt_net.encoder.parameters(), lr=lr, 
                         weight_decay=weight_decay, betas=betas)
    opt_dis = optim.Adam(net.discriminator.parameters(), lr=lr, 
                         weight_decay=weight_decay, betas=betas)
    opt_selector = optim.Adam(net.fc_selector.parameters(), lr=lr*0.1, 
                              weight_decay=weight_decay, betas=betas)
    opt_classifier = optim.Adam(net.tgt_net.classifier.parameters(), lr=lr*0.1, 
                                weight_decay=weight_decay, betas=betas)

    ##############
    # Train mann #
    #############

    #folder creation
    checkpoint_path=os.path.join(config['mannnet_exp_path'],'checkpoints')
    score_files=os.path.join(config['mannnet_exp_path'],'score_files')
    os.mkdir(checkpoint_path)
    os.mkdir(score_files)
    config['mannnet_score_files']=score_files

    f_summary_file=os.path.join(config['mannnet_exp_path'],"summary.txt")
    config["f_summary_file"]=f_summary_file

    # tnsorboard_writer=SummaryWriter(log_dir=)
    # hter,acc=eval2(config,tgt_val_loader,tgt_test_loader,net,-1)
    # print("Start: HTER {}  acc {}".format(hter,acc))

    for epoch in range(num_epoch):
        err = train_epoch(config,src_data_loader, tgt_train_loader, net, opt_net, opt_dis, opt_selector, opt_classifier,epoch,config['device']) 

        hter,acc=eval2(config,tgt_val_loader,tgt_test_loader,net,epoch)
        print("Epoch {} HTER {}  acc {}".format(epoch,hter,acc))

        if err == -1:
            print("No suitable discriminator")

        ##############
        # Save Model #
        ##############
        outfile = join(checkpoint_path, 'mann_net_{:s}_{:s}_epoch{:02d}.pt'.format(config['mann_net']['src_dataset'], config['mann_net']['tgt_dataset'],epoch))
        print('Saving to', outfile)
        net.save(outfile)
    
    
    
    print("end")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--net_type', type=str, default='lstmmot')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--experiment_path', type=str, default='output/fas_project/DG_exp/lstmmot_exp_013')
    parser.add_argument('--src_checkpoint_file', type=str, default='checkpoints/net_00039439.pt')
    parser.add_argument('--mannnet_outpath', type=str, default='ocda_fas_files/mann_net')
    parser.add_argument('--centroids_path', type=str, default='ocda_fas_files')

    args = parser.parse_args()
    train_mann_multi(args)
