import os
from os.path import join
from argparse import ArgumentParser

# Import from torch
import torch
import torch.optim as optim

import sys
sys.path.append("src")
from utils import get_config, make_dir
from source.models.dg_domain_factor_net import DgDomainFactorNet
from data_utils import get_domain_list,domain_combined_data_loaders
import pdb


def soft_cross_entropy(input, target, size_average=True):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def train_epoch(loader_src, loader_tgt, net, opt_domain_factor, opt_decoder, opt_dis_cls, epoch,
                gamma_dispell, gamma_rec, num_cls, fake_label_type,device):
   
    log_interval = 10  # specifies how often to display
  
    N = min(len(loader_src.dataset), len(loader_tgt.dataset)) 
    joint_loader = zip(loader_src, loader_tgt)

    # Only make discriminator trainable
    net.domain_factor_net.train()
    net.domain_factor_net.gen.fc.train()
    # net.domain_factor_net.train()
    net.decoder.eval()
    net.tgt_net.eval()

    net.to(device)
    
    last_update = -1
    for batch_idx, ((data_s,_,cls_s_gt), (data_t,_,cls_t_gt)) in enumerate(joint_loader):
        
        # log basic mann train info
        info_str = "[Train domain_factor Net] Epoch: {} [{}/{} ({:.2f}%)]".format(epoch, batch_idx*len(data_t),
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
        # Optimize Discriminator #
        ##########################

        # extract features and logits
        data_cat = torch.cat((data_s, data_t), dim=0).detach()

        with torch.no_grad():  # content branch

            content_ftr_s,_,_ = net.tgt_net(data_s.clone())
            content_ftr_s = content_ftr_s.detach()

            content_ftr_t,logit_t_pseudo,_  = net.tgt_net(data_t.clone())
            logit_t_pseudo = logit_t_pseudo.detach()
            content_ftr_t = content_ftr_t.detach()

            domain_factor_ftr,_,_ = net.domain_factor_net(data_cat.clone())  # domain_factor branch
            domain_factor_ftr = domain_factor_ftr.detach()

        # predict classes with discriminator using domain_factor feature
        pred_cls_from_domain_factor = net.domain_factor_net.gen.fc(domain_factor_ftr.clone())

        # prepare class labels
        cls_t_pseudo = logit_t_pseudo.argmax(dim=1)
        pseudo_acc = (cls_t_pseudo == cls_t_gt.to(device)).float().mean()  # acc of pseudo label
        info_str += " pseudo_acc: {:0.1f}".format(pseudo_acc.item() * 100)
        cls_real = torch.cat((cls_s_gt.to(device), cls_t_pseudo), dim=0).to(device)  # real

        # compute loss for class disciminator
        loss_dis_cls = net.gan_criterion(pred_cls_from_domain_factor, cls_real)

        # zero gradients for optimizer
        opt_dis_cls.zero_grad()
        # loss backprop
        loss_dis_cls.backward()
        # optimize discriminator
        opt_dis_cls.step()

        # compute discriminator acc
        pred_dis_cls = torch.squeeze(pred_cls_from_domain_factor.argmax(1))
        acc_cls = (pred_dis_cls == cls_real).float().mean()
        
        # log discriminator update info
        info_str += " D_acc: {:0.1f} D_loss: {:.3f}".format(acc_cls.item()*100, loss_dis_cls.item())

        ##########################
        # Optimize domain_factor Network #
        ##########################

        if acc_cls.item() > 0.3:

            # Make domain_factor net trainable
            # net.discriminator_cls.eval()
            # net.discriminator_cls.train()
            net.domain_factor_net.train()
            net.decoder.train()

            # update domain_factor net
            last_update = batch_idx

            ###############
            # GAN loss - domain_factor should not include class information
            # Calculate domain_factors again and predict classes with it
            domain_factor_ftr,pred_cls_from_domain_factor,_ = net.domain_factor_net(data_cat.clone())

            # Calculate loss using random class labels
            if fake_label_type == 'random':
                cls_fake = torch.randint(0, num_cls, (cls_real.size(0),)).long().to(device)
                loss_gan_domain_factor = net.gan_criterion(pred_cls_from_domain_factor, cls_fake)
            elif fake_label_type == 'uniform':
                cls_fake = torch.ones((cls_real.size(0), num_cls), dtype=torch.float32).to(device) * 1. / num_cls
                loss_gan_domain_factor = soft_cross_entropy(pred_cls_from_domain_factor, cls_fake)
            else:
                raise Exception("No such fake_label_type: {}".format(fake_label_type))

            ###############
            # reconstruction loss - However, domain_factor should be able to help reconstruct the data into domain specific appearences

            # Concate source and target contents
            cls_ftr = torch.cat((content_ftr_s, content_ftr_t), 0).detach()
            # Concate contents and domain_factors of each sample and feed into decoder
            combined_ftr = torch.cat((cls_ftr, domain_factor_ftr), dim=1)

            data_rec = net.decoder(combined_ftr)

            # Calculate reconstruction loss based on the decoder outputs
            loss_rec = net.rec_criterion(data_rec, data_cat)

            loss = gamma_dispell * loss_gan_domain_factor + gamma_rec * loss_rec

            opt_dis_cls.zero_grad()
            opt_domain_factor.zero_grad()
            opt_decoder.zero_grad()

            loss.backward()

            opt_domain_factor.step()
            opt_decoder.step()

            info_str += " G_loss: {:.3f}".format(loss_gan_domain_factor.item())
            info_str += " R_loss: {:.3f}".format(loss_rec.item())

        ###########
        # Logging #
        ###########
        if batch_idx % log_interval == 0:
            print(info_str)
        # if batch_idx>0:
        #     break
    return last_update


def train_domain_factor_multi(args):

    """Main function for training domain_factor."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
    config_fname='src/configs/train.yaml'
    config= get_config(config_fname)
    config['device']=device
    config['net_type']=args.net_type
    config['tgt_mann_checkpoint_file']=os.path.join(args.experiment_path,args.tgt_checkpoint_file)
    config['domainfactor_outpath']=os.path.join(args.experiment_path,args.domainfactor_outpath)
    # config['centroids_file']=os.path.join(args.experiment_path,args.centroids_path,config['mann_net']['centroid_fname'])

    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)

    # domain list
    source_domain_list,target_domain_list= get_domain_list(config,'domain_factor_net')

    #  source and target Data loaders
    
    src_data_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='train',net='domain_factor_net',type='src')

    tgt_train_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='train',net='domain_factor_net',type='tgt')

    # tgt_test_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='test',net='domain_factor_net',type='tgt')

    # if (len(src_data_loader.dataset)/ len(tgt_train_loader.dataset)< 0.7):
    #     comb_dataset = torch.utils.data.ConcatDataset([src_data_loader.dataset,src_data_loader.dataset])
    #     src_data_loader = torch.utils.data.DataLoader(comb_dataset, batch_size=config['domain_factor_net']['batch_size_src'], shuffle=True,drop_last=True,num_workers=config['num_workers'])

    ###########################
    # Setup cuda and networks #
    ###########################
    num_cls=2
    # setup network 
    net = DgDomainFactorNet(config,num_cls)
    
    # print network and arguments
    # print(net)
    print('Training domain_factor {} model for {}->{}'.format('DgDomainFactorNet',config['domain_factor_net']['src_dataset'], config['domain_factor_net']['tgt_dataset']))


    ######################
    # Optimization setup #
    ######################
    param_domain_factor_encoder=[{'params': net.domain_factor_net.gen.conv1.parameters()},
        {'params': net.domain_factor_net.gen.bn1.parameters()},
        {'params': net.domain_factor_net.gen.layer1.parameters()},
        {'params': net.domain_factor_net.gen.layer2.parameters()},
        {'params': net.domain_factor_net.gen.layer3.parameters()},
        {'params': net.domain_factor_net.gen.layer4.parameters()}
    ]

    lr= config['domain_factor_net']['lr']
    weight_decay= config['domain_factor_net']['weight_decay']
    beta1= config['domain_factor_net']['beta1']
    beta2= config['domain_factor_net']['beta2']
    num_epoch= config['domain_factor_net']['epochs']
    betas=(beta1,beta2)
    opt_domain_factor = optim.Adam(param_domain_factor_encoder,
                           lr=lr, weight_decay=weight_decay, betas=betas)
    opt_decoder = optim.Adam(net.decoder.parameters(),
                           lr=lr, weight_decay=weight_decay, betas=betas)
    opt_dis_cls = optim.Adam(net.domain_factor_net.gen.fc.parameters(), lr=lr,
                             weight_decay=weight_decay, betas=betas)

    ##############
    # Train Mann #
    ##############
    gamma_dispell = config['domain_factor_net']['gamma_dispell']
    gamma_rec = config['domain_factor_net']['gamma_rec']
    fake_label_type = config['domain_factor_net']['fake_label_type']
    for epoch in range(num_epoch):

        err = train_epoch(src_data_loader, tgt_train_loader, net, opt_domain_factor, opt_decoder, opt_dis_cls,
                          epoch, gamma_dispell, gamma_rec, num_cls, fake_label_type,device)

    ######################
    # Save Total Weights #
    ######################
    outdir=config['domainfactor_outpath']
    os.makedirs(outdir, exist_ok=True)
    outfile = join(outdir, 'DomainFactorNet_{:s}_{:s}_{}.pt'.format(config['domain_factor_net']['src_dataset'], config['domain_factor_net']['tgt_dataset'],num_epoch))
    print('Saving to', outfile)
    net.save(outfile)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--net_type', type=str, default='lstmmot')
    parser.add_argument('--experiment_path', type=str, default='output/fas_project/DG_exp/lstmmot_exp_013')
    parser.add_argument('--tgt_checkpoint_file', type=str, default='ocda_fas_files/mann_net_MsCaOu_Ce.pt')
    parser.add_argument('--domainfactor_outpath', type=str, default='ocda_fas_files')
    # parser.add_argument('--centroids_path', type=str, default='ocda_fas_files')

    args = parser.parse_args()
    train_domain_factor_multi(args)
