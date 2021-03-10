import os
from os.path import join
import numpy as np
from argparse import ArgumentParser

# Import from torch
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score,roc_curve

import sys
sys.path.append("src")
print(os.getcwd())
from ocda_fas.utils.data_utils import make_exp_dir,get_domain_list,domain_combined_data_loaders
from utils import get_config, make_dir
from utils_dg import get_data_loader as get_dataloader_train,get_part_labels
from data_loader_anet import get_dataset
from torch.utils.tensorboard import SummaryWriter
from source.algorithms.eval import perf_measure
from source.models.dg_resnet2 import DgEncoder
from schedulers import PolynomialLR

import pdb


def train_epoch(config,net,train_data_loader,optimizer,criterion,writer,epoch):

    print_step=500
    device=config['device']
    dataloader_len=len(train_data_loader)

    net.train()
    predict_lst=[]
    label_lst=[]
    running_loss=0.0
    for batch_idx, (data,_,labels) in enumerate(train_data_loader):
        
        data = data.to(device)
        data.require_grad = False
        labels = labels.to(device)
        labels.require_grad = False

        optimizer.zero_grad()
        x,score,prob= net(data)

        loss = criterion(score.clone(), labels)
        loss.backward()
        optimizer.step()

        pred = torch.argmax(prob, dim=1) 

        running_loss+=loss.item()
        predict_lst+= pred.tolist()
        label_lst+= labels.tolist()

        if (batch_idx+1)%print_step==0:
            avg_loss=running_loss/(batch_idx+1)
            acc=accuracy_score(label_lst,predict_lst)
            print("Train:: Epoch:{} [{}/{} ({:.2f})] Loss:{:.5f} Acc:{:.4f}".format(epoch,batch_idx+1,dataloader_len,(batch_idx+1)/dataloader_len,avg_loss,acc))

            writer.add_scalar('Train/loss',avg_loss ,(epoch*dataloader_len)+(batch_idx+1))
            writer.add_scalar('Train/accuracy', acc, (epoch*dataloader_len)+(batch_idx+1))

            # reintialize
            running_loss=0.0
            predict_lst=[]
            label_lst=[]


        if config['ocda_debug']:
            break

    return 0

def eval_epoch(config,net,val_data_loader,test_data_loader,writer,epoch):

    device=config['device']
    net.eval()
    prob_lst=[]
    label_lst=[]
    with torch.no_grad():
        for idx, (data,_,labels) in enumerate(tqdm(val_data_loader)):
            
            data = data.to(device)
            data.require_grad = False
            labels = labels.to(device)
            labels.require_grad = False

            x,score,prob= net(data)

            prob_lst+=prob[:,0].tolist()
            label_lst+=labels.tolist()

            if config['ocda_debug']:
                break

    # print(label_lst)
    # print(prob_lst)
    fpr, tpr, thresholds = roc_curve(label_lst, prob_lst, pos_label=0)
    fnr=1-tpr
    diff=np.absolute(fpr-fnr)
    idx=diff.argmin()
    thrs=thresholds[idx]
    hter=(fpr[idx]+fnr[idx])/2
    print("Val :: HTER :{} Threshold:{} idx:{} FAR:{} FRR:{}".format(hter, thrs,idx,fpr[idx],fnr[idx]))

    writer.add_scalar('Eval/val_hter',hter ,epoch)
    writer.add_scalar('Eval/val_threshold', thrs, epoch)   

    predict_lst=[]
    label_lst=[]

    if 'f_summary_file' in config.keys():
        fsum=open(config["f_summary_file"],'a')
        fsum.write("\nEpoch:{}\n".format(epoch+1))
        fsum.write("Val_HTER:{} Val_thrs:{} Val_FAR:{} Val_FRR:{} idx:{}\n".format(hter, thrs,fpr[idx],fnr[idx],idx))
        fsum.close()

    with torch.no_grad():
        for idx, (data,_,labels) in enumerate(tqdm(test_data_loader)):

            data = data.to(device)
            data.require_grad = False
            labels = labels.to(device)
            labels.require_grad = False

            x,score,prob= net(data)

            pred = prob[:,0]<thrs
            pred=pred.double()

            predict_lst+=pred.tolist()
            label_lst+=labels.tolist()
            
            if config['ocda_debug']:
                break

    tp, fp, tn, fn=perf_measure(label_lst,predict_lst)
    fpr=fp/(tn+fp) # False rejection rate
    fnr=fn/(tp+fn) # false acceptance rate
    hter= (fpr+fnr)/2
    acc=(tp+tn)/(tp+fp+tn+fn)
    print("Test :: HTER :{} FAR:{} FRR:{}".format(hter,fpr,fnr))

    if 'f_summary_file' in config.keys():
        fsum=open(config["f_summary_file"],'a')
        fsum.write("Test_HTER:{} Test_thrs:{} Test_FAR:{} Test_FRR:{}\n".format(hter,thrs,fpr,fnr))
        fsum.close()

    writer.add_scalar('Eval/test_hter',hter ,epoch)

    return 0


def train_src_net(args):

    """Main function for training mann."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
    config_fname='src/configs/train.yaml'
    config= get_config(config_fname)
    config['device']=device
    config['ocda_debug']=args.debug
    config['net_type']=args.net_type
    config['srcnet_outpath']=os.path.join(args.experiment_path,args.srcnet_outpath)
    os.makedirs(config['srcnet_outpath'],exist_ok=True)
    config['srcnet_exp_path']=make_exp_dir(config['srcnet_outpath'],"src_net")
    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)

    # domain list
    source_domain_list,target_domain_list= get_domain_list(config,'src_net')

    #  source and target Data loaders
    train_data_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='train',net='src_net',type='src')
    val_data_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='val',net='src_net',type='src')
    test_data_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='test',net='src_net',type='src')


    # Target test Data loaders
    
    ###########################
    # Setup networks #
    ###########################
    num_cls=2
    net = DgEncoder(config)
    net.to(device)

    # print network and arguments
    print('Training Src model for {} datasets'.format(config['src_net']['src_dataset'] ))

    ######################
    # Optimization setup #
    ######################
    lr= config['src_net']['lr']
    weight_decay= config['src_net']['weight_decay']
    beta1= config['src_net']['beta1']
    beta2= config['src_net']['beta2']
    betas=(beta1,beta2)

    if config['ocda_debug']:
        num_epoch=2
    else:
        num_epoch= config['src_net']['epochs']

    optimizer = optim.Adam(net.parameters(), lr=lr, 
                         weight_decay=weight_decay, betas=betas)

    decay_iter= config['src_net']['lr_scheduler']['decay_iter']
    gamma= config['src_net']['lr_scheduler']['gamma']
    scheduler_net= PolynomialLR(optimizer,num_epoch,decay_iter,gamma)

    criterion=torch.nn.CrossEntropyLoss()
    ##############
    # Train mann #
    #############

    #folder creation
    checkpoint_path=os.path.join(config['srcnet_exp_path'],'checkpoints')
    score_files=os.path.join(config['srcnet_exp_path'],'score_files')
    os.mkdir(checkpoint_path)
    os.mkdir(score_files)
    
    f_summary_file=os.path.join(config['srcnet_exp_path'],"summary.txt")
    config["f_summary_file"]=f_summary_file

    tnsrboard_path=os.path.join(config['srcnet_exp_path'],'tensorboardfiles')
    writer = SummaryWriter(tnsrboard_path)

    #initial Evaluation
    eval_epoch(config,net,val_data_loader,test_data_loader,writer,-1)

    for epoch in range(num_epoch):
        print("Learning Rate:",optimizer.param_groups[0]['lr'])
        train_epoch(config,net,train_data_loader,optimizer,criterion,writer,epoch) 

        eval_epoch(config,net,val_data_loader,test_data_loader,writer,epoch)

        scheduler_net.step()
        ##############
        # Save Model #
        ##############
        outfile = join(checkpoint_path, 'src_net_{:s}_epoch{:02d}.pt'.format(config['src_net']['src_dataset'],epoch))
        print('Saving to', outfile)
        net.save(outfile)
    
    writer.close()
    
    print("end")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--net_type', type=str, default='lstmmot')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--experiment_path', type=str, default='output/fas_project/ocda_exp')
    parser.add_argument('--srcnet_outpath', type=str, default='ocda_rev/src_net')

    args = parser.parse_args()
    train_src_net(args)
