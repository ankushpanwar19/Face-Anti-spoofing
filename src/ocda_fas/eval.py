import os
from os.path import join
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from sklearn import metrics
import numpy as np

import sys
sys.path.append("src")
print(os.getcwd())
from utils import get_config, make_dir
from source.models.dg_mann_net import DgMannNet
from data_loader_anet import get_dataset
from data_utils import get_domain_list,domain_combined_data_loaders

def eval2(config,val_test_loader,tgt_test_loader,net,epoch):
    
    device=config['device']
    net.to(device)
    predict_lst=[]
    label_lst=[]
    total=len(val_test_loader)
    path=config['mannnet_score_files']
    
    val_file=os.path.join(path,"val_score_epoch{}.txt".format(epoch))
    
    # threshold=0.2974
    print("******* writing the val Dataset result******")
    f=open(val_file, 'w')
    with torch.no_grad():
        with tqdm(total=total) as pbar:
            for batch_idx, (data_t, t,labels) in enumerate(val_test_loader):

                data_t = data_t.to(device)
                data_t.require_grad = False

                x_t,score_t,_ = net.tgt_net(data_t.clone())

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

                score_t = net.tgt_net.classifier(x_t.clone())
                pred_sc=torch.softmax(score_t,dim=1)
                pred = torch.argmax(pred_sc, dim=1) 

                for i,_ in enumerate(pred):
                    predict_lst.append(pred_sc[i,0].item())
                    label_lst.append(labels[i].item())
                    f.write("{:.5f},{:d},{:s}\n".format(pred_sc[i,0].item(),labels[i].item(),t[i]))
                pbar.update(1)

                if config['ocda_debug']:
                    break
    f.close()
    fpr, tpr, thresholds = metrics.roc_curve(label_lst, predict_lst, pos_label=0)
    fnr=1-tpr
    diff=np.absolute(fpr-fnr)
    idx=diff.argmin()
    thrs=thresholds[idx]
    hter=(fpr[idx]+fnr[idx])/2
    print("Val :: HTER :{} Threshold:{} idx:{} FAR:{} FRR:{}".format(hter, thrs,idx,fpr[idx],fnr[idx]))
    
    if 'f_summary_file' in config.keys():
        fsum=open(config["f_summary_file"],'a')
        fsum.write("\nEpoch:{}\n".format(epoch))
        fsum.write("Val_HTER:{} Val_thrs:{} Val_FAR:{} Val_FRR:{} idx:{}\n".format(hter, thrs,fpr[idx],fnr[idx],idx))
        fsum.close()
        
    ################### TEST Dataset ######################
    total=len(tgt_test_loader)
    predict_lst=[]
    label_lst=[]

    test_file=os.path.join(path,"test_score_epoch{}.txt".format(epoch))
    print("******* writing the test Dataset result******")
    f=open(test_file, 'w')
    with torch.no_grad():
        with tqdm(total=total) as pbar:
            for batch_idx, (data_t, t,labels) in enumerate(tgt_test_loader):

                data_t = data_t.to(device)
                data_t.require_grad = False

                x_t,score_t,_ = net.tgt_net(data_t.clone())

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

                score_t = net.tgt_net.classifier(x_t.clone())
                pred_sc=torch.softmax(score_t,dim=1)
                pred = pred_sc[:,0]<thrs
                pred=pred.double()

                for i,_ in enumerate(pred):
                    predict_lst.append(pred[i].item())
                    label_lst.append(labels[i].item())
                    f.write("{:.5f},{:},{:d},{:s}\n".format(pred_sc[i,0].item(),int(pred[i].item()),labels[i].item(),t[i]))
                
                if config['ocda_debug']:
                    break

    tp, fp, tn, fn=perf_measure(label_lst,predict_lst)
    fpr=fp/(tn+fp) # False rejection rate
    fnr=fn/(tp+fn) # false acceptance rate
    
    hter= (fpr+fnr)/2
    acc=(tp+tn)/(tp+fp+tn+fn)
    print("Test :: HTER :{} FAR:{} FRR:{}".format(hter,fpr,fnr))
    f.write("HTER:{},FAR:{},FRR:{},thr:{}".format(hter,fpr,fnr,thrs))
    f.close()

    if 'f_summary_file' in config.keys():
        fsum=open(config["f_summary_file"],'a')
        fsum.write("Test_HTER:{} Test_thrs:{} Test_FAR:{} Test_FRR:{}\n".format(hter,thrs,fpr,fnr))
        fsum.close()
    return hter,acc


    #*********write score to files

    #***********calc EER, threshold, hter and write to the file
    
    #************get test dataloader

    #************write score to files

    #**************cal  EER , HTER and write to file

# def calc_y_pred(prob)÷
def eval_tgt(config,configdl,tgt_test_loader,net):

    device=config['device']
    net.to(device)
    predict_lst=[]
    label_lst=[]
    total=len(tgt_test_loader)
    with torch.no_grad():
        with tqdm(total=total) as pbar:
            for batch_idx, (data_t, _,labels) in enumerate(tgt_test_loader):

                data_t = data_t.to(device)
                data_t.require_grad = False

                x_t,score_t,_ = net.tgt_net(data_t.clone())

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

                score_t = net.tgt_net.gen.fc(x_t.clone())

                pred = torch.argmax(score_t, dim=1)   

                for i,_ in enumerate(pred):
                    predict_lst.append(pred[i].item())
                    label_lst.append(labels[i].item())
                pbar.update(1)
                # break
    tp, fp, tn, fn=perf_measure(label_lst,predict_lst)
    fpr=fp/(tn+fp) # False rejection rate
    fnr=fn/(tp+fn) # false acceptance rate
    
    hter= (fpr+fnr)/2
    acc=(tp+tn)/(tp+fp+tn+fn)
    return hter,acc


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==0:
           TP += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==1:
           TN += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--net_type', type=str, default='lstmmot')
    parser.add_argument('--experiment_path', type=str, default='output/fas_project/DG_exp/lstmmot_exp_013')
    parser.add_argument('--src_checkpoint_file', type=str, default='checkpoints/net_00039439.pt')
    parser.add_argument('--mannnet_outpath', type=str, default='ocda_fas_files')
    parser.add_argument('--centroids_path', type=str, default='ocda_fas_files')
    args = parser.parse_args()
    # train_mann_multi(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
    config_fname='src/configs/train.yaml'
    config= get_config(config_fname)
    config['device']=device
    config['net_type']=args.net_type
    config['src_checkpoint_file']=os.path.join(args.experiment_path,args.src_checkpoint_file)
    config['mannnet_outpath']=os.path.join(args.experiment_path,args.mannnet_outpath)
    config['centroids_file']=os.path.join(args.experiment_path,args.centroids_path,config['mann_net']['centroid_fname'])
    config['mannnet_chkpt_file']=os.path.join(args.experiment_path,args.centroids_path,'mann_net_MsCaOu_Ce.pt')

    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)

    # domain list
    source_domain_list,target_domain_list= get_domain_list(config,'mann_net')

    source_domain_list,target_domain_list= get_domain_list(config,'mann_net')

    #  source and target Data loaders
    # print
    # src_data_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='train',net='mann_net',type='src')

    # tgt_train_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='train',net='mann_net',type='tgt')

    tgt_test_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='test',net='mann_net',type='tgt')
    val_test_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='val',net='mann_net',type='tgt')

    num_cls=2
    net = DgMannNet(config,num_cls)
    # net.load(config['mannnet_chkpt_file'])

    eval2(config,val_test_loader,tgt_test_loader,net,epoch=0)
