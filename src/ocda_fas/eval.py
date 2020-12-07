import os
from os.path import join
from argparse import ArgumentParser
from tqdm import tqdm

import torch

import sys
sys.path.append("src")
print(os.getcwd())
from utils import get_config, make_dir
from source.models.dg_mann_net import DgMannNet
from data_loader_anet import get_dataset
from data_utils import get_domain_list,domain_combined_data_loaders

def eval(config,configdl,src_domain_list,tgt_domain_list,net):

    #**********get data loader for validation get of all 3 set

    src_val_data_loader=domain_combined_data_loaders(config,configdl,src_domain_list,'test',net='mann_net',type='src')

    tgt_test_loader=domain_combined_data_loaders(config,configdl,tgt_domain_list,mode='test',net='mann_net',type='tgt')

    print("end")

    #*********write score to files

    #***********calc EER, threshold, hter and write to the file
    
    #************get test dataloader

    #************write score to files

    #**************cal  EER , HTER and write to file

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

    configdl_fname= 'src/configs/data_loader_dg.yaml'
    configdl= get_config(configdl_fname)

    # domain list
    source_domain_list,target_domain_list= get_domain_list(config,'mann_net')

    num_cls=2
    net = DgMannNet(config,num_cls)

    eval(config,configdl,source_domain_list,target_domain_list,net)
