import os
from os.path import join
from argparse import ArgumentParser
import torch
import torch.optim as optim
import numpy as np
from sklearn import metrics
import sys
sys.path.append("src")

from ocda_fas.eval import perf_measure
from ocda_fas.source.models.dg_resnet import DgEncoder

class DABaselineSrc():

    def __init__(self,config,configdl):

        self.config=config
        self.configdl=configdl
        self.device=self.config['device']
        #******* Network Initialization *************
        self.source_net = DgEncoder(self.config)

        #******* optmizer Initialization *************
        lr= self.config['da_baseline']['lr']
        weight_decay= self.config['da_baseline']['weight_decay']
        beta1= self.config['da_baseline']['beta1']
        beta2= self.config['da_baseline']['beta2']
        betas=(beta1,beta2)

        self.optim = optim.Adam(self.source_net.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        self.loss=torch.nn.CrossEntropyLoss()

    def train_epoch(self,src_train_loader,epoch,num_epoch):
        print("train_epoch enter")
        N = len(src_train_loader.dataset)
        batch_size=src_train_loader.batch_size
        print("train_epoch 2 enter")
        self.source_net.train()
        print("train_epoch 3 enter")
        print_each=10
        last_update = -1
        running_loss=0.0
        batch_idx=0
        src_train_iter=iter(src_train_loader)
        for (data,_,label) in src_train_loader:
            print("train_epoch loop enter")
            ########################
            # Setup data variables #
            ########################
            self.optim.zero_grad()

            self.source_net.to(self.device)
            data = data.to(self.device)
            label = label.to(self.device)

            _,logits,_= self.source_net(data)
            print("train_epoch fwd enter")
            loss= self.loss(logits,label)
            running_loss+=loss.item()

            loss.backward()
            self.optim.step()
            
            if (batch_idx+1)%print_each==0:
                print("Train: Epoch:{:d}/{:d} Data:{:d}/{:d} Loss:{:.5f} ".format(epoch,num_epoch,(batch_idx+1)*batch_size,N,running_loss/(print_each)))
                running_loss=0.0
            batch_idx+=1
            if self.config['debug']:
                break

        return 0

    def val(self,dataldr,type="val",eer_thr=None):

        self.source_net.eval()
        predict_lst=[]
        label_lst=[]
        live_prob_lst=[]
        with torch.no_grad():
            for batch_idx, (data,p, label) in enumerate(dataldr):
                
                self.source_net.to(self.device)
                data = data.to(self.device)
                label = label.to(self.device)

                _,logits,pred_prob= self.source_net(data)
                pred=torch.argmax(logits,dim=1)

                for i,_ in enumerate(pred_prob):
                    live_prob_lst.append(pred_prob[i,0].item())
                    label_lst.append(label[i].item())
                    predict_lst.append(pred[i].item())

                if self.config['debug']:
                    break
        
        if type=="val":
            fpr, tpr, thresholds = metrics.roc_curve(label_lst,live_prob_lst, pos_label=0)
            acc=metrics.accuracy_score(predict_lst,label_lst)
            fnr=1-tpr
            diff=np.absolute(fpr-fnr)
            idx=diff.argmin()
            eer_thr=thresholds[idx]
            hter=(fpr[idx]+fnr[idx])/2
            print("Val :: Acc:{:.3f} HTER :{:.3f} Threshold:{:.5f} idx:{:d} FAR:{:.3f} FRR:{:.3f}".format(acc,hter,eer_thr,idx,fpr[idx],fnr[idx]))
        elif type== 'test':
            predict_lst=live_prob_lst<eer_thr
            predict_lst = list(map(int, predict_lst))
            tp, fp, tn, fn=perf_measure(label_lst,predict_lst)
            fpr=fp/(tn+fp) # False rejection rate
            fnr=fn/(tp+fn) # false acceptance rate
            hter= (fpr+fnr)/2
            acc=(tp+tn)/(tp+fp+tn+fn)
            print("Test :: Acc:{:.3f} HTER :{:.3f} FAR:{:.3f} FRR:{:.3f} ".format(acc,hter,fpr,fnr))

        return hter,eer_thr

    def save(self,path):
        torch.save({
                'src_baseline_encoder': self.source_net.encoder.state_dict(),
                'src_baseline_cls': self.source_net.classifier.state_dict()},path)
    
    def load(self,checkpoint_file):
        self.source_net.load(checkpoint_file,'src_baseline_encoder','src_baseline_cls')