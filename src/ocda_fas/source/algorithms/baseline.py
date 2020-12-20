import os
from os.path import join
from argparse import ArgumentParser
import torch
import math
import torch.optim as optim
import numpy as np
from sklearn import metrics
import sys
import time
sys.path.append("src")
from networks import GradRevLayer
from ocda_fas.eval import perf_measure
from ocda_fas.source.models.dg_resnet2 import DgEncoder

class DABaselineSrc():

    def __init__(self,config,configdl,writer):

        self.config=config
        self.configdl=configdl
        self.device=self.config['device']
        # print("DABaselineSrc init enter",self.device)
        #******* Network Initialization *************
        self.source_net = DgEncoder(self.config)
        self.source_net.to(self.device)
        #******* optmizer Initialization *************
        lr= self.config['da_baseline']['src']['lr']
        weight_decay= self.config['da_baseline']['src']['weight_decay']
        beta1= self.config['da_baseline']['src']['beta1']
        beta2= self.config['da_baseline']['src']['beta2']
        betas=(beta1,beta2)

        self.optim = optim.Adam(self.source_net.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        self.criterion=torch.nn.CrossEntropyLoss()
        self.tensor_writer=writer

    def train_epoch(self,src_train_loader,src_val_loader,src_test_loader,epoch,num_epoch,best_hter_val):

        dataset_len = len(src_train_loader.dataset)
        dataloader_len = len(src_train_loader)
        batch_size=src_train_loader.batch_size
        print("train_epoch enter")
        self.source_net.train()

        print_each=1000
        last_update = -1
        running_loss=0.0
        batch_continue=math.ceil(dataloader_len/print_each)
        start = time.time()
        for batch_idx,(data,_,label) in enumerate(src_train_loader):
            ########################
            # Setup data variables #
            ########################
            # net_start = time.time()
            self.optim.zero_grad()
            print("print",self.device)
            data = data.to(self.device)
            label = label.to(self.device)

            _,logits,_= self.source_net(data)
            loss= self.criterion(logits,label)
            running_loss+=loss.item()

            loss.backward()
            self.optim.step()
            # net_end = time.time()
            # print("Network time:",net_end-net_start)
            
            if (batch_idx)%print_each==0 or ((batch_idx+1)==dataloader_len):
                loss_iter=running_loss/(print_each)
                print("Train: Epoch:{:d}/{:d} Data:{:d}/{:d} Loss:{:.5f} ".format(epoch,num_epoch,(batch_idx+1)*batch_size,dataset_len,loss_iter))
                running_loss=0.0
                # ******* val and test ******
                loss_val,acc_val,hter_val,eer_thr=self.val(src_val_loader,type='val')
                loss_test,acc_test,hter_test,_=self.val(src_test_loader,type='test',eer_thr=eer_thr)

                # ****** Writing to Tensorboard ******
                self.tensor_writer.add_scalar('Loss/train', loss_iter, (epoch*print_each*batch_continue)+batch_idx)
                self.tensor_writer.add_scalar('Loss/Val', loss_val, (epoch*print_each*batch_continue)+batch_idx)
                self.tensor_writer.add_scalar('Loss/test', loss_test, (epoch*print_each*batch_continue)+batch_idx)

                self.tensor_writer.add_scalar('Accuracy/Val', acc_val, (epoch*print_each*batch_continue)+batch_idx)
                self.tensor_writer.add_scalar('Accuracy/test', acc_test, (epoch*print_each*batch_continue)+batch_idx)

                self.tensor_writer.add_scalar('HTER/Val', hter_val, (epoch*print_each*batch_continue)+batch_idx)
                self.tensor_writer.add_scalar('HTER/test', hter_test, (epoch*print_each*batch_continue)+batch_idx)
                self.tensor_writer.add_scalar('HTER/EER-thrs', eer_thr, (epoch*print_each*batch_continue)+batch_idx)

                self.save(join(self.config['da_src_exp_path'],"src_baseline_checkpoint_last.pt"))
                print("*****Checkpoint saved*****")
                if hter_val<= best_hter_val:
                    self.save(join(self.config['da_src_exp_path'],"src_baseline_checkpoint_best.pt"))
                    best_hter_val=hter_val


            if self.config['debug']:
                break
            # end = time.time()
            # print("Total Time:",end-start)
            # start=time.time()
        end = time.time()
        print("Total Time for Epoch{}:{}".format(epoch,end-start))
        return best_hter_val

    def val(self,dataldr,type="val",eer_thr=None):

        self.source_net.eval()
        predict_lst=[]
        label_lst=[]
        live_prob_lst=[]
        total_loss=0.0
        with torch.no_grad():
            for batch_idx, (data,p, label) in enumerate(dataldr):
                
                self.source_net.to(self.device)
                data = data.to(self.device)
                label = label.to(self.device)

                _,logits,pred_prob= self.source_net(data)
                pred=torch.argmax(logits,dim=1)

                loss= self.criterion(logits,label)
                total_loss+=loss.item()

                for i,_ in enumerate(pred_prob):
                    live_prob_lst.append(pred_prob[i,0].item())
                    label_lst.append(label[i].item())
                    predict_lst.append(pred[i].item())

                if self.config['debug']:
                    break
        
        loss_avg=total_loss/len(dataldr)
        if type=="val":
            fpr_arr, tpr_arr, thresholds = metrics.roc_curve(label_lst,live_prob_lst, pos_label=0)
            acc=metrics.accuracy_score(predict_lst,label_lst)
            fnr_arr=1-tpr_arr
            diff=np.absolute(fpr_arr-fnr_arr)
            idx=diff.argmin()
            eer_thr=thresholds[idx]
            fpr=fpr_arr[idx]
            fnr=fnr_arr[idx]
            hter=(fpr+fnr)/2
            print("Val :: Acc:{:.3f} HTER :{:.3f} Threshold:{:.5f} idx:{:d} FAR:{:.3f} FRR:{:.3f}".format(acc,hter,eer_thr,idx,fpr,fnr))
            #write to tensorbpard
        elif type== 'test':
            predict_lst=live_prob_lst<eer_thr
            predict_lst = list(map(int, predict_lst))
            tp, fp, tn, fn=perf_measure(label_lst,predict_lst)
            fpr=fp/(tn+fp) # False rejection rate
            fnr=fn/(tp+fn) # false acceptance rate
            hter= (fpr+fnr)/2
            acc=(tp+tn)/(tp+fp+tn+fn)
            print("Test :: Acc:{:.3f} HTER :{:.3f} Threshold:{:.5f} FAR:{:.3f} FRR:{:.3f} ".format(acc,hter,eer_thr,fpr,fnr))
            #write to tensorbpard
        return loss_avg,acc,hter,eer_thr

    def save(self,path):
        torch.save({
                'src_baseline_encoder': self.source_net.encoder.state_dict(),
                'src_baseline_cls': self.source_net.classifier.state_dict()},path)


class DABaselineTgt():

    def __init__(self,config,configdl,writer):

        self.config=config
        self.configdl=configdl
        self.device=self.config['device']
        # print("DABaselineSrc init enter",self.device)
        #******* Network Initialization *************
        self.source_net = DgEncoder(self.config)
        self.source_net.to(self.device)
        self.source_net.load(self.config['da_src_checkpoint'],'src_baseline_encoder','src_baseline_cls')
        print("***** Weights of source Net initialized*****")

        self.tgt_net = DgEncoder(self.config)
        self.tgt_net.to(self.device)
        self.tgt_net.load(self.config['da_src_checkpoint'],'src_baseline_encoder','src_baseline_cls')
        print("***** Weights of target Net initialized*****")

        self.da_lambda=self.config['da_baseline']['tgt']['da_lambda']
        #******* optmizer Initialization *************
        lr= self.config['da_baseline']['tgt']['lr']
        weight_decay= self.config['da_baseline']['tgt']['weight_decay']
        beta1= self.config['da_baseline']['tgt']['beta1']
        beta2= self.config['da_baseline']['tgt']['beta2']
        betas=(beta1,beta2)

        self.optim_encoder = optim.Adam(self.tgt_net.encoder.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        self.optim_domain_cls = optim.Adam(self.tgt_net.classifier.parameters(), lr=lr*10, weight_decay=weight_decay, betas=betas)
        self.criterion=torch.nn.CrossEntropyLoss()
        self.tensor_writer=writer


    def train_epoch(self,src_train_loader,tgt_train_loader,tgt_val_loader,epoch,num_epoch,best_loss):
        
        dataset_len= min(len(src_train_loader.dataset), len(tgt_train_loader.dataset))
        dataloader_len = min(len(src_train_loader), len(tgt_train_loader)) 
        joint_loader = zip(src_train_loader, tgt_train_loader)

        batch_size=src_train_loader.batch_size
        print("train_epoch enter")
        self.source_net.train()
        print_each=2000
        batch_continue=math.ceil(dataloader_len/print_each)
        last_update = -1
        running_loss=0.0
        start = time.time()
        for batch_idx, ((data_s, _,_), (data_t,_, _)) in enumerate(joint_loader):
            ########################
            # Setup data variables #
            ########################
            # net_start = time.time()
            self.optim_encoder.zero_grad()
            self.optim_domain_cls.zero_grad()
            print("print",self.device)
            data_s = data_s.to(self.device)
            data_t = data_t.to(self.device)

            with torch.no_grad():
                X_s= self.source_net.encoder(data_s)

            X_t= self.tgt_net.encoder(data_t)
            grl_out = GradRevLayer.apply(X_t, self.da_lambda)

            X_comb = torch.cat((X_s, grl_out), 0)

            domain_logits=self.tgt_net.classifier(X_comb)

            # Domain labels Source =0 target =1

            target_dom_s = torch.zeros(len(data_s), requires_grad=False).long()
            target_dom_t = torch.ones(len(data_t), requires_grad=False).long()
            label_concat = torch.cat((target_dom_s, target_dom_t), 0).to(self.device)

            loss= self.criterion(domain_logits,label_concat)
            running_loss+=loss.item()

            loss.backward()
            self.optim_encoder.step()
            self.optim_domain_cls.step()
            # net_end = time.time()
            # print("Network time:",net_end-net_start)
            
            if (batch_idx)%print_each==0 or ((batch_idx+1)==dataloader_len):
                loss_iter=running_loss/(print_each)
                print("Train: Epoch:{:d}/{:d} Data:{:d}/{:d} Loss:{:.5f} ".format(epoch,num_epoch,(batch_idx+1)*batch_size,dataset_len,loss_iter))
                running_loss=0.0
                loss_val,acc_val=self.val(tgt_val_loader)
                self.tensor_writer.add_scalar('Loss/train', loss_iter, (epoch*print_each*batch_continue)+batch_idx)
                self.tensor_writer.add_scalar('Loss/Val', loss_val, (epoch*print_each*batch_continue)+batch_idx)
                self.tensor_writer.add_scalar('Accuracy/Val', acc_val, (epoch*print_each*batch_continue)+batch_idx)

                self.save(join(self.config['da_tgt_exp_path'],"tgt_baseline_checkpoint_last.pt"))
                print("*****Checkpoint saved*****")
                if loss_iter>= best_loss:
                    self.save(join(self.config['da_tgt_exp_path'],"tgt_baseline_checkpoint_best.pt"))
                    best_loss=loss_iter

            if self.config['debug']:
                break

        end = time.time()
        print("Total Time for Epoch{}:{}".format(epoch,end-start))
        return best_loss

    def val(self,dataldr):

        self.source_net.eval()
        predict_lst=[]
        label_lst=[]
        total_loss=0.0
        with torch.no_grad():
            for batch_idx, (data,_, _) in enumerate(dataldr):
                
                self.source_net.to(self.device)
                data = data.to(self.device)
                label = torch.ones(len(data), requires_grad=False).long()

                _,logits,pred_prob= self.tgt_net(data)

                loss= self.criterion(logits,label)
                total_loss+=loss.item()
                pred= torch.argmax(pred_prob,dim=1)
                for i,_ in enumerate(pred_prob):
                    predict_lst.append(pred[i].item())
                    label_lst.append(label[i].item())

                if self.config['debug']:
                    break
        
        loss_avg=total_loss/len(dataldr)
        acc=metrics.accuracy_score(predict_lst,label_lst)

        return loss_avg,acc

    def save(self,path):
        torch.save({
                'tgt_baseline_encoder': self.tgt_net.encoder.state_dict(),
                'tgt_baseline_domain_cls': self.tgt_net.classifier.state_dict()},path)