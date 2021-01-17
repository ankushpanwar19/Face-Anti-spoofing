import os
from os.path import join
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import torch.nn as nn
import math
import torch.optim as optim
import numpy as np
from sklearn import metrics
import sys
import time
sys.path.append("src")
from networks import GradRevLayer,ResNetClsNet
from ocda_fas.source.algorithms.eval import perf_measure
from ocda_fas.source.models.dg_resnet2 import DgEncoder
from schedulers import PolynomialLR
# from data_utils import MyDataParallel

class DABaselineSrc():

    def __init__(self,config,configdl,writer=None):

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
        if writer!= None:
            self.tensor_writer=writer

    def train_epoch(self,src_train_loader,src_val_loader,src_test_loader,epoch,num_epoch,best_hter_val):

        dataset_len = len(src_train_loader.dataset)
        dataloader_len = len(src_train_loader)
        batch_size=src_train_loader.batch_size
       
        self.source_net.train()

        eval_each= math.floor(dataloader_len/2)
        last_update = -1
        running_loss=0.0
        batch_continue=epoch*dataloader_len
    
        for batch_idx,(data,_,label) in enumerate(tqdm(src_train_loader)):
            ########################
            # Setup data variables #
            ########################
            self.optim.zero_grad()

            
            data = data.to(self.device)
            label = label.to(self.device)
            data.require_grad = False
            label.require_grad = False

            _,logits,_= self.source_net(data.clone())
            loss= self.criterion(logits,label.clone())
            running_loss+=loss.item()

            loss.backward()
            self.optim.step()
            if (batch_idx+1)%eval_each==0 or ((batch_idx+1)==dataloader_len):
                loss_iter=running_loss/(eval_each)
                print("Train: Epoch:{:d}/{:d} Data:{:d}/{:d} Loss:{:.5f} ".format(epoch,num_epoch,(batch_idx+1)*batch_size,dataset_len,loss_iter),flush=True)
                running_loss=0.0
                # ******* val and test ******
                loss_val,acc_val,hter_val,eer_thr=self.val(src_val_loader,type='val')
                loss_test,acc_test,hter_test,_=self.val(src_test_loader,type='test',eer_thr=eer_thr)

                # ****** Writing to Tensorboard ******
                self.tensor_writer.add_scalar('Loss/train', loss_iter,batch_continue+batch_idx)
                self.tensor_writer.add_scalar('Loss/Val', loss_val, batch_continue+batch_idx)
                self.tensor_writer.add_scalar('Loss/test', loss_test, batch_continue+batch_idx)

                self.tensor_writer.add_scalar('Accuracy/Val', acc_val, batch_continue+batch_idx)
                self.tensor_writer.add_scalar('Accuracy/test', acc_test, batch_continue+batch_idx)

                self.tensor_writer.add_scalar('HTER/Val', hter_val, batch_continue+batch_idx)
                self.tensor_writer.add_scalar('HTER/test', hter_test, batch_continue+batch_idx)
                self.tensor_writer.add_scalar('HTER/EER-thrs', eer_thr,batch_continue+batch_idx)

                self.save(join(self.config['da_src_exp_path'],"src_baseline_checkpoint_last.pt"))
                print("*****Checkpoint saved*****")
                if hter_val<= best_hter_val:
                    self.save(join(self.config['da_src_exp_path'],"src_baseline_checkpoint_best.pt"))
                    best_hter_val=hter_val

            
            if self.config['debug']:
                break

        #test after each epoch
        
            
        return best_hter_val


    def val(self,dataldr,type="val",eer_thr=None):

        self.source_net.eval()
        print_each=1000
        predict_lst=[]
        label_lst=[]
        live_prob_lst=[]
        total_loss=0.0
        batch_size=dataldr.batch_size
        dataset_len=len(dataldr.dataset)
        print("val_epoch enter")
        with torch.no_grad():
            for batch_idx, (data,p, label) in enumerate(tqdm(dataldr)):
                # print("BATCH print",batch_idx)
                data = data.to(self.device)
                label = label.to(self.device)
                data.require_grad = False
                label.require_grad = False

                _,logits,pred_prob= self.source_net(data.clone())
                pred=torch.argmax(logits,dim=1)

                loss= self.criterion(logits,label)
                total_loss+=loss.item()

                for i,_ in enumerate(pred_prob):
                    live_prob_lst.append(pred_prob[i,0].item())
                    label_lst.append(label[i].item())
                    predict_lst.append(pred[i].item())

                # if (batch_idx)%print_each==0 :
                #     print("Val:Data:{:d}/{:d}  ".format((batch_idx+1)*batch_size,dataset_len),flush=True)


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
            print("Val :: Acc:{:.3f} HTER :{:.3f} Threshold:{:.5f} idx:{:d} FAR:{:.3f} FRR:{:.3f}".format(acc,hter,eer_thr,idx,fpr,fnr),flush=True)
            #write to tensorbpard
        elif type== 'test':
            predict_lst=live_prob_lst<eer_thr
            predict_lst = list(map(int, predict_lst))
            tp, fp, tn, fn=perf_measure(label_lst,predict_lst)
            fpr=fp/(tn+fp) # False rejection rate
            fnr=fn/(tp+fn) # false acceptance rate
            hter= (fpr+fnr)/2
            acc=(tp+tn)/(tp+fp+tn+fn)
            print("Test :: Acc:{:.3f} HTER :{:.3f} Threshold:{:.5f} FAR:{:.3f} FRR:{:.3f} ".format(acc,hter,eer_thr,fpr,fnr),flush=True)
            #write to tensorbpard
        return loss_avg,acc,hter,eer_thr

    def save(self,path):
        torch.save({
                'src_baseline_encoder': self.source_net.encoder.state_dict(),
                'src_baseline_cls': self.source_net.classifier.state_dict()},path)
    
    def load(self,checkpoint):
        self.source_net.load(checkpoint,'src_baseline_encoder','src_baseline_cls')
        print("\n***Source Checkpoint loaded****\n")


class DABaselineTgt_ADDA():

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
        lr_encoder= self.config['da_baseline']['tgt']['lr_encoder']
        lr_discrim= self.config['da_baseline']['tgt']['lr_discrim']
        weight_decay= self.config['da_baseline']['tgt']['weight_decay']
        beta1= self.config['da_baseline']['tgt']['beta1']
        beta2= self.config['da_baseline']['tgt']['beta2']
        betas=(beta1,beta2)

        self.optim_encoder = optim.Adam(self.tgt_net.encoder.parameters(), lr=lr_encoder, weight_decay=weight_decay, betas=betas)
        self.optim_domain_cls = optim.Adam(self.tgt_net.classifier.parameters(), lr=lr_discrim, weight_decay=weight_decay, betas=betas)
        self.criterion=torch.nn.CrossEntropyLoss()
        self.tensor_writer=writer


    def train_epoch(self,src_train_loader,tgt_train_loader,tgt_val_loader,epoch,num_epoch,best_loss):
        
        dataset_len= min(len(src_train_loader.dataset), len(tgt_train_loader.dataset))
        dataloader_len = min(len(src_train_loader), len(tgt_train_loader)) 
        joint_loader = zip(src_train_loader, tgt_train_loader)

        batch_size=src_train_loader.batch_size
        print("train_epoch enter")
        self.source_net.eval()
        self.tgt_net.train()
        print_each=4000
        batch_continue=epoch*dataloader_len
        running_loss=0.0
        gan_running_loss=0.0
        update_encoder=False
        update_dicriminator=True
        # start = time.time()
        predict_lst=[]
        label_lst=[]
        for batch_idx, ((data_s, _,_), (data_t,_, _)) in enumerate(joint_loader):
            ########################
            # Setup data variables #
            ########################
            # net_start = time.time()
            self.optim_encoder.zero_grad()
            self.optim_domain_cls.zero_grad()
            # print("print",self.device)
            data_s = data_s.to(self.device)
            data_t = data_t.to(self.device)
            data_s.require_grad = False
            data_t.require_grad = False

            with torch.no_grad():
                X_s= self.source_net.encoder(data_s.clone())

            X_t= self.tgt_net.encoder(data_t.clone())
            # grl_out = GradRevLayer.apply(X_t, self.da_lambda)

            X_comb = torch.cat((X_s, X_t), 0)
            grl_out = GradRevLayer.apply(X_comb, self.da_lambda)
            domain_logits=self.tgt_net.classifier(grl_out)

            # Domain labels Source =0 target =1

            target_dom_s = torch.zeros(len(data_s), requires_grad=False).long()
            target_dom_t = torch.ones(len(data_t), requires_grad=False).long()
            label_concat = torch.cat((target_dom_s, target_dom_t), 0).to(self.device)

            loss= self.criterion(domain_logits,label_concat)
            running_loss+=loss.item()
            gan_running_loss+=loss.item()
            loss.backward()
            if update_encoder:
                self.optim_encoder.step()
            if  update_dicriminator:
                self.optim_domain_cls.step()

            pred= torch.argmax(domain_logits,dim=1)
            for i,_ in enumerate(domain_logits):
                    predict_lst.append(pred[i].item())
                    label_lst.append(label_concat[i].item())
            
            if (batch_idx+1)%100==0 or ((batch_idx+1)==dataloader_len):
                acc_train=metrics.accuracy_score(predict_lst,label_lst)
                gan_loss=gan_running_loss/100
                print("Train: Epoch:{:d}/{:d} Data:{:d}/{:d} DomainAcc:{:.4f} Encoder:{:b} Discriminator:{:b}".format(epoch,num_epoch,(batch_idx+1)*batch_size,dataset_len,acc_train,update_encoder,update_dicriminator),flush=True)
                self.tensor_writer.add_scalar('GAN_Train/loss', gan_loss, (epoch*dataloader_len)+batch_idx)
                self.tensor_writer.add_scalar('GAN_Train/acc', acc_train, (epoch*dataloader_len)+batch_idx)
                
                #reinitialize
                predict_lst=[]
                label_lst=[]
                gan_running_loss=0.0

                if acc_train>0.6:
                    update_encoder=True
                    update_dicriminator=False
                else:
                    update_encoder=False
                    update_dicriminator=True
                    

            if (batch_idx+1)%print_each==0 or ((batch_idx+1)==dataloader_len) or self.config['debug']:
                loss_iter=running_loss/(print_each)
                print("Train: Epoch:{:d}/{:d} Data:{:d}/{:d} Loss:{:.5f} ".format(epoch,num_epoch,(batch_idx+1)*batch_size,dataset_len,loss_iter),flush=True)
                running_loss=0.0
                loss_val,acc_val=self.val(tgt_val_loader)
                self.tensor_writer.add_scalar('Loss/train', loss_iter, (epoch*dataloader_len)+batch_idx)
                self.tensor_writer.add_scalar('Loss/Val', loss_val, (epoch*dataloader_len)+batch_idx)
                self.tensor_writer.add_scalar('Accuracy/Val', acc_val, (epoch*dataloader_len)+batch_idx)

                self.save(join(self.config['da_tgt_exp_path'],"tgt_baseline_checkpoint_last.pt"))
                print("*****Checkpoint saved*****")
                if loss_val>= best_loss:
                    self.save(join(self.config['da_tgt_exp_path'],"tgt_baseline_checkpoint_best.pt"))
                    best_loss=loss_val

            if self.config['debug']:
                break

        # end = time.time()
        # print("Total Time for Epoch{}:{}".format(epoch,end-start))
        return best_loss

    def val(self,dataldr):

        self.tgt_net.eval()
        predict_lst=[]
        label_lst=[]
        total_loss=0.0
        with torch.no_grad():
            for batch_idx, (data,_, _) in enumerate(tqdm(dataldr)):
                
                data = data.to(self.device)
                data.require_grad = False
                label = torch.ones(len(data.clone()), requires_grad=False).long()
                label = label.to(self.device)
                _,logits,pred_prob= self.tgt_net(data.clone())

                loss= self.criterion(logits,label)
                total_loss+=loss.item()
                pred= torch.argmax(pred_prob,dim=1)
                for i,_ in enumerate(pred_prob):
                    predict_lst.append(pred[i].item())
                    label_lst.append(label[i].item())
        
        loss_avg=total_loss/len(dataldr)
        acc=metrics.accuracy_score(predict_lst,label_lst)

        print("Val :: LOSS:{:.3f} ACC :{:.3f}".format(loss_avg,acc),flush=True)

        return loss_avg,acc

    def save(self,path):
        torch.save({
                'tgt_baseline_encoder': self.tgt_net.encoder.state_dict(),
                'tgt_baseline_domain_cls': self.tgt_net.classifier.state_dict()},path)
    def load(self,checkpoint):
        self.source_net.load(checkpoint,'src_baseline_encoder','src_baseline_cls')
        print("\n***Source Net Intialized with src checkpoint****\n")
        self.tgt_net.load(checkpoint,'src_baseline_encoder','src_baseline_cls')
        print("\n***Target Net Intialized with src checkpoint****\n")
        

class DABaselineTgt_GRL():

    def __init__(self,config,configdl,writer):

        self.config=config
        self.configdl=configdl
        self.device=self.config['device']
        # print("DABaselineSrc init enter",self.device)
        #******* Network Initialization *************

        self.tgt_net = DgEncoder(self.config)
        config_resnet_clsnet = config['resent_clsnet']
        self.discriminator = ResNetClsNet(config_resnet_clsnet, config['debug'],'ResNetClsNet')

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.tgt_net = nn.DataParallel(self.tgt_net)
            self.discriminator = nn.DataParallel(self.discriminator)
        
        self.tgt_net.to(self.device)
        self.discriminator.to(self.device)


        self.da_lambda=self.config['da_baseline']['tgt']['da_lambda']
        self.da_lambda_type=self.config['da_baseline']['tgt']['da_lambda_type']
        self.dom_loss_wt=self.config['da_baseline']['tgt']['dom_loss_wt']
        #******* optmizer Initialization *************
        lr_encoder= self.config['da_baseline']['tgt']['lr_encoder']
        lr_discrim= self.config['da_baseline']['tgt']['lr_discrim']
        weight_decay= self.config['da_baseline']['tgt']['weight_decay']
        beta1= self.config['da_baseline']['tgt']['beta1']
        beta2= self.config['da_baseline']['tgt']['beta2']
        betas=(beta1,beta2)

        self.optim_net = optim.Adam(self.tgt_net.parameters(), lr=lr_encoder, weight_decay=weight_decay, betas=betas)
        self.optim_discrim = optim.Adam(self.discriminator.parameters(), lr=lr_discrim, weight_decay=weight_decay, betas=betas)

        decay_iter= self.config['da_baseline']['tgt']['lr_scheduler']['decay_iter']
        gamma= self.config['da_baseline']['tgt']['lr_scheduler']['gamma']
        numepochs=config['da_baseline']['tgt']['epochs']
        self.scheduler_net= PolynomialLR(self.optim_net,numepochs,decay_iter,gamma)
        self.scheduler_discrim=PolynomialLR(self.optim_discrim,numepochs,decay_iter,gamma)

        self.criterion=torch.nn.CrossEntropyLoss()
        self.tensor_writer=writer


    def train_epoch(self,src_train_loader,tgt_train_loader,tgt_val_loader,epoch,num_epoch):
        
        dataset_len= min(len(src_train_loader.dataset), len(tgt_train_loader.dataset))
        dataloader_len = min(len(src_train_loader), len(tgt_train_loader)) 
        joint_loader = zip(src_train_loader, tgt_train_loader)

        batch_size=src_train_loader.batch_size
        print("train_epoch enter")
        self.tgt_net.train()
        
        print_each=200
        if self.config['debug']:
            print_each=1
        running_loss_cls=0.0
        running_loss_discrim=0.0
        predict_src_lst=[]
        label_src_lst=[]
        predict_domain=[]
        label_domain=[]

        print("Learning Rate:",self.optim_net.param_groups[0]['lr'])
        for batch_idx, ((data_s, _,label_s), (data_t,_, _)) in enumerate(joint_loader):
            ########################
            # Setup data variables #
            ########################
            # net_start = time.time()
            self.optim_net.zero_grad()
            self.optim_discrim.zero_grad()
            # print("print",self.device)
            data_s = data_s.to(self.device)
            data_t = data_t.to(self.device)
            label_s = label_s.to(self.device)
            data_s.require_grad = False
            data_t.require_grad = False
            label_s.require_grad = False

            #*****clasfication branch****
            X_s,logits_s,_= self.tgt_net(data_s.clone())
 
            loss_cls= self.criterion(logits_s,label_s.clone())

            running_loss_cls+=loss_cls.item()

            #******Domian Discrimator branch******
            if torch.cuda.device_count() > 1:
                
                X_t= self.tgt_net.module.encoder(data_t.clone())
            else:
                X_t= self.tgt_net.encoder(data_t.clone())

            X_comb = torch.cat((X_s, X_t), 0)

            # print("Feature shape",X_comb.shape)
            # Setting GRL lambda
            if self.da_lambda_type=='vary':
                p = float(batch_idx + epoch * dataloader_len) / (num_epoch * dataloader_len)
                self.da_lambda=2. / (1. + np.exp(-10 * p)) - 1

            grl_out = GradRevLayer.apply(X_comb, self.da_lambda)
            domain_logits=self.discriminator(grl_out)

            # print("out shape",domain_logits.shape)
            # Domain labels Source =0 target =1
            target_dom_s = torch.zeros(len(data_s), requires_grad=False).long()
            target_dom_t = torch.ones(len(data_t), requires_grad=False).long()
            label_concat = torch.cat((target_dom_s, target_dom_t), 0).to(self.device)

            loss_discrim= self.criterion(domain_logits,label_concat)
            running_loss_discrim+=loss_discrim.item()


            loss_overall=loss_cls+ self.dom_loss_wt*loss_discrim
            loss_overall.backward()
            
            # update weights
            self.optim_net.step()
            self.optim_discrim.step()


            pred_src= torch.argmax(logits_s,dim=1)
            pred_dom= torch.argmax(domain_logits,dim=1)
            for i,_ in enumerate(logits_s):
                    predict_src_lst.append(pred_src[i].item())
                    label_src_lst.append(label_s[i].item())
                    predict_domain.append(pred_dom[i].item())
                    label_domain.append(label_concat[i].item())

            
            if (batch_idx+1)%print_each==0 or ((batch_idx+1)==dataloader_len):
                acc_src_train=metrics.accuracy_score(predict_src_lst,label_src_lst)
                acc_dom=metrics.accuracy_score(predict_domain,label_domain)
                loss_cls_avg=running_loss_cls/print_each
                loss_discrim_avg=running_loss_discrim/print_each
                print("\nTrain: Epoch:{:d}/{:d} Data:{:d}/{:d} Cls_Src_Acc:{:.4f} ClsLoss:{:4f} Dom_Acc:{:.4f} DiscrimLoss:{:4f}".format(epoch,num_epoch,(batch_idx+1)*batch_size,dataset_len,acc_src_train,loss_cls_avg,acc_dom,loss_discrim_avg),flush=True)
                self.tensor_writer.add_scalar('Training/Cls_src_loss', loss_cls_avg, (epoch*dataloader_len)+batch_idx)
                self.tensor_writer.add_scalar('Training/Discrim_loss', loss_discrim_avg, (epoch*dataloader_len)+batch_idx)
                self.tensor_writer.add_scalar('Training/Acc_cls_src', acc_src_train, (epoch*dataloader_len)+batch_idx)
                self.tensor_writer.add_scalar('Training/Acc_dom', acc_dom, (epoch*dataloader_len)+batch_idx)
                
                #reinitialize
                predict_src_lst=[]
                label_src_lst=[]
                running_loss_cls=0.0
                running_loss_discrim=0.0

            if self.config['debug']:
                break

        return 0

    def val(self,dataldr,domain,epoch,type="val",eer_thr=None):

        self.tgt_net.eval()
        print_each=1000
        predict_lst=[]
        label_lst=[]
        live_prob_lst=[]
        total_loss=0.0
        batch_size=dataldr.batch_size
        dataset_len=len(dataldr.dataset)
        print("val_epoch enter")
        val_file=join(self.config['da_tgt_exp_path'],(domain+"_"+type+"_score_" +str(epoch+1)+".txt"))
        f=open(val_file, 'w')
        with torch.no_grad():
            for batch_idx, (data,path, label) in enumerate(tqdm(dataldr)):
                # print("BATCH print",batch_idx)
                data = data.to(self.device)
                label = label.to(self.device)
                data.require_grad = False
                label.require_grad = False

                _,logits,pred_prob= self.tgt_net(data.clone())
                pred=torch.argmax(logits,dim=1)

                loss= self.criterion(logits,label.clone())
                total_loss+=loss.item()

                for i,_ in enumerate(pred_prob):
                    live_prob_lst.append(pred_prob[i,0].item())
                    label_lst.append(label[i].item())
                    predict_lst.append(pred[i].item())
                    f.write("{:.5f},{:d},{:s}\n".format(pred_prob[i,0].item(),label[i].item(),path[i]))

                # if (batch_idx)%print_each==0 :
                #     print("Val:Data:{:d}/{:d}  ".format((batch_idx+1)*batch_size,dataset_len),flush=True)

                if self.config['debug']:
                    break
        f.close()
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
            print("\nVal_{:s} :: Acc:{:.3f} HTER :{:.3f} Threshold:{:.5f} idx:{:d} FAR:{:.3f} FRR:{:.3f}".format(domain,acc,hter,eer_thr,idx,fpr,fnr),flush=True)
            #write to tensorbpard
            if domain=='src':
                self.tensor_writer.add_scalar('Eval_src/Val_Acc', acc, epoch+1)
                self.tensor_writer.add_scalar('Eval_src/Val_eer_thr', eer_thr, epoch+1)
                self.tensor_writer.add_scalar('Eval_src/Val_Hter', hter, epoch+1)
            elif domain=='tgt':
                self.tensor_writer.add_scalar('Eval_tgt/Val_Acc', acc, epoch+1)
                self.tensor_writer.add_scalar('Eval_tgt/Val_eer_thr', eer_thr, epoch+1)
                self.tensor_writer.add_scalar('Eval_tgt/Val_Hter', hter, epoch+1)

        elif type== 'test':
            predict_lst=live_prob_lst<eer_thr
            predict_lst = list(map(int, predict_lst))
            tp, fp, tn, fn=perf_measure(label_lst,predict_lst)
            fpr=fp/(tn+fp) # False rejection rate
            fnr=fn/(tp+fn) # false acceptance rate
            hter= (fpr+fnr)/2
            acc=(tp+tn)/(tp+fp+tn+fn)
            print("\nTest_{:s} :: Acc:{:.3f} HTER :{:.3f} Threshold:{:.5f} FAR:{:.3f} FRR:{:.3f} ".format(domain,acc,hter,eer_thr,fpr,fnr),flush=True)
            #write to tensorbpard
            if domain=='src':
                self.tensor_writer.add_scalar('Eval_src/test_Acc', acc, epoch+1)
                self.tensor_writer.add_scalar('Eval_src/test_eer_thr', eer_thr, epoch+1)
                self.tensor_writer.add_scalar('Eval_src/test_Hter', hter, epoch+1)
            elif domain=='tgt':
                self.tensor_writer.add_scalar('Eval_tgt/test_Acc', acc, epoch+1)
                self.tensor_writer.add_scalar('Eval_tgt/test_eer_thr', eer_thr, epoch+1)
                self.tensor_writer.add_scalar('Eval_tgt/test_Hter', hter, epoch+1)

        return loss_avg,acc,hter,eer_thr

    def save(self,path):

        if torch.cuda.device_count() > 1:
            torch.save({
                    'encoder': self.tgt_net.module.encoder.state_dict(),
                    'classifier': self.tgt_net.module.classifier.state_dict(),
                    'dicriminator':self.discriminator.state_dict()},path)
        else:
            torch.save({
                    'encoder': self.tgt_net.encoder.state_dict(),
                    'classifier': self.tgt_net.classifier.state_dict(),
                    'dicriminator':self.discriminator.state_dict()},path)

    def load(self,checkpoint):
        self.tgt_net.load(checkpoint,'encoder','classifier')
        print("\n***Target Net Intialized with src checkpoint****\n")


class SrcTgtDist():

    def __init__(self,config,configdl,writer):

        self.config=config
        self.configdl=configdl
        self.device=self.config['device']
        #******* Network Initialization *************

        self.tgt_net = DgEncoder(self.config)
        config_resnet_clsnet = config['resent_clsnet']
        self.discriminator = ResNetClsNet(config_resnet_clsnet, config['debug'],'ResNetClsNet')

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.tgt_net = nn.DataParallel(self.tgt_net)
            self.discriminator = nn.DataParallel(self.discriminator)
        
        self.tgt_net.to(self.device)
        self.discriminator.to(self.device)


        self.da_lambda=self.config['da_baseline']['tgt']['da_lambda']
        self.da_lambda_type=self.config['da_baseline']['tgt']['da_lambda_type']
        self.dom_loss_wt=self.config['da_baseline']['tgt']['dom_loss_wt']
        #******* optmizer Initialization *************
        lr_encoder= self.config['da_baseline']['tgt']['lr_encoder']
        lr_discrim= self.config['da_baseline']['tgt']['lr_discrim']
        weight_decay= self.config['da_baseline']['tgt']['weight_decay']
        beta1= self.config['da_baseline']['tgt']['beta1']
        beta2= self.config['da_baseline']['tgt']['beta2']
        betas=(beta1,beta2)

        self.optim_net = optim.Adam(self.tgt_net.parameters(), lr=lr_encoder, weight_decay=weight_decay, betas=betas)
        self.optim_discrim = optim.Adam(self.discriminator.parameters(), lr=lr_discrim, weight_decay=weight_decay, betas=betas)

        decay_iter= self.config['da_baseline']['tgt']['lr_scheduler']['decay_iter']
        gamma= self.config['da_baseline']['tgt']['lr_scheduler']['gamma']
        numepochs=config['da_baseline']['tgt']['epochs']
        self.scheduler_net= PolynomialLR(self.optim_net,numepochs,decay_iter,gamma)
        self.scheduler_discrim=PolynomialLR(self.optim_net,numepochs,decay_iter,gamma)

        self.criterion=torch.nn.CrossEntropyLoss()
        self.tensor_writer=writer


    def train_epoch(self,src_train_loader,tgt_train_loader,epoch,num_epoch):
        
        dataset_len= min(len(src_train_loader.dataset), len(tgt_train_loader.dataset))
        dataloader_len = min(len(src_train_loader), len(tgt_train_loader)) 
        joint_loader = zip(src_train_loader, tgt_train_loader)

        batch_size=src_train_loader.batch_size
        print("train_epoch enter")
        self.tgt_net.train()
        
        print_each=200
        if self.config['debug']:
            print_each=1
        running_loss_cls=0.0
        running_loss_discrim=0.0
        predict_src_lst=[]
        label_src_lst=[]
        predict_domain=[]
        label_domain=[]

        print("Learning Rate:",self.optim_net.param_groups[0]['lr'])
        for batch_idx, ((data_s, _,label_s), (data_t,_, _)) in enumerate(joint_loader):
            ########################
            # Setup data variables #
            ########################
            # net_start = time.time()
            self.optim_net.zero_grad()
            self.optim_discrim.zero_grad()
            # print("print",self.device)
            data_s = data_s.to(self.device)
            data_t = data_t.to(self.device)
            label_s = label_s.to(self.device)
            data_s.require_grad = False
            data_t.require_grad = False
            label_s.require_grad = False

            #*****clasfication branch****
            X_s,logits_s,_= self.tgt_net(data_s.clone())
 
            loss_cls= self.criterion(logits_s,label_s.clone())

            running_loss_cls+=loss_cls.item()

            #******Domian Discrimator branch******
            if torch.cuda.device_count() > 1:
                
                X_t= self.tgt_net.module.encoder(data_t.clone())
            else:
                X_t= self.tgt_net.encoder(data_t.clone())

            if sum(label_s==0).item()>0:
                X_s_live=X_s[label_s==0]
                X_comb = torch.cat((X_s, X_t), 0)
            else:
                X_comb = X_t

            # print("Feature shape",X_comb.shape)
            # Setting GRL lambda
            if self.da_lambda_type=='vary':
                p = float(batch_idx + epoch * dataloader_len) / (num_epoch * dataloader_len)
                self.da_lambda=2. / (1. + np.exp(-10 * p)) - 1

            grl_out = GradRevLayer.apply(X_comb, self.da_lambda)
            domain_logits=self.discriminator(grl_out)

            # print("out shape",domain_logits.shape)
            # Domain labels Source =0 target =1
            target_dom_t = torch.ones(len(data_t), requires_grad=False).long()
            if sum(label_s==0).item()>0:
                target_dom_s = torch.zeros(len(data_s), requires_grad=False).long()
                label_concat = torch.cat((target_dom_s, target_dom_t), 0).to(self.device)
            else:
                label_concat=target_dom_t.to(self.device)

            loss_discrim= self.criterion(domain_logits,label_concat)
            running_loss_discrim+=loss_discrim.item()


            loss_overall=loss_cls+ self.dom_loss_wt*loss_discrim
            loss_overall.backward()
            
            # update weights
            self.optim_net.step()
            self.optim_discrim.step()


            pred_src= torch.argmax(logits_s,dim=1)
            pred_dom= torch.argmax(domain_logits,dim=1)
            for i,_ in enumerate(logits_s):
                    predict_src_lst.append(pred_src[i].item())
                    label_src_lst.append(label_s[i].item())
                    predict_domain.append(pred_dom[i].item())
                    label_domain.append(label_concat[i].item())

            
            if (batch_idx+1)%print_each==0: #or ((batch_idx+1)==dataloader_len):
                acc_src_train=metrics.accuracy_score(predict_src_lst,label_src_lst)
                acc_dom=metrics.accuracy_score(predict_domain,label_domain)
                loss_cls_avg=running_loss_cls/print_each
                loss_discrim_avg=running_loss_discrim/print_each
                print("\nTrain: Epoch:{:d}/{:d} Data:{:d}/{:d} Cls_Src_Acc:{:.4f} ClsLoss:{:4f} Dom_Acc:{:.4f} DiscrimLoss:{:4f}".format(epoch,num_epoch,(batch_idx+1)*batch_size,dataset_len,acc_src_train,loss_cls_avg,acc_dom,loss_discrim_avg),flush=True)
                self.tensor_writer.add_scalar('Training/Cls_src_loss', loss_cls_avg, (epoch*dataloader_len)+batch_idx)
                self.tensor_writer.add_scalar('Training/Discrim_loss', loss_discrim_avg, (epoch*dataloader_len)+batch_idx)
                self.tensor_writer.add_scalar('Training/Acc_cls_src', acc_src_train, (epoch*dataloader_len)+batch_idx)
                self.tensor_writer.add_scalar('Training/Acc_dom', acc_dom, (epoch*dataloader_len)+batch_idx)
                
                #reinitialize
                predict_src_lst=[]
                label_src_lst=[]
                running_loss_cls=0.0
                running_loss_discrim=0.0

            if self.config['debug']:
                break

        return 0

    def val(self,dataldr,domain,epoch,type="val"):

        self.tgt_net.eval()
        print_each=1000
        predict_lst=[]
        label_lst=[]
        live_prob_lst=[]
        total_loss=0.0
        batch_size=dataldr.batch_size
        dataset_len=len(dataldr.dataset)
        print("val_epoch enter")
        mean_vec=torch.zeros(2048)
        count_live=0.0
        with torch.no_grad():
            for batch_idx, (data,path, label) in enumerate(tqdm(dataldr)):
                # print("BATCH print",batch_idx)

                data = data.to(self.device)
                label = label.to(self.device)
                data.require_grad = False
                label.require_grad = False

                x,logits,pred_prob= self.tgt_net(data.clone())
                pred=torch.argmax(logits,dim=1)

                for i,_ in enumerate(pred_prob):
                    # live_prob_lst.append(pred_prob[i,0].item())
                    label_lst.append(label[i].item())
                    predict_lst.append(pred[i].item())

                if domain=='src':
                    if sum(label==0).item()>0:
                        x=x[label==0]
                        label=label[label==0]
                    else:
                        continue
                    
                batch_sum=torch.sum(x,dim=0)
                mean_vec += batch_sum.cpu()
                count_live += label.shape[0]

                if self.config['debug']:
                    break

             
        acc=metrics.accuracy_score(predict_lst,label_lst)
        metric=domain+"_"+"accuracy"
        self.tensor_writer.add_scalar('Eval_src/'+metric, acc, epoch+1)
        mean_vec=mean_vec/count_live 

        return mean_vec

    def vec_dist(self,m1,m2,epoch,norm_type=2.0):
        m1=m1.unsqueeze(0)/m1.norm()
        m2=m2.unsqueeze(0)/m2.norm()
        dist=torch.cdist(m1,m2,p=norm_type)

        self.tensor_writer.add_scalar('Eval/Distance', dist.item(), epoch+1)
        return dist.item()

    def save(self,path):

        if torch.cuda.device_count() > 1:
            torch.save({
                    'encoder': self.tgt_net.module.encoder.state_dict(),
                    'classifier': self.tgt_net.module.classifier.state_dict(),
                    'dicriminator':self.discriminator.state_dict()},path)
        else:
            torch.save({
                    'encoder': self.tgt_net.encoder.state_dict(),
                    'classifier': self.tgt_net.classifier.state_dict(),
                    'dicriminator':self.discriminator.state_dict()},path)

    def load(self,checkpoint):
        self.tgt_net.load(checkpoint,'encoder','classifier')
        print("\n***Target Net Intialized with src checkpoint****\n")
        


