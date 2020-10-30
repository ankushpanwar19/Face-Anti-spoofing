import torch
import torch.nn as nn
from utils import weights_init, get_model_list
import os
from pytorch_resnet import ResNet, Bottleneck
import torch.utils.model_zoo as model_zoo
from utils import new_dictionary
# from networks_new import GradRevLayer, ResNetClsNet, ClsCondDomainNet, ClsCondDomainNetHeads, Cls_lstm # this with new classification layer and for CrossEntropy Loss
from networks import GradRevLayer, ResNetClsNet, ClsCondDomainNet, ClsCondDomainNetHeads, Cls_lstm  # this with old classification layer and for NLLLoss
from os.path import join
from schedulers import get_scheduler
import torch.nn.functional as F

class Trainer_ResNetMOTDG(nn.Module): ## the trainer used for current train_DA.py for ResNet + LSTM
    def __init__(self, config):
        super(Trainer_ResNetMOTDG, self).__init__()
        print('--- trainer.py --> class Trainer_ResNetMOT --> __init()__ ---')
        machine = config['machine']
        self.debug = config['debug']
        self.device = config['device']
        # num_classes = 2
        resnet_config = config #['resnet']
        self.resnet_arc = config['resnet_arch']
        # self.model_dir = config['resnet_model_file']['file3_machine{}'.format(machine)] #* where to get this model

        if 'lstmmot' in config['net_type']:
            self.is_lstm = True
        else:
            self.is_lstm = False

        if not config['option'] or (not 'option' in config.keys()):
            self.option = 0
        elif config['option'] == 1:
            self.option = 1

        ## Resnet models: 
        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        }


        self.gen = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)  # 8631
        self.gen.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

        self.config_resnet_clsnet = config['resent_clsnet']
        self.gen.fc = ResNetClsNet(self.config_resnet_clsnet, self.debug, 'ResNetClsNet')

        if config['net_type'] == 'lstmmot':
            self.lstm = Cls_lstm(config['batch_size_lstm'])

        self.cnn_cond_dom_head = ClsCondDomainNet(config['ClsCondDomainNet'], self.debug, 'ClsCondDomainNet')
        self.cnn_cond_dom_tail_live = ClsCondDomainNetHeads(config['ClsCondDomainNetHeads'], self.debug, 'ClsCondDomainNetHeadsLive')
        self.cnn_cond_dom_tail_spoof = ClsCondDomainNetHeads(config['ClsCondDomainNetHeads'], self.debug, 'ClsCondDomainNetHeadsSpoof')

        self.lstm_cond_dom_head = ClsCondDomainNet(config['ClsCondDomainNet'], self.debug, 'ClsCondDomainNet')
        self.lstm_cond_dom_tail_live = ClsCondDomainNetHeads(config['ClsCondDomainNetHeads'], self.debug, 'lstmDomainNetHeadsLive')
        self.lstm_cond_dom_tail_spoof = ClsCondDomainNetHeads(config['ClsCondDomainNetHeads'], self.debug, 'lstmDomainNetHeadsSpoof')

        net_params = list(self.gen.parameters()) + \
                     list(self.lstm.parameters()) + \
                     list(self.cnn_cond_dom_head.parameters()) + \
                     list(self.cnn_cond_dom_tail_live.parameters()) + \
                     list(self.cnn_cond_dom_tail_spoof.parameters()) + \
                     list(self.lstm_cond_dom_head.parameters()) + \
                     list(self.lstm_cond_dom_tail_live.parameters()) + \
                     list(self.lstm_cond_dom_tail_spoof.parameters())

        self.nll_loss = nn.NLLLoss()

        lr = resnet_config['optim']['lr']
        optim_type = resnet_config['optim']['optim_type']
        beta1 = resnet_config['optim']['beta1']
        beta2 = resnet_config['optim']['beta2']
        weight_decay = resnet_config['optim']['weight_decay']
        momentum = resnet_config['optim']['momentum']

        if optim_type == 'sgd':
            print('>>> Using SGD <<<')
            self.net_opt = torch.optim.SGD([p for p in net_params if p.requires_grad], lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optim_type == 'adam':
            print('>>> Using Adam <<<')
            self.net_opt = torch.optim.Adam([p for p in net_params if p.requires_grad], lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        else:
            pass

        scheduler_config = config['scheduler']
        print(scheduler_config)
        print(scheduler_config['parameters'])
        scheduler_cls = get_scheduler(scheduler_config['lr_policy'])

        if scheduler_config['lr_policy'] == 'PolynomialLR':
            self.net_scheduler = scheduler_cls(self.net_opt, **scheduler_config['parameters'])
        else:
            self.net_scheduler = scheduler_cls(self.net_opt)


    def cls_criterion(self, cnet_out, clslabels):
        cls_loss = self.nll_loss(cnet_out, clslabels)
        return cls_loss



    def net_update_cnn(self, imgBatch, gtClsLabels, da_lambda, batch_size_cnn, idxLive, liveDomainGTLables, idxSpoof, spoofDomainGTLables):
        self.net_opt.zero_grad()
        if self.debug:
            print(' --- trainer.py --> class Trainer_ResNetDG --> net_update() --- ')

        resnet_out = self.gen(imgBatch) # --- 48*2048
        resnet_clsnet_out = self.gen.fc(resnet_out) # --- 48*2

        grl_out = GradRevLayer.apply(resnet_out, da_lambda)

        clsCondOut = self.cnn_cond_dom_head(grl_out) # --- 48*1024
        clsCondLiveOut = self.cnn_cond_dom_tail_live(clsCondOut.index_select(0, idxLive.squeeze())) # --- a*3
        clsCondSpoofOut = self.lstm_cond_dom_tail_spoof(clsCondOut.index_select(0, idxSpoof.squeeze())) # --- (48-a)*3

        self.loss_cnn_cls = self.cls_criterion(resnet_clsnet_out, gtClsLabels) # --- gtClslabels: 48
    
        if liveDomainGTLables.size(0) == 0:
            self.loss_cnnLive = torch.tensor(0).to(self.device)
        else:
            self.loss_cnnLive = self.cls_criterion(clsCondLiveOut, liveDomainGTLables) # --- livedomaingtlabels: a
        
        if spoofDomainGTLables.size(0) == 0:
            self.loss_cnnSpoof = torch.tensor(0).to(self.device)
        else:
            self.loss_cnnSpoof = self.cls_criterion(clsCondSpoofOut, spoofDomainGTLables) # --- spoofDomainGTLables : 48-a
    
        self.loss_total = self.loss_cnn_cls + self.loss_cnnLive + self.loss_cnnSpoof
        self.loss_total.backward()
        self.net_opt.step()
    

    def net_update_lstm(self, videos, gtClsLabels, da_lambda, batch_size_lstm, idxLive, liveDomainGTLables, idxSpoof, spoofDomainGTLables):
        
        # --- Data preparation
        self.net_opt.zero_grad()
        sizes = videos.size() # 12 * 8(frames) * 3 *224* 224
        frames = videos.size(1) # --- should be 8 by now
        
        video_inp = videos.view(-1,sizes[2],sizes[3],sizes[4]) # --- video input: 96* 3* 224* 224
        resnet_out = self.gen(video_inp) # --- resnet_out: 96 *2048

        lstm_input = resnet_out.view(sizes[0],frames,-1) # --- 12* 8* 2048
        lstm_out, lstm_feat = self.lstm(lstm_input,return_feat = True) # --- !lstm_out: 12*2; lstm_feat: 12*2048
        lstm_grl_out = GradRevLayer.apply(lstm_feat, da_lambda) # --- lstm_grl_out: 12*2048
        lstm_cond_out = self.lstm_cond_dom_head(lstm_grl_out) # --- 12* 1024

        lstm_cond_live = lstm_cond_out.index_select(0, idxLive.squeeze()) # --- b *1024
        lstm_cond_spoof = lstm_cond_out.index_select(0, idxSpoof.squeeze()) # ---(12-b) *1024
        lstm_live_out = self.lstm_cond_dom_tail_live(lstm_cond_live) # --- !lstm_live_out: b*3
        lstm_spoof_out = self.lstm_cond_dom_tail_spoof(lstm_cond_spoof) # --- !lstm_spoof_out: (12-b)*3
        self.loss_lstm = self.cls_criterion(lstm_out,gtClsLabels) 

    # --- Option 0 ---
        if self.option == 0:
            if liveDomainGTLables.size(0) == 0:
                self.loss_lstm_live = torch.tensor(0).to(self.device)
            else:
                self.loss_lstm_live = self.cls_criterion(lstm_live_out,liveDomainGTLables) 

            if spoofDomainGTLables.size(0) == 0:
                self.loss_lstm_spoof = torch.tensor(0).to(self.device)
            else:
                self.loss_lstm_spoof = self.cls_criterion(lstm_spoof_out,spoofDomainGTLables) 
    # --- End of Option 0 --- 
    # --- Option 1 --- 
        elif self.option == 1:
            if liveDomainGTLables.size(0) == 0:
                self.loss_lstm_spoof = self.cls_criterion(lstm_spoof_out,spoofDomainGTLables)
                self.loss_lstm_live = self.loss_lstm_spoof
            elif spoofDomainGTLables.size(0) == 0:
                self.loss_lstm_live = self.cls_criterion(lstm_live_out,liveDomainGTLables)
                self.loss_lstm_spoof = self.loss_lstm_live
            else:
                self.loss_lstm_live = self.cls_criterion(lstm_live_out,liveDomainGTLables) 
                self.loss_lstm_spoof = self.cls_criterion(lstm_spoof_out,spoofDomainGTLables)

    # --- End of Option 1 --- 


        self.loss_total = self.loss_lstm + self.loss_lstm_live + self.loss_lstm_spoof
        self.loss_total.backward()
        self.net_opt.step()

    def net_update_lstm_no_dg(self, videos, gtClsLabels, da_lambda, batch_size_lstm, idxLive, liveDomainGTLables, idxSpoof, spoofDomainGTLables):
        # --- Data preparation
        self.net_opt.zero_grad()
        sizes = videos.size()  # 12 * 8(frames) * 3 *224* 224
        frames = videos.size(1)  # --- should be 8 by now

        video_inp = videos.view(-1, sizes[2], sizes[3], sizes[4])  # --- video input: 96* 3* 224* 224
        resnet_out = self.gen(video_inp)  # --- resnet_out: 96 *2048

        lstm_input = resnet_out.view(sizes[0], frames, -1)  # --- 12* 8* 2048
        lstm_out, lstm_feat = self.lstm(lstm_input, return_feat=True)  # --- !lstm_out: 12*2; lstm_feat: 12*2048
        self.loss_lstm = self.cls_criterion(lstm_out, gtClsLabels)
        self.loss_lstm.backward()
        self.net_opt.step()

    def forward_lstm(self, video):
        sizes = video.size()
        vids = video.view(-1,sizes[2],sizes[3],sizes[4]) # --- 16*3*224*224
        resnet_out = self.gen(vids)
        resnet_clsnet_out = self.gen.fc(resnet_out) # --- 16*2
        lstm_input = resnet_out.view(sizes[0],sizes[1],-1) # --- should be 2*8*2048
        lstm_out = self.lstm(lstm_input) # --- 4*2
        return resnet_clsnet_out, lstm_out

    def forward(self, video):
        print('>>> Forwarding <<<')
        sizes = video.size()
        vids = video.view(-1,sizes[2],sizes[3],sizes[4]) # --- 16*3*224*224
        resnet_out = self.gen(vids)
        resnet_clsnet_out = self.gen.fc(resnet_out) # --- 16*2
        lstm_input = resnet_out.view(sizes[0],sizes[1],-1) # --- should be 2*8*2048
        lstm_out = self.lstm(lstm_input) # --- 4*2
        return resnet_clsnet_out,lstm_out

    def save(self, snapshot_dir, iterations):
        # Save net params
        net_name = os.path.join(snapshot_dir, 'net_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')

        torch.save({
                'gen': self.gen.state_dict(),
                'lstm': self.lstm.state_dict(),
                'cnn_cond_dom_head': self.cnn_cond_dom_head.state_dict(),
                'cnn_cond_dom_tail_live': self.cnn_cond_dom_tail_live.state_dict(),
                'cnn_cond_dom_tail_spoof': self.cnn_cond_dom_tail_spoof.state_dict(),
                'lstm_cond_dom_head' : self.lstm_cond_dom_head.state_dict(),
                'lstm_cond_dom_tail_live' : self.lstm_cond_dom_tail_live.state_dict(),
                'lstm_cond_dom_tail_spoof' : self.lstm_cond_dom_tail_spoof.state_dict()
                }, net_name)

        torch.save({'net_opt': self.net_opt.state_dict()}, opt_name)
        
        return net_name

    def resume(self, checkpoint_dir, config, resume_iteration=None):
        # Load generators
        if not resume_iteration:
            last_model_name = get_model_list(checkpoint_dir, "net")
            state_dict = torch.load(last_model_name, map_location=self.device)
            self.gen.load_state_dict(state_dict['gen'])
            self.lstm.load_state_dict(state_dict['lstm'])
            self.cnn_cond_dom_head.load_state_dict(state_dict['cnn_cond_dom_head'])
            self.cnn_cond_dom_tail_live.load_state_dict(state_dict['cnn_cond_dom_tail_live'])
            self.cnn_cond_dom_tail_spoof.load_state_dict(state_dict['cnn_cond_dom_tail_spoof'])
            self.lstm_cond_dom_head.load_state_dict(state_dict['lstm_cond_dom_head'])
            self.lstm_cond_dom_tail_live.load_state_dict(state_dict['lstm_cond_dom_tail_live'])
            self.lstm_cond_dom_tail_spoof.load_state_dict(state_dict['lstm_cond_dom_tail_spoof'])

            splitStr1 = last_model_name.split('/')
            splitStr1 = splitStr1[-1].split('_')[-1]
            iterations = int(splitStr1.split('.')[0])
        else:
            last_model_name = join(checkpoint_dir, 'net_{:08d}.pt'.format(resume_iteration))
            state_dict = torch.load(last_model_name, map_location=self.device)
            self.gen.load_state_dict(state_dict['gen'])
            self.lstm.load_state_dict(state_dict['lstm'])
            self.cnn_cond_dom_head.load_state_dict(state_dict['cnn_cond_dom_head'])
            self.cnn_cond_dom_tail_live.load_state_dict(state_dict['cnn_cond_dom_tail_live'])
            self.cnn_cond_dom_tail_spoof.load_state_dict(state_dict['cnn_cond_dom_tail_spoof'])
            self.lstm_cond_dom_head.load_state_dict(state_dict['lstm_cond_dom_head'])
            self.lstm_cond_dom_tail_live.load_state_dict(state_dict['lstm_cond_dom_tail_live'])
            self.lstm_cond_dom_tail_spoof.load_state_dict(state_dict['lstm_cond_dom_tail_spoof'])
            iterations = resume_iteration

        print(' --- resuming from checkpoint at iteration {} --- '.format(iterations))
        print(' --- checkpoint loaded from: {} --- '.format(last_model_name))
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'), map_location=self.device)
        self.net_opt.load_state_dict(state_dict['net_opt'])
        # Reinitilize schedulers
        self.net_scheduler = get_scheduler(self.net_opt, config['optim'], iterations)
        iterations -= 1  # iteration is 0 indexed, so need to minus one
        return iterations

    def update_learning_rate(self):
        self.net_scheduler.step()

    def set2train(self):
        self.train()
        print(' ++++++++++++++ setting network in train mode ++++++++++++++')

    def set2eval(self):
        self.eval()
        print(' ++++++++++++++ setting network in eval mode ++++++++++++++')


    def restore_model(self, iter):
        pass
