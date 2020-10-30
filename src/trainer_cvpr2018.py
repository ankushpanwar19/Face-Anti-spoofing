import torch
import torch.nn as nn
from networks import CVPR2018EncoderV2, ResizeAndStackFeatureMaps, ConvHead, ClsNet, ResNetClsNet, New_convhead
from utils import weights_init, get_model_list
import os
from utils import print_feat_shape
# from schedulers import get_scheduler
from utils import get_scheduler
from networks import GradRevLayer, ResNetClsNet, ClsCondDomainNet, ClsCondDomainNetHeads, Cls_lstm


class TrainerCvpr2018Dg(nn.Module):
    def __init__(self, config):
        super(TrainerCvpr2018Dg, self).__init__()

        print(' ---- class TrainerCvpr2018Dg --> __int__() --- ')
        self.clsnet = config['anet_clsnet']
        self.device = config['device']
        self.debug = config['debug']

        self.gen = CVPR2018EncoderV2(config['gen'], self.debug)
        self.lstm = Cls_lstm(config['batch_size_lstm'],inp_dim = 1024, out_dim1 =512)
        self.ResStack = ResizeAndStackFeatureMaps(config['res_dim'], self.debug) 
        # self.ConvHead = ConvHead(config['convhead'], self.debug, 'ConvHead')
        self.New_convhead = New_convhead()
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.ClsNet = ClsNet(self.clsnet, self.debug, 'ClsNet')

        self.cnn_cond_dom_head = ClsCondDomainNet(config['ClsCondDomainNet4Baseline'], self.debug, 'ClsCondDomainNet4Baseline')
        self.cnn_cond_dom_tail_live = ClsCondDomainNetHeads(config['ClsCondDomainNetHeads'], self.debug, 'ClsCondDomainNetHeadsLive')
        self.cnn_cond_dom_tail_spoof = ClsCondDomainNetHeads(config['ClsCondDomainNetHeads'], self.debug, 'ClsCondDomainNetHeadsSpoof')

        self.lstm_cond_dom_head = ClsCondDomainNet(config['ClsCondDomainNet4Baseline'], self.debug, 'ClsCondDomainNet4Baseline')
        self.lstm_cond_dom_tail_live = ClsCondDomainNetHeads(config['ClsCondDomainNetHeads'], self.debug, 'lstmDomainNetHeadsLive')
        self.lstm_cond_dom_tail_spoof = ClsCondDomainNetHeads(config['ClsCondDomainNetHeads'], self.debug, 'lstmDomainNetHeadsSpoof')

        net_params = list(self.gen.parameters()) + \
                     list(self.ResStack.parameters()) + \
                     list(self.ClsNet.parameters()) + \
                     list(self.cnn_cond_dom_head.parameters()) + \
                     list(self.cnn_cond_dom_tail_live.parameters()) + \
                     list(self.cnn_cond_dom_tail_spoof.parameters()) + \
                     list(self.lstm_cond_dom_head.parameters()) + \
                     list(self.lstm_cond_dom_tail_live.parameters()) + \
                     list(self.lstm_cond_dom_tail_spoof.parameters()) + \
                     list(self.New_convhead.parameters())
                     # list(self.ConvHead.parameters()) 

        self.nll_loss = nn.NLLLoss()
        optim_type = config['optim']['optim_type']
        lr = config['optim']['lr']
        momentum = config['optim']['momentum']
        beta1 = config['optim']['beta1']
        beta2 = config['optim']['beta2']
        weight_decay = config['optim']['weight_decay']
        if optim_type == 'sgd':
            self.net_opt = torch.optim.SGD([p for p in net_params if p.requires_grad], lr=lr, momentum=momentum)
        elif optim_type == 'adam':
            self.net_opt = torch.optim.Adam([p for p in net_params if p.requires_grad], lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        else:
            pass

        print('--- optimizer: {} ---'.format(optim_type))

        scheduler_config = config['scheduler']
        print(scheduler_config)
        print(scheduler_config['parameters'])
        scheduler_cls = get_scheduler(scheduler_config['lr_policy'])
        if scheduler_config['lr_policy'] == 'PolynomialLR':
            self.net_scheduler = scheduler_cls(self.net_opt, **scheduler_config['parameters'])
        else:
            self.net_scheduler = scheduler_cls(self.net_opt)

        if config['optim']['weight_init']:
            self.apply(weights_init(config['optim']['init']))

        if not config['option'] or (not 'option' in config.keys()):
            self.option = 0
        elif config['option'] == 1:
            self.option = 1

        self.loss_cnn_cls = 0
        self.loss_total = 0

    def cls_criterion(self, cnet_out, clslabels):
        cls_loss = self.nll_loss(cnet_out, clslabels)
        return cls_loss

    def net_update_cnn(self, imgBatch, gtClsLabels, da_lambda, batch_size_cnn, idxLive, liveDomainGTLables, idxSpoof, spoofDomainGTLables):
        self.net_opt.zero_grad()
        if self.debug:
            print(' --- trainer_cvpr2018.py --> class TrainerCvpr2018Dg --> net_update_cnn() --- ')

        out_b2, out_b3, out_b4 = self.gen(imgBatch)
        resstack_out = self.ResStack(out_b2, out_b3, out_b4) # --- ([24, 384, 32, 32])
        # convhead_out = self.ConvHead(resstack_out) # --- 
        convhead_out = self.New_convhead(resstack_out) # --- 24*512*2*2
        convhead_out = convhead_out.view(convhead_out.size(0),-1)
        clsnet_out = self.ClsNet(convhead_out)

        grl_out = GradRevLayer.apply(convhead_out, da_lambda)

        clsCondOut = self.cnn_cond_dom_head(grl_out)

        clsCondLiveOut = self.cnn_cond_dom_tail_live(clsCondOut.index_select(0, idxLive.squeeze()))
        clsCondSpoofOut = self.cnn_cond_dom_tail_spoof(clsCondOut.index_select(0, idxSpoof.squeeze()))

        if liveDomainGTLables.size(0) == 0:
            self.loss_cnnLive = torch.tensor(0).to('cuda')
        else:
            self.loss_cnnLive = self.cls_criterion(clsCondLiveOut, liveDomainGTLables)  # --- livedomaingtlabels: a

        if spoofDomainGTLables.size(0) == 0:
            self.loss_cnnSpoof = torch.tensor(0).to('cuda')
        else:
            self.loss_cnnSpoof = self.cls_criterion(clsCondSpoofOut, spoofDomainGTLables)  # --- spoofDomainGTLables : 48-a
        self.loss_cnn_cls = self.cls_criterion(clsnet_out, gtClsLabels)
        self.loss_total = self.loss_cnn_cls + self.loss_cnnLive + self.loss_cnnSpoof
        self.loss_total.backward()
        self.net_opt.step()

    def net_update_bline(self, imgBatch, gtClsLabels, da_lambda, batch_size_cnn, idxLive, liveDomainGTLables, idxSpoof, spoofDomainGTLables):
        self.net_opt.zero_grad()
        if self.debug:
            print(' --- trainer_cvpr2018.py --> class TrainerCvpr2018Dg --> net_update_cnn() --- ')

        out_b2, out_b3, out_b4 = self.gen(imgBatch)
        resstack_out = self.ResStack(out_b2, out_b3, out_b4) # --- ([24, 384, 32, 32])
        # convhead_out = self.ConvHead(resstack_out) # --- 
        convhead_out = self.New_convhead(resstack_out) # --- 24*512*2*2
        convhead_out = convhead_out.view(convhead_out.size(0),-1)
        clsnet_out = self.ClsNet(convhead_out)

        self.loss_cnn_cls = self.cls_criterion(clsnet_out, gtClsLabels)
        self.loss_cnn_cls.backward()
        self.net_opt.step()


    def net_update_lstm(self, videos, gtClsLabels, da_lambda, batch_size_lstm, idxLive, liveDomainGTLables, idxSpoof, spoofDomainGTLables):
        self.net_opt.zero_grad()
        if self.debug:
            print(' --- trainer_cvpr2018.py --> class TrainerCvpr2018Dg --> net_update_lstm() --- ')

        sizes = videos.size()
        # frames = videos.size(1)
        frames = 8
        
        video_inp = videos.view(-1, sizes[2], sizes[3], sizes[4])
        out_b2, out_b3, out_b4 = self.gen(video_inp)
        resstack_out = self.ResStack(out_b2, out_b3, out_b4)
        # convhead_out = self.ConvHead(resstack_out)
        convhead_out = self.New_convhead(resstack_out)
        
        lstm_input = convhead_out.view(sizes[0], frames, -1)
        # print(' ???? convhead out: {}, lstm_input: {}'.format(convhead_out.size(),lstm_input.size()))
        lstm_out, lstm_feat = self.lstm(lstm_input, return_feat=True)
        # print(' ???? convhead out: {}, lstm_input: {}, lstm_feat :{}'.format(convhead_out.size(),lstm_input.size(),lstm_feat.size()))
        lstm_grl_out = GradRevLayer.apply(lstm_feat, da_lambda)
        lstm_cond_out = self.lstm_cond_dom_head(lstm_grl_out)

        lstm_cond_live = lstm_cond_out.index_select(0, idxLive.squeeze())
        lstm_cond_spoof = lstm_cond_out.index_select(0, idxSpoof.squeeze())

        
        lstm_live_out = self.lstm_cond_dom_tail_live(lstm_cond_live)
        lstm_spoof_out = self.lstm_cond_dom_tail_spoof(lstm_cond_spoof)
        self.loss_lstm = self.cls_criterion(lstm_out, gtClsLabels)

        # --- Option 0 ---
        if self.option == 0:
            if liveDomainGTLables.size(0) == 0:
                self.loss_lstm_live = torch.tensor(0).to('cuda')
            else:
                self.loss_lstm_live = self.cls_criterion(lstm_live_out, liveDomainGTLables)

            if spoofDomainGTLables.size(0) == 0:
                self.loss_lstm_spoof = torch.tensor(0).to('cuda')
            else:
                self.loss_lstm_spoof = self.cls_criterion(lstm_spoof_out, spoofDomainGTLables)
                # --- End of Option 0 ---
        # --- Option 1 ---
        elif self.option == 1:
            if liveDomainGTLables.size(0) == 0:
                self.loss_lstm_spoof = self.cls_criterion(lstm_spoof_out, spoofDomainGTLables)
                self.loss_lstm_live = self.loss_lstm_spoof
            elif spoofDomainGTLables.size(0) == 0:
                self.loss_lstm_live = self.cls_criterion(lstm_live_out, liveDomainGTLables)
                self.loss_lstm_spoof = self.loss_lstm_live
            else:
                self.loss_lstm_live = self.cls_criterion(lstm_live_out, liveDomainGTLables)
                self.loss_lstm_spoof = self.cls_criterion(lstm_spoof_out, spoofDomainGTLables)

        # --- End of Option 1 ---

        self.loss_total = self.loss_lstm + self.loss_lstm_live + self.loss_lstm_spoof
        self.loss_total.backward()
        self.net_opt.step()

    def net_update_lstm_no_dg(self, videos, gtClsLabels, da_lambda, batch_size_lstm, idxLive, liveDomainGTLables, idxSpoof, spoofDomainGTLables):
        self.net_opt.zero_grad()
        if self.debug:
            print(' --- trainer_cvpr2018.py --> class TrainerCvpr2018Dg --> net_update_lstm_no_dg() --- ')

        sizes = videos.size()
        frames = videos.size(1)

        video_inp = videos.view(-1, sizes[2], sizes[3], sizes[4])
        out_b2, out_b3, out_b4 = self.gen(video_inp)
        resstack_out = self.ResStack(out_b2, out_b3, out_b4)
        convhead_out = self.ConvHead(resstack_out)

        lstm_input = convhead_out.view(sizes[0], frames, -1)
        lstm_out, lstm_feat = self.lstm(lstm_input, return_feat=True)
        self.loss_lstm = self.cls_criterion(lstm_out, gtClsLabels)
        self.loss_lstm.backward()
        self.net_opt.step()

    def forward_lstm(self, videos):
        self.net_opt.zero_grad()
        if self.debug:
            print(' --- trainer_cvpr2018.py --> class TrainerCvpr2018Dg --> net_update_lstm_no_dg() --- ')

        sizes = videos.size()
        frames = videos.size(1)
        # print('>>>> sizes: {}'.format(sizes))
        video_inp = videos.view(-1, sizes[2], sizes[3], sizes[4])
        # print('>>>> vid inp size',video_inp.size())
        out_b2, out_b3, out_b4 = self.gen(video_inp)
        resstack_out = self.ResStack(out_b2, out_b3, out_b4)
        # convhead_out = self.ConvHead(resstack_out)
        convhead_out = self.New_convhead(resstack_out)
        clsnet_out = self.ClsNet(convhead_out)
        # print('>>> convhead out : {}'.format(convhead_out.size()))
        lstm_input = convhead_out.view(sizes[0], 1, -1)
        # print('>>> lstm input: {}'.format(lstm_input.size()))
        lstm_out, lstm_feat = self.lstm(lstm_input, return_feat=True)

        return clsnet_out, lstm_out


    def forward(self, images):
        if self.debug:
            print(' --- trainer_cvpr2018.py --> class TrainerCvpr2018Dg --> net_update_cnn() --- ')

        out_b2, out_b3, out_b4 = self.gen(images)
        resstack_out = self.ResStack(out_b2, out_b3, out_b4)
        convhead_out = self.ConvHead(resstack_out)
        clsnet_out = self.ClsNet(convhead_out)

        if self.debug:
            print(' --- trainer_cvpr2018.py --> class TrainerCvpr2018Dg --> net_update_cnn() --- ')
            kwargs = {'resstack_out': resstack_out, 'convhead_out': convhead_out, 'clsnet_out': clsnet_out}
            print_feat_shape(**kwargs)

        return clsnet_out

    def set2train(self):
        self.train()
        print(' ++++++++++++++ setting network in train mode ++++++++++++++')

    def set2eval(self):
        self.eval()
        print(' ++++++++++++++ setting network in eval mode ++++++++++++++')

    def save(self, snapshot_dir, iterations):
        # Save net params
        net_name = os.path.join(snapshot_dir, 'net_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({
            'gen': self.gen.state_dict(),
            'ResStack': self.ResStack.state_dict(),
            'ClsNet': self.ClsNet.state_dict(),
            'cnn_cond_dom_head': self.cnn_cond_dom_head.state_dict(),
            'cnn_cond_dom_tail_live': self.cnn_cond_dom_tail_live.state_dict(),
            'cnn_cond_dom_tail_spoof': self.cnn_cond_dom_tail_spoof.state_dict(),
            'lstm_cond_dom_head': self.lstm_cond_dom_head.state_dict(),
            'lstm_cond_dom_tail_live': self.lstm_cond_dom_tail_live.state_dict(),
            'lstm_cond_dom_tail_spoof': self.lstm_cond_dom_tail_spoof.state_dict(),
            'New_convhead':self.New_convhead
            # 'ConvHead': self.ConvHead.state_dict(),

        }, net_name)

        torch.save({'net_opt': self.net_opt.state_dict()}, opt_name)
        return net_name

    def update_learning_rate(self):
        self.net_scheduler.step()



class TrainerCvpr2018(nn.Module):
    def __init__(self, config):
        super(TrainerCvpr2018, self).__init__()

        print(' ---- class TrainerCvpr2018 --> __int__() --- ')
        self.clsnet = config['resent_clsnet']
        self.device = config['device']
        self.debug = config['debug']

        self.gen = CVPR2018EncoderV2(config['gen'], self.debug)
        self.ResStack = ResizeAndStackFeatureMaps(config['res_dim'], self.debug)
        self.ConvHead = ConvHead(config['convhead'], self.debug, 'ConvHead')

        if config['clsnet_type'] == 'ClsNet':
            self.ClsNet = ClsNet(self.clsnet, self.debug, 'ClsNet')
        elif config['clsnet_type'] == 'ResNetClsNet':
            self.ClsNet = ResNetClsNet(self.clsnet, self.debug, 'ResNetClsNet')
        else:
            pass

        net_params = list(self.gen.parameters()) + \
                     list(self.ResStack.parameters()) + \
                     list(self.ConvHead.parameters()) + list(self.ClsNet.parameters())

        self.nll_loss = nn.NLLLoss()
        optim_type = config['optim']['optim_type']
        lr = config['optim']['lr']
        momentum = config['optim']['momentum']
        beta1 = config['optim']['beta1']
        beta2 = config['optim']['beta2']
        weight_decay = config['optim']['weight_decay']
        if optim_type == 'sgd':
            self.net_opt = torch.optim.SGD([p for p in net_params if p.requires_grad], lr=lr, momentum=momentum)
        elif optim_type == 'adam':
            self.net_opt = torch.optim.Adam([p for p in net_params if p.requires_grad], lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        else:
            pass

        print('--- optimizer: {} ---'.format(optim_type))

        scheduler_config = config['scheduler']
        print(scheduler_config)
        print(scheduler_config['parameters'])
        scheduler_cls = get_scheduler(scheduler_config['lr_policy'])
        if scheduler_config['lr_policy'] == 'PolynomialLR':
            self.net_scheduler = scheduler_cls(self.net_opt, **scheduler_config['parameters'])
        else:
            self.net_scheduler = scheduler_cls(self.net_opt)

        if config['optim']['weight_init']:
            self.apply(weights_init(config['optim']['init']))

    def cls_criterion(self, cnet_out, clslabels):
        cls_loss = self.nll_loss(cnet_out, clslabels)
        return cls_loss

    def net_update(self, images, clslabels):

        self.net_opt.zero_grad()

        if self.debug:
            print(' --- trainer_cvpr2018.py --> class TrainerCvpr2018 --> net_update() --- ')

        out_b2, out_b3, out_b4 = self.gen(images)
        resstack_out = self.ResStack(out_b2, out_b3, out_b4)
        convhead_out = self.ConvHead(resstack_out)
        clsnet_out = self.ClsNet(convhead_out)

        if self.debug:
            print(' --- trainer_cvpr2018.py --> class TrainerCvpr2018 --> net_update() --- ')
            kwargs = {'resstack_out': resstack_out, 'convhead_out': convhead_out, 'clsnet_out': clsnet_out}
            print_feat_shape(**kwargs)

        self.loss_cls = self.cls_criterion(clsnet_out, clslabels)
        self.loss_cls.backward()
        self.net_opt.step()

    def forward(self, images):
        if self.debug:
            print(' --- trainer_cvpr2018.py --> class TrainerCvpr2018 --> net_update() --- ')

        out_b2, out_b3, out_b4 = self.gen(images)
        resstack_out = self.ResStack(out_b2, out_b3, out_b4)
        convhead_out = self.ConvHead(resstack_out)
        clsnet_out = self.ClsNet(convhead_out)

        if self.debug:
            print(' --- trainer_cvpr2018.py --> class TrainerCvpr2018 --> net_update() --- ')
            kwargs = {'resstack_out': resstack_out, 'convhead_out': convhead_out, 'clsnet_out': clsnet_out}
            print_feat_shape(**kwargs)

        return clsnet_out

    def set2train(self):
        self.train()
        print(' ++++++++++++++ setting network in train mode ++++++++++++++')

    def set2eval(self):
        self.eval()
        print(' ++++++++++++++ setting network in eval mode ++++++++++++++')

    def save(self, snapshot_dir, iterations):
        # Save net params
        net_name = os.path.join(snapshot_dir, 'net_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({
            'gen': self.gen.state_dict(),
            'ResStack': self.ResStack.state_dict(),
            'ConvHead': self.ConvHead.state_dict(),
            'ClsNet': self.ClsNet.state_dict()
        }, net_name)

        torch.save({'net_opt': self.net_opt.state_dict()}, opt_name)
        return net_name

    def update_learning_rate(self):
        self.net_scheduler.step()