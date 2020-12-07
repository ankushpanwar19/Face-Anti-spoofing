from copy import deepcopy
import os
import torch
import torch.nn as nn
# from utils import register_model, get_model
# from . import cos_norm_classifier
from .dg_resnet import DgEncoder


class DgDomainFactorNet(nn.Module):

    """Defines a Dynamic Meta-Embedding Network."""

    def __init__(self, config,num_cls=2,use_init_weight=True):

        super(DgDomainFactorNet, self).__init__()

        self.device=config['device']

        self.cls_criterion = nn.CrossEntropyLoss()
        self.gan_criterion = nn.CrossEntropyLoss()
        self.rec_criterion = nn.SmoothL1Loss()


        self.tgt_net = DgEncoder(config)

        self.domain_factor_net = DgEncoder(config)

        # self.discriminator_cls = cos_norm_classifier.create_model(512, self.num_cls)

        self.decoder = Decoder(input_dim=4096)

        if use_init_weight:
            if os.path.isfile(config['tgt_mann_checkpoint_file']):
                self.load_init_weights(config['tgt_mann_checkpoint_file'])
            else:
                raise Exception('DgDomainFactorNet must be initialized with weights.')

    def forward(self, x):
        pass

    def load_init_weights(self, checkpoint_file):

        """
        Load weights from pretrained tgt model
        and initialize DomainFactorNet from pretrained tgr model.
        """
        self.tgt_net.load(checkpoint_file,'tgt_gen')
        self.domain_factor_net.load(checkpoint_file,'tgt_gen')
        print("**** Weights have been initialized with mann net *****")

    def save(self, out_path):
        torch.save({
                'domain_gen': self.domain_factor_net.gen.state_dict(),
                'decoder': self.decoder.state_dict()
                }, out_path)

    def save_domain_factor_net(self, out_path):
        torch.save({self.domain_factor_net.state_dict()}, out_path)

    def load(self,domain_chkpoint_file,tgt_chkpoint_file=None):

        """
        Load weights from pretrained tgt model
        and initialize DomainFactorNet from pretrained tgr model.
        """
        if (tgt_chkpoint_file is not None):
            self.tgt_net.load(tgt_chkpoint_file,'tgt_gen')
        self.domain_factor_net.load(domain_chkpoint_file,'domain_gen')

        self.decoder.load(domain_chkpoint_file,self.device)
        print("**** Weights have been initialized with previously Domain Factor net *****")


class Decoder(nn.Module):

    def __init__(self, input_dim=4096):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 3920))

        self.decoder = nn.Sequential(
            nn.Conv2d(20, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="bilinear",align_corners=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="bilinear",align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="bilinear",align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="bilinear",align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 3, kernel_size=1))

    def forward(self, x):
        assert x.size(1) == self.input_dim
        x = self.fc(x)
        x = x.view(x.size(0), 20, 14, 14)
        x = self.decoder(x)
        return x

    def load(self , checkpoint,device):

        state_dict = torch.load(checkpoint, map_location=device)

        self.load_state_dict(state_dict['decoder'])


if __name__ == "__main__":
    
    dec=DomainFactorNet()
    inp=torch.rand((16,3,224,224))

#%%
# %%
