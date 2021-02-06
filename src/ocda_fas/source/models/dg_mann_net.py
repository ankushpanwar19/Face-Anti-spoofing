import numpy as np
import torch
import torch.nn as nn
# from .utils import register_model, get_model
from . import cos_norm_classifier
import os
from .dg_resnet2 import DgEncoder


class DgMannNet(nn.Module):

    """Defines a Dynamic Meta-Embedding Network."""

    def __init__(self,config, num_cls=2, use_init_weights=True,feat_dim=2048,discrim_feat=False,use_domain_factor_selector=False):

        super(DgMannNet, self).__init__()
        self.config=config
        self.device=config['device']
        self.discrim_feat=discrim_feat
        self.num_cls = num_cls
        self.feat_dim = feat_dim
        self.cls_criterion = nn.CrossEntropyLoss()
        self.gan_criterion = nn.CrossEntropyLoss()

        self.src_net = DgEncoder(config)
        self.tgt_net = DgEncoder(config)
        self.use_domain_factor_selector=use_domain_factor_selector
        self.centroids = torch.from_numpy(np.load(config['centroids_file'])).float()
        assert self.centroids is not None
        self.centroids.requires_grad = False

        if use_init_weights:
            self.load_src_net(config['src_checkpoint_file'])
        else:
            print('MannNet is not initialized with weights.')

        if self.discrim_feat:
            input_dim = self.feat_dim
        else:
            input_dim=self.num_cls

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
            )

        self.fc_selector = nn.Linear(self.feat_dim, self.feat_dim)

        if self.use_domain_factor_selector:
            self.domain_factor_selector = nn.Linear(self.feat_dim, self.feat_dim)



    def forward(self, x_s, x_t):

        """Pass source and target images through their respective networks."""

        x_s,score_xs,prob_xs = self.src_net(x_s,)
        x_t,score_xt,prob_xt = self.tgt_net(x_t)

        if self.discrim_feat:
            score_ds = self.discriminator(x_s.clone())
            score_dt = self.discriminator(x_t.clone())
        else:
            score_ds = self.discriminator(score_xs.clone())
            score_dt = self.discriminator(score_xt.clone())

        return score_xs, score_xt, score_ds, score_dt

    def load_src_net(self, checkpoint_file):
        """Initialize source and target with source weights."""
        if self.config['mann_net']['pretrained']=='dg':
            self.src_net.load_init_weights(checkpoint_file)
            self.tgt_net.load_init_weights(checkpoint_file)
            print("Intialized with Dg net weights")
        elif self.config['mann_net']['pretrained']=='src_net':
            self.src_net.load(checkpoint_file,'encoder','classifier')
            self.tgt_net.load(checkpoint_file,'encoder','classifier')
            print("Intialized with src net weights")

    def save(self, out_path):

        if self.use_domain_factor_selector:
            torch.save({
                'tgt_encoder': self.tgt_net.encoder.state_dict(),
                'tgt_clf': self.tgt_net.classifier.state_dict(),
                'src_encoder': self.src_net.encoder.state_dict(),
                'src_clf': self.src_net.classifier.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'fc_selector':self.fc_selector.state_dict(),
                'domain_factor_selector':self.domain_factor_selector.state_dict(),
                }, out_path)
        else:
            torch.save({
                'tgt_encoder': self.tgt_net.encoder.state_dict(),
                'tgt_clf': self.tgt_net.classifier.state_dict(),
                'src_encoder': self.src_net.encoder.state_dict(),
                'src_clf': self.src_net.classifier.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'fc_selector':self.fc_selector.state_dict(),
                }, out_path)
        

    def load(self,checkpoint_file):
        self.src_net.load(checkpoint_file,'src_encoder','src_clf')
        self.tgt_net.load(checkpoint_file,'tgt_encoder','tgt_clf')
        state_dict = torch.load(checkpoint_file, map_location=self.device)
        self.discriminator.load_state_dict(state_dict['discriminator'])
        # self.fc_selector.load_state_dict(state_dict['fc_selector'])

        print('MannNet is initialized with previously trained MannNet.')

    def save_tgt_net(self, out_path):
        torch.save(self.tgt_net.state_dict(), out_path)

