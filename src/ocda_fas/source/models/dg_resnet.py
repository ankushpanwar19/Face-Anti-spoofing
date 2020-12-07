import os
import torch
import torch.nn as nn
import sys
sys.path.append("src")
from pytorch_resnet import ResNet, Bottleneck
import torch.utils.model_zoo as model_zoo
from networks import ResNetClsNet
from utils import get_config, make_dir

class DgEncoder(nn.Module):

    def __init__(self, config):
        super(DgEncoder, self).__init__()

        self.device=config['device']
        self.debug = config['debug']
        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        }

        self.gen = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)  # 8631
        self.config_resnet_clsnet = config['resent_clsnet']
        self.gen.fc = ResNetClsNet(self.config_resnet_clsnet, self.debug,'ResNetClsNet')

    
    def load_init_weights(self, checkpoint_file):
        # Load generators
        print(self.device)
        state_dict = torch.load(checkpoint_file, map_location=self.device)
        self.gen.load_state_dict(state_dict['gen'])
        print("***** Weight initialized with DgNet trained previously *****")

    def forward(self, images):
        resnet_out = self.gen(images)
        resnet_clsnet_out = self.gen.fc(resnet_out) # --- 16*2
        softmax=nn.Softmax(dim=1)
        soft_clsnet_out=softmax(resnet_clsnet_out)
        return resnet_out,resnet_clsnet_out,soft_clsnet_out

    def load(self, checkpoint_file,key):
        state_dict = torch.load(checkpoint_file, map_location=self.device)
        self.gen.load_state_dict(state_dict[key])
        

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_fname='src/configs/train.yaml'
    config= get_config(config_fname)
    config['device']=device

    m=DgEncoder(config)
    print("end")