import os
import torch
import torch.nn as nn
import sys
sys.path.append("src")
from pytorch_resnet import ResNet, Bottleneck
import torch.utils.model_zoo as model_zoo
from networks import ResNetClsNet
from utils import get_config, make_dir

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

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

        self.gen = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)
        self.gen.fc=Identity()  # 8631
        self.config_resnet_clsnet = config['resent_clsnet']
        self.classifier = ResNetClsNet(self.config_resnet_clsnet, self.debug,'ResNetClsNet')

    
    def load_init_weights(self, checkpoint_file):
        # Load generators
        print(self.device)
        pre_state_dict = torch.load(checkpoint_file, map_location=self.device)
        pretrained_weights=list(pre_state_dict['gen'].items())

        mymodel=self.gen.state_dict()
        count=0
        for key,value in mymodel.items():
            layer_name,weights=pretrained_weights[count]      
            mymodel[key]=weights
            count+=1

        cls_pretrained_wt=[]
        for k in pre_state_dict['gen'].keys():
            if 'fc' in k:
                cls_pretrained_wt.append(pre_state_dict['gen'][k])

        cls_mymodel=self.classifier.state_dict()
        count=0
        for key,value in cls_mymodel.items():
            cls_mymodel[key]=cls_pretrained_wt[count]
            count+=1
        
        self.gen.load_state_dict(mymodel)
        self.classifier.load_state_dict(cls_mymodel)
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
    checkpoint_file='output/fas_project/DG_exp/lstmmot_exp_013/checkpoints/net_00039439.pt'
    m=DgEncoder(config)
    m.load_init_weights(checkpoint_file)
    print("end")