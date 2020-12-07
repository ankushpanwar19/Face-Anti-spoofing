import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter

import pdb

class ResNetClsNet(nn.Module):
    def __init__(self, config, debug, nettype=None):
        super(ResNetClsNet, self).__init__()
        # print(' --- networks.py --> class ResNetClsNet -->  __init__() for {} ---'.format(nettype))
        self.debug = debug
        self.nettype = nettype
        self.inp_dim = config['inp_dim']
        self.out_dim1 = config['out_dim1']
        self.out_dim2 = config['out_dim2']
        self.cost_func = config['cost_func']
        self.drpval = config['dropoutv']
        # print('[inp_dim:{}] [out_dim1:{}] [out_dim2: {}] [loss: {}]'.format(self.inp_dim, self.out_dim1, self.out_dim2, self.cost_func))
        # print('-------------------------------')
        model = []
        model += [nn.Linear(self.inp_dim, self.out_dim1)]
        model += [nn.ReLU()]
        model += [nn.Dropout(self.drpval)]
        model += [nn.Linear(self.out_dim1, self.out_dim2)]
        self.model = nn.Sequential(*model)
        self.nettype = nettype

    def forward(self, x):
        out = self.model(x)
        if self.debug:
            print('--- networks.py --> class ResNetClsNet --> forward() for {} ---'.format(self.nettype))
            kwargs = {'[Input shape of {}]'.format(self.nettype): x}
            print_feat_shape(**kwargs)

        return F.log_softmax(out, dim=1)