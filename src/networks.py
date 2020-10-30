import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import print_feat_shape
from torch.autograd import Function

class GradRevLayer(Function):
    @staticmethod
    def forward(ctx, x, da_lambda):
        ctx.da_lambda = da_lambda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.da_lambda
        return output, None


class CVPR2018Encoder(nn.Module):
    def __init__(self, config, debug):
        super(CVPR2018Encoder, self).__init__()
        # print('networks.py --> class CVPR2018Encoder --> __init__()')
        # print('-------------------------------')
        self.debug = debug
        inp_dim = config['input_dim']
        num_kernel = config['num_kernels']
        kernel_size = config['kernel_size']
        stride = config['stride']
        padding = config['padding']
        norm = config['norm']
        activ = config['activ']
        pad_type = config['pad_type']
        # print(' --- CVPR2018Encoder config ---')
        # print(config)
        self.block1 = FirstBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm, activ, pad_type)
        inp_dim = num_kernel
        num_kernel = num_kernel*2
        # print('--- networks.py --> class CVPR2018Encoder --> __init__() --> inp_dim {} num_kernel {} ---'.format(inp_dim, num_kernel))
        self.block2 = MiddleBlocks(inp_dim, num_kernel, kernel_size, stride, padding, norm, activ, pad_type)
        inp_dim = num_kernel
        # print('--- networks.py --> class CVPR2018Encoder -->  __init__() --> inp_dim {} num_kernel {} ---'.format(inp_dim, num_kernel))
        self.block3 = MiddleBlocks(inp_dim, num_kernel, kernel_size, stride, padding, norm, activ, pad_type)
        self.block4 = MiddleBlocks(inp_dim, num_kernel, kernel_size, stride, padding, norm, activ, pad_type)
        self.res = FeatMapsResize(config['res_dim'])

    def forward(self, x):
        out_b1 = self.block1(x)
        out_b2 = self.block2(out_b1)
        out_b3 = self.block3(out_b2)
        out_b4 = self.block4(out_b3)
        out_b2_res = self.res(out_b2)
        out_b3_res = self.res(out_b3)
        out_b4_res = self.res(out_b4)
        if self.debug:
            print('--- networks.py --> class CVPR2018Encoder --> forward() ---')
            kwargs = {'out_b1':out_b1, 'out_b2':out_b2, 'out_b3':out_b3, 'out_b4':out_b4, 'out_b2_res':out_b2_res, 'out_b3_res':out_b3_res, 'out_b4_res':out_b4_res}
            print_feat_shape(**kwargs)
        return torch.cat((out_b2_res, out_b3_res, out_b4_res), 1)


class CVPR2018EncoderV2(nn.Module):
    def __init__(self, config, debug):
        super(CVPR2018EncoderV2, self).__init__()
        # print('networks.py --> class CVPR2018EncoderV2 --> __init__()')
        # print('-------------------------------')
        self.debug = debug
        inp_dim = config['input_dim']
        num_kernel = config['num_kernels']
        kernel_size = config['kernel_size']
        stride = config['stride']
        padding = config['padding']
        norm = config['norm']
        activ = config['activ']
        pad_type = config['pad_type']
        # print(' --- CVPR2018EncoderV2 config ---')
        # print(config)
        self.block1 = FirstBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm, activ, pad_type)
        inp_dim = num_kernel
        num_kernel = num_kernel*2
        # print('--- networks.py --> class CVPR2018EncoderV2 --> __init__() --> inp_dim {} num_kernel {} ---'.format(inp_dim, num_kernel))
        self.block2 = MiddleBlocks(inp_dim, num_kernel, kernel_size, stride, padding, norm, activ, pad_type)
        inp_dim = num_kernel
        # print('--- networks.py --> class CVPR2018EncoderV2 -->  __init__() --> inp_dim {} num_kernel {} ---'.format(inp_dim, num_kernel))
        self.block3 = MiddleBlocks(inp_dim, num_kernel, kernel_size, stride, padding, norm, activ, pad_type)
        self.block4 = MiddleBlocks(inp_dim, num_kernel, kernel_size, stride, padding, norm, activ, pad_type)

    def forward(self, x):
        out_b1 = self.block1(x)
        out_b2 = self.block2(out_b1)
        out_b3 = self.block3(out_b2)
        out_b4 = self.block4(out_b3)
        if self.debug:
            # print('--- networks.py --> class CVPR2018EncoderV2 --> forward() ---')
            kwargs = {'out_b1': out_b1, 'out_b2': out_b2, 'out_b3': out_b3, 'out_b4': out_b4}
            print_feat_shape(**kwargs)
        return out_b2, out_b3, out_b4


class ResizeAndStackFeatureMaps(nn.Module):
    def __init__(self, res_dim, debug):
        super(ResizeAndStackFeatureMaps, self).__init__()
        # print('networks.py --> class ResizeAndStackFeatureMaps --> __init__()')
        self.res = FeatMapsResize(res_dim)
        self.debug = debug

    def forward(self, out_b2, out_b3, out_b4):
        out_b2_res = self.res(out_b2)
        out_b3_res = self.res(out_b3)
        out_b4_res = self.res(out_b4)
        if self.debug:
            print('--- networks.py --> class ResizeAndStackFeatureMaps --> forward() ---')
            kwargs = {'out_b2_res': out_b2_res, 'out_b3_res': out_b3_res, 'out_b4_res': out_b4_res}
            print_feat_shape(**kwargs)
        return torch.cat((out_b2_res, out_b3_res, out_b4_res), 1)


class FeatMapsResize(nn.Module):
    def __init__(self, res_dim):
        super(FeatMapsResize, self).__init__()
        self.resmap = nn.AdaptiveAvgPool2d(res_dim)
        # print('--- networks.py --> class FeatMapsResize -->  __init__() --> res_dim {} ---'.format(res_dim))

    def forward(self, x):
        x = self.resmap(x)
        return x


class FirstBlock(nn.Module):
    def __init__(self, inp_dim, num_kernel, kernel_size, stride, padding, norm, activ, pad_type):
        super(FirstBlock, self).__init__()
        # print('--- networks.py --> class FirstBlock --> __init__() ---')
        model = []
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class MiddleBlocks(nn.Module):
    def __init__(self, inp_dim, num_kernel, kernel_size, stride, padding, norm, activ, pad_type):
        super(MiddleBlocks, self).__init__()
        # print('--- networks.py --> class MiddleBlocks --> __init__() ---')
        model = []
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        inp_dim = num_kernel
        num_kernel = 196
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        inp_dim = num_kernel
        num_kernel = 128
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*model)
        model += [nn.MaxPool2d(2, 2)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ConvHead(nn.Module):
    def __init__(self, config, debug, nettype):
        super(ConvHead, self).__init__()
        self.nettype = nettype
        self.debug = debug
        # print(' --- networks.py --> class ConvHead -->  __init__() for {} ---'.format(self.nettype))
        inp_dim = config['input_dim']
        num_kernel = config['num_kernels']
        kernel_size = config['kernel_size']
        stride = config['stride']
        padding = config['padding']
        norm = config['norm']
        activ = config['activ']
        pad_type = config['pad_type']
        # print(' --- ConvHead config for {}---'.format(nettype))
        # print(config)
        # print('-------------------------------')
        model = []
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        inp_dim = num_kernel
        num_kernel = 64
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        inp_dim = num_kernel
        num_kernel = 1
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        if nettype == 'ANet' or nettype == 'DANet':
            model += [nn.MaxPool2d(2, 2)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.debug:
            print(' --- networks.py --> class ConvHead -->  forward() for {} ---'.format(self.nettype))
        return self.model(x)

class New_convhead(nn.Module):
    def __init__(self):
        super(New_convhead, self).__init__()
        # print(' --- networks.py --> class ConvHead -->  __init__() for {} ---'.format(self.nettype))
        inp_dim = 384
        num_kernel = 256
        kernel_size = 3
        stride = 1
        padding = 1
        norm = 'bn'
        activ = 'elu'
        pad_type = 'zero'
        self.use_bias = True
        model = []
        model += [nn.Conv2d(inp_dim, num_kernel, kernel_size, stride, bias=self.use_bias), nn.BatchNorm2d(num_kernel),nn.ELU()]
        model += [nn.AdaptiveAvgPool2d((8,8))]
        model += [nn.Conv2d(num_kernel, num_kernel, kernel_size, stride, bias=self.use_bias), nn.BatchNorm2d(num_kernel),nn.ELU()]
        model += [nn.AdaptiveAvgPool2d((2,2))]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        
        return self.model(x)


class ConvHead4ResNet(nn.Module):
    def __init__(self, config, debug, nettype):
        super(ConvHead4ResNet, self).__init__()
        self.nettype = nettype
        self.debug = debug
        # print(' --- networks.py --> class ConvHead4ResNet -->  __init__() for {} ---'.format(self.nettype))
        inp_dim = config['input_dim']
        num_kernel = config['num_kernels']
        kernel_size = config['kernel_size']
        stride = config['stride']
        padding = config['padding']
        norm = config['norm']
        activ = config['activ']
        pad_type = config['pad_type']
        # print(' --- ConvHead4ResNet config for {}---'.format(nettype))
        # print(config)
        # print('-------------------------------')
        model = []
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        inp_dim = num_kernel
        num_kernel = 64
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        inp_dim = num_kernel
        num_kernel = 1
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        if nettype == 'ANet' or nettype == 'DANet':
            model += [nn.MaxPool2d(2, 2)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.debug:
            print(' --- networks.py --> class ConvHead4ResNet -->  forward() for {} ---'.format(self.nettype))
        return self.model(x)


class ClsCondDomainNetHeads(nn.Module):
    def __init__(self, config, debug, nettype=None,use_cvpr_2019_bbone = 0):
        super(ClsCondDomainNetHeads, self).__init__()
        '''
        Individual head of DIB layer (only 1 fc)
        '''
        # print(' --- networks.py --> class ClsCondDomainNetHeads -->  __init__() for {} ---'.format(nettype))
        self.debug = debug
        self.nettype = nettype
        # self.inp_dim = config['inp_dim']
        # self.out_dim1 = config['out_dim1']
        if use_cvpr_2019_bbone == 0:
            self.inp_dim = 1024
            self.out_dim1 = 3
        elif use_cvpr_2019_bbone == 1:
            self.inp_dim = 512
            self.out_dim1 = 3
        self.cost_func = config['cost_func']
        # print('[inp_dim:{}] [out_dim1:{}] [loss: {}]'.format(self.inp_dim, self.out_dim1, self.cost_func))
        # print('-------------------------------')
        model = []
        model += [nn.Linear(self.inp_dim, self.out_dim1)]
        self.model = nn.Sequential(*model)
        self.nettype = nettype

    def forward(self, x):
        out = self.model(x)
        if self.debug:
            # print('--- networks.py --> class ClsCondDomainNetHeads --> forward() for {} ---'.format(self.nettype))
            kwargs = {'[Input shape of {}]'.format(self.nettype): x}
            print_feat_shape(**kwargs)
        if self.cost_func == 'bce':
            return out
        elif self.cost_func == 'softmax':
            return F.log_softmax(out, dim=1)
        else:
            pass
        return out # --- These two are used in DG


class ClsCondDomainNet(nn.Module):
    def __init__(self, config, debug, nettype=None,use_cvpr_2019_bbone = 0):
        super(ClsCondDomainNet, self).__init__()
        '''
        DIB network layers before live and spoof heads
        2 FC layers
        '''
        # print(' --- networks.py --> class ClsCondDomainNet -->  __init__() for {} ---'.format(nettype))
        self.debug = debug
        self.nettype = nettype
        # self.inp_dim = config['inp_dim']
        # self.out_dim1 = config['out_dim1']
        # self.drpval = config['dropoutv']
        if use_cvpr_2019_bbone == 0:
            self.inp_dim = 1024
            self.out_dim1 = 1024
        elif use_cvpr_2019_bbone == 1:
            self.inp_dim =512
            self.out_dim1 = 512
        self.drpval = 0.2
        # print('[inp_dim:{}] [out_dim1:{}]'.format(self.inp_dim, self.out_dim1))
        # print('-------------------------------')
        model = []
        model += [nn.Linear(self.inp_dim, self.out_dim1)]
        model += [nn.ReLU(True)]
        model += [nn.Dropout(self.drpval)]
        model += [nn.Linear(self.out_dim1, self.out_dim1)]
        model += [nn.ReLU(True)]
        model += [nn.Dropout(self.drpval)]
        self.model = nn.Sequential(*model)
        self.nettype = nettype

    def forward(self, x):
        x = x.view(-1,self.inp_dim)
        out = self.model(x)
        if self.debug:
            print('--- networks.py --> class ClsCondDomainNet --> forward() for {} ---'.format(self.nettype))
            kwargs = {'[Input shape of {}]'.format(self.nettype): x}
            print_feat_shape(**kwargs)
        return out # --- These two are used in DG


# class ClsPriorNormDomainNet(nn.Module):
#     def __init__(self, config, debug, nettype=None):
#         super(ClsPriorNormDomainNet, self).__init__()
#         # print(' --- networks.py --> class ClsPriorNormDomainNet -->  __init__() for {} ---'.format(nettype))
#         self.debug = debug
#         self.nettype = nettype
#         self.inp_dim = config['inp_dim']
#         self.out_dim1 = config['out_dim1']
#         self.out_dim2 = config['out_dim2']
#         self.cost_func = config['cost_func']
#         self.drpval = config['dropoutv']
#         # print('[inp_dim:{}] [out_dim1:{}] [out_dim2: {}] [loss: {}]'.format(self.inp_dim, self.out_dim1, self.out_dim2, self.cost_func))
#         # print('-------------------------------')
#         model = []
#         model += [nn.Linear(self.inp_dim, self.out_dim1)]
#         model += [nn.ReLU(True)]
#         model += [nn.Dropout(self.drpval)]
#         model += [nn.Linear(self.out_dim1, self.out_dim1)]
#         model += [nn.ReLU(True)]
#         model += [nn.Dropout(self.drpval)]
#         model += [nn.Linear(self.out_dim1, self.out_dim2)]
#         self.model = nn.Sequential(*model)
#         self.nettype = nettype

#     def forward(self, x):
#         out = self.model(x)
#         if self.debug:
#             print('--- networks.py --> class ClsPriorNormDomainNet --> forward() for {} ---'.format(self.nettype))
#             kwargs = {'[Input shape of {}]'.format(self.nettype): x}
#             print_feat_shape(**kwargs)
#         if self.cost_func == 'bce':
#             return out
#         elif self.cost_func == 'softmax':
#             return F.log_softmax(out, dim=1)
#         else:
#             pass


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

# class DAClsNet4ResNet(nn.Module):
#     def __init__(self, config, debug, nettype=None):
#         super(DAClsNet4ResNet, self).__init__()
#         # print(' --- networks.py --> class DAClsNet4ResNet -->  __init__() for {} ---'.format(nettype))
#         self.debug = debug
#         self.nettype = nettype
#         self.inp_dim = config['inp_dim']
#         self.out_dim1 = config['out_dim1']
#         self.out_dim2 = config['out_dim2']
#         self.cost_func = config['cost_func']
#         self.drpval = config['dropoutv']
#         # print('[inp_dim:{}] [out_dim1:{}] [out_dim2: {}] [loss: {}]'.format(self.inp_dim, self.out_dim1, self.out_dim2, self.cost_func))
#         # print('-------------------------------')
#         model = []
#         model += [nn.Linear(self.inp_dim, self.out_dim1)]
#         model += [nn.ReLU(True)]
#         model += [nn.Dropout(self.drpval)]
#         model += [nn.Linear(self.out_dim1, self.out_dim1)]
#         model += [nn.ReLU(True)]
#         model += [nn.Dropout(self.drpval)]
#         model += [nn.Linear(self.out_dim1, self.out_dim2)]
#         self.model = nn.Sequential(*model)
#         self.nettype = nettype

#     def forward(self, x):
#         out = self.model(x)
#         if self.debug:
#             print('--- networks.py --> class DAClsNet4ResNet --> forward() for {} ---'.format(self.nettype))
#             kwargs = {'[Input shape of {}]'.format(self.nettype): x}
#             print_feat_shape(**kwargs)
#         if self.cost_func == 'bce':
#             return out
#         elif self.cost_func == 'softmax':
#             return F.log_softmax(out, dim=1)
#         else:
#             pass


class ClsNet(nn.Module):
    def __init__(self, config, debug, nettype=None):
        super(ClsNet, self).__init__()
        # print(' --- networks.py --> class ClsNet -->  __init__() for {} ---'.format(nettype))
        self.debug = debug
        self.nettype = nettype
        self.inp_dim = config['inp_dim']
        self.out_dim1 = config['out_dim1']
        self.out_dim2 = config['out_dim2']
        self.cost_func = config['cost_func']
        
        model = []
        model += [nn.Linear(self.inp_dim, self.out_dim1)]
        model += [nn.ReLU(True)]
        model += [nn.Dropout()]
        model += [nn.Linear(self.out_dim1, self.out_dim1)]
        model += [nn.ReLU(True)]
        model += [nn.Dropout()]
        model += [nn.Linear(self.out_dim1, self.out_dim2)]
        self.model = nn.Sequential(*model)
        self.nettype = nettype

    def forward(self, x):
        if self.debug:
            print('--- networks.py --> class ClsNet --> forward() for {} ---'.format(self.nettype))
            kwargs = {'x':x}
            print_feat_shape(**kwargs)
            xsize = x.size()
            assert(self.inp_dim == xsize[2]*xsize[3])
        
        x = x.view(-1, self.inp_dim)
        out = self.model(x)
        return F.log_softmax(out, dim=1)

class ConvHead3(nn.Module):
    def __init__(self, config, debug, nettype):
        super(ConvHead3, self).__init__()
        self.nettype = nettype
        self.debug = debug
        # print(' --- networks.py --> class ConvHead3 -->  __init__() for {} ---'.format(self.nettype))
        inp_dim = config['input_dim']
        inp_dim2 = config['input_dim2']
        kernel_size = config['kernel_size']
        stride = config['stride']
        padding = config['padding']
        norm = config['norm']
        activ = config['activ']
        pad_type = config['pad_type']
        # print(' --- ConvHead3 config for {}---'.format(nettype))
        # print(config)
        # print('-------------------------------')
        model = []
        model += [Conv2dBlock(inp_dim, inp_dim2, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        model += [nn.MaxPool2d(2, 2)]
        model += [Conv2dBlock(inp_dim2, inp_dim2, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        model += [nn.MaxPool2d(2, 2)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.debug:
            print(' --- networks.py --> class ConvHead3 -->  forward() for {} ---'.format(self.nettype))
        return self.model(x)


class ClsNet3(nn.Module):
    def __init__(self, config, debug, nettype=None):
        super(ClsNet3, self).__init__()
        # print(' --- networks.py --> class ClsNet3 -->  __init__() for {} ---'.format(nettype))
        self.debug = debug
        self.nettype = nettype
        self.inp_dim = config['inp_dim']
        self.out_dim1 = config['out_dim1']
        self.out_dim2 = config['out_dim2']
        self.cost_func = config['cost_func']
        # print('[inp_dim:{}] [out_dim1:{}] [out_dim2: {}] [loss: {}]'.format(self.inp_dim, self.out_dim1, self.out_dim2, self.cost_func))
        # print('-------------------------------')
        model = []
        model += [nn.Linear(self.inp_dim, self.out_dim1)]  # original  (self.inp_dim, self.inp_dim)
        model += [nn.ReLU(True)]
        model += [nn.Dropout()]
        model += [nn.Linear(self.out_dim1, self.out_dim1)]  # original  (self.inp_dim, self.out_dim1)
        model += [nn.ReLU(True)]
        model += [nn.Dropout()]
        model += [nn.Linear(self.out_dim1, self.out_dim2)]
        self.model = nn.Sequential(*model)
        self.nettype = nettype

    def forward(self, x):
        if self.debug:
            print('--- networks.py --> class ClsNet3 --> forward() for {} ---'.format(self.nettype))
            kwargs = {'x':x}
            print_feat_shape(**kwargs)
        x = x.view(-1, self.inp_dim)
        out = self.model(x)
        if self.cost_func == 'bce':
            return out
        elif self.cost_func == 'softmax':
            return F.log_softmax(out, dim=1)
        else:
            pass


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


# class AdaptiveInstanceNorm2d(nn.Module):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1):
#         super(AdaptiveInstanceNorm2d, self).__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.weight = None
#         self.bias = None
#         self.register_buffer('running_mean', torch.zeros(num_features))
#         self.register_buffer('running_var', torch.ones(num_features))

#     def forward(self, x):
#         assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
#         b, c = x.size(0), x.size(1)
#         running_mean = self.running_mean.repeat(b)
#         running_var = self.running_var.repeat(b)
#         x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
#         out = F.batch_norm(
#             x_reshaped, running_mean, running_var, self.weight, self.bias,
#             True, self.momentum, self.eps)
#         return out.view(b, c, *x.size()[2:])

#     def __repr__(self):
#         return self.__class__.__name__ + '(' + str(self.num_features) + ')'


# class LayerNorm(nn.Module):
#     def __init__(self, num_features, eps=1e-5, affine=True):
#         super(LayerNorm, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps

#         if self.affine:
#             self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
#             self.beta = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x):
#         shape = [-1] + [1] * (x.dim() - 1)
#         # print(x.size())
#         if x.size(0) == 1:
#             # These two lines run much faster in pytorch 0.4 than the two lines listed below.
#             mean = x.view(-1).mean().view(*shape)
#             std = x.view(-1).std().view(*shape)
#         else:
#             mean = x.view(x.size(0), -1).mean(1).view(*shape)
#             std = x.view(x.size(0), -1).std(1).view(*shape)

#         x = (x - mean) / (std + self.eps)

#         if self.affine:
#             shape = [1, -1] + [1] * (x.dim() - 2)
#             x = x * self.gamma.view(*shape) + self.beta.view(*shape)
#         return x


class ConvHead2(nn.Module):
    def __init__(self, config, nettype):
        super(ConvHead2, self).__init__()
        # print(' --- networks.py --> class ConvHead2 -->  __init__() ---')
        inp_dim = config['input_dim']
        num_kernel = config['num_kernels']
        kernel_size = config['kernel_size']
        stride = config['stride']
        padding = config['padding']
        norm = config['norm']
        activ = config['activ']
        pad_type = config['pad_type']
        # print(' --- ConvHead2 config for {}---'.format(nettype))
        # print(config)
        # print('-------------------------------')
        model = []
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        inp_dim = num_kernel
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        model += [nn.MaxPool2d(2, 2)]
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        model += [nn.MaxPool2d(2, 2)]
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        model += [Conv2dBlock(inp_dim, num_kernel, kernel_size, stride, padding, norm=norm, activation=activ, pad_type=pad_type)]
        model += [nn.MaxPool2d(2, 2)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# --- New Network Structures: 
class Cls_lstm(nn.Module):
    def __init__(self, batch_size, nettype=None, inp_dim = 2048, out_dim1 = 512):
        super(Cls_lstm, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        self.batch_size = batch_size
        self.inp_dim = inp_dim
        self.num_layers = 1 # --- using one layer lstm
        # self.hidden_size = 256
        self.hidden_size = int(self.inp_dim/8)
        
        
        self.nettype = nettype
        self.lstm = nn.LSTM(input_size = inp_dim, hidden_size = self.hidden_size,num_layers = self.num_layers, batch_first = True)
        self.fc_1 = nn.Linear(inp_dim,out_dim1)
        self.fc = nn.Linear(out_dim1,2)
        self.calc_total_param()
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def calc_total_param(self):
        self.net_param = list(self.lstm.parameters())
        para_sum = 0
        for p in self.net_param:
            para_sum += torch.prod(torch.tensor(p.size()))
        print('TOTAL PARAMS IN LSTM: {}'.format(para_sum))

    def forward(self, x, return_feat = False):
        
        xsize = x.size()
        b_size = x.size(0)
        h0 = torch.randn(self.num_layers,b_size,self.hidden_size).to(self.device) # --- move hidden vectors to device
        c0 = torch.randn(self.num_layers,b_size,self.hidden_size).to(self.device)
        
        x = x.view(b_size,-1,self.inp_dim) # --- b_size is either 8 or 4, depending on whether it is Train procedure or 
        
        
        y,(hn,cn) = self.lstm(x,(h0,c0))
        
        y = y.contiguous().view(b_size,-1) # --- y: 8*2048 or 4* 2048
        
        out = self.fc_1(y)
        
        out = self.fc(out) # --- out: 8*2 or 4*2 

        if return_feat:
            return F.log_softmax(out,dim=1),y 
        else:
            return F.log_softmax(out,dim=1)

# class DAClsNet4lstm(nn.Module):
#     def __init__(self, config, debug, nettype=None):
#         super(DAClsNet4lstm, self).__init__()
#         # print(' --- networks.py --> class DAClsNet4ResNet -->  __init__() for {} ---'.format(nettype))
#         self.debug = debug
#         self.nettype = nettype
#         self.inp_dim = config['inp_dim']
#         self.out_dim1 = config['out_dim1']
#         self.out_dim2 = config['out_dim2']
#         self.cost_func = config['cost_func']
#         self.drpval = config['dropoutv']
#         # print('[inp_dim:{}] [out_dim1:{}] [out_dim2: {}] [loss: {}]'.format(self.inp_dim, self.out_dim1, self.out_dim2, self.cost_func))
#         # print('-------------------------------')
#         model = []
#         model += [nn.Linear(self.inp_dim, self.out_dim1)]  # original  (self.inp_dim, self.inp_dim)
#         model += [nn.ReLU(True)]
#         model += [nn.Dropout(self.drpval)]
#         model += [nn.Linear(self.out_dim1, self.out_dim1)]  # original  (self.inp_dim, self.out_dim1)
#         model += [nn.ReLU(True)]
#         model += [nn.Dropout(self.drpval)]
#         model += [nn.Linear(self.out_dim1, self.out_dim2)]
#         self.model = nn.Sequential(*model)
#         self.nettype = nettype

#     def forward(self, x):
#         out = self.model(x)
#         if self.debug:
#             print('--- networks.py --> class DAClsNet4lstm --> forward() for {} ---'.format(self.nettype))
#             kwargs = {'[Input shape of {}]'.format(self.nettype): x}
#             print_feat_shape(**kwargs)
#         if self.cost_func == 'bce':
#             return out
#         elif self.cost_func == 'softmax':
#             return F.log_softmax(out, dim=1)
#         else:
#             pass


