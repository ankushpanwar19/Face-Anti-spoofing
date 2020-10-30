import os
import yaml
from torch.optim import lr_scheduler
import torch.nn.init as init
import math
from torchvision import transforms
# from data import ImageFilelist
from torch.utils.data import DataLoader
# from data_loader_dnet import get_loader
import sys
import torch
import statistics
# from get_examples_labels_siw import get_examples_labels as get_siw_dataset
from get_examples_labels_oulu import get_examples_labels as get_oulu_dataset
from get_examples_labels_replayattack import get_examples_labels as get_repatack_dataset
# from get_examples_labels_replaymobile import get_examples_labels as get_repmob_dataset
from get_examples_labels_casia import get_examples_labels as get_casia_dataset
from get_examples_labels_msu import get_examples_labels as get_msu_dataset

from data_loader_anet import get_loader as get_loader_all
from data_loader_anet import get_MOT_loader
import pickle

def load_Trainer_DNetANetDA(trainer, model_name, device, net_type, fusion_net_arch):
    state_dict = torch.load(model_name, map_location=device)
    trainer.gen.load_state_dict(state_dict['gen'])
    trainer.ResStack.load_state_dict(state_dict['ResStack'])
    trainer.DNet.load_state_dict(state_dict['DNet'])
    trainer.ANet.load_state_dict(state_dict['ANet'])
    trainer.ANetClsNet.load_state_dict(state_dict['ANetClsNet'])
    trainer.DANet.load_state_dict(state_dict['DANet'])
    trainer.DAClsNet.load_state_dict(state_dict['DAClsNet'])
    return trainer


def load_model_older_version(trainer, model_name, device, net_type):
    state_dict = torch.load(model_name, map_location=device)
    trainer.gen.load_state_dict(state_dict['gen'])
    if net_type in 'anet' or net_type in 'fusion':
        trainer.ANet.load_state_dict(state_dict['anet'])
        trainer.ClsNet.load_state_dict(state_dict['cnet'])
    if net_type in 'dnet' or net_type in 'fusion':
        trainer.DNet.load_state_dict(state_dict['dnet'])
    return trainer


def load_model(trainer, model_name, device, net_type, fusion_net_arch):
    state_dict = torch.load(model_name, map_location=device)
    if net_type in 'resnet':
        trainer.gen.load_state_dict(state_dict['gen'])
    else:
        trainer.gen.load_state_dict(state_dict['gen'])
        if net_type in 'anet' or net_type in 'fusion':
            trainer.ANet.load_state_dict(state_dict['anet'])
            trainer.ClsNet.load_state_dict(state_dict['cnet'])
        if net_type in 'dnet' or net_type in 'fusion' or net_type in 'dadpnet':
            trainer.DNet.load_state_dict(state_dict['dnet'])
            if fusion_net_arch == 2:
                trainer.ClsNet2.load_state_dict(state_dict['cnet2'])
        if fusion_net_arch == 1 or fusion_net_arch == 2:
            trainer.ANetResStack.load_state_dict(state_dict['ANetResStack'])
            trainer.DNetResStack.load_state_dict(state_dict['DNetResStack'])
        if net_type in 'dadpnet':
            trainer.ResStack.load_state_dict(state_dict['ResStack'])
    return trainer

def print_feat_shape(**kwargs):
    for key, value in kwargs.items():
        print('{}: {}'.format(key,value.shape))

def str2bool(v):
    return v.lower() in ('true')


def get_data_loader_da(config, mode, small_trainset=False, drop_last=False):
    machine = config['machine']
    dataset = config['da_target_dataset']['train_dataset']
    protocol = config['da_target_dataset'][dataset]['protocol']
    split = config['da_target_dataset'][dataset]['split']
    dataset_path = config['da_target_dataset'][dataset]['dataset_path_machine{}'.format(machine)]
    img_path = config['da_target_dataset'][dataset]['img_path_machine{}'.format(machine)]
    depth_map_path = config['da_target_dataset'][dataset]['depth_map_path_machine{}'.format(machine)]

    pickle_fname = config['pickle_fname']
    num_workers = config['num_workers']
    inp_dim = (config['inp_dim'], config['inp_dim'])
    depth_map_size = (config['depth_map_size'], config['depth_map_size'])
    sel_every = config['sel_every']
    sel_these_many = config['sel_these_many']
    app_feats = config['app_feats']
    batch_size = config['batch_size']
    eval_on_oulu_devset = config['eval_on_oulu_devset']
    print(' --- sampling {} examples,labels for dataset {} --- '.format(mode, dataset))
    print(' --- dataset read path {} ---'.format(dataset_path))
    num_exmps = None
    part = None
    labels = None
    if dataset == 'siw':
        part, labels, gtFlags, scores, num_exmps = get_siw_dataset(dataset_path, mode, protocol, split, sel_every, sel_these_many, pickle_fname, config['net_type'], small_trainset = small_trainset)
    elif dataset == 'oulu-npu':
        part, labels, gtFlags, scores, num_exmps = get_oulu_dataset(dataset_path, mode, protocol, split, sel_every, sel_these_many, img_path, config['net_type'], eval_on_oulu_devset, small_trainset = small_trainset)
    epoch_size = math.floor(num_exmps / batch_size)
    model_save_step = epoch_size
    print('mini-batch size [{}]; epoch-size [{}]; model_save_step [{}]'.format(batch_size, epoch_size, model_save_step))
    params = {'app_feats': app_feats,
              'batch_size': batch_size,
              'shuffle': True,
              'num_workers': num_workers,
              'res': inp_dim,
              'dataset_name': dataset,
              'depth_map_size': depth_map_size
              }
    my_dataloader = get_loader(part[mode], labels, img_path, mode, depth_map_path, drop_last, **params)
    return my_dataloader, epoch_size, model_save_step


def new_dictionary(initialization_path):
    init_dictionary_np = pickle.load(open(initialization_path, "rb"))
    init_dictionary = {}
    for key, value in init_dictionary_np.items():
        init_dictionary[key] = torch.tensor(value)
    return init_dictionary


def get_data_loader(config, dataset_name, mode, small_trainset = False, drop_last = False):
    data_loader = None
    epoch_size = None
    modelsave_step = None
    if config['dataloader_type'] == 'mydataloader':
        data_loader, epoch_size, modelsave_step = get_my_dataloader(config, dataset_name, mode, small_trainset = small_trainset, drop_last = drop_last)
    elif config['dataloader_type'] == 'unit':
        data_loader = get_unit_dataloader(config, config['transforms'])
    if not epoch_size or not modelsave_step:
        sys.exit('utils.py --> you are uisng get_unit_dataloader() which should return epoch_size, modelsave_step, update the code first!')
    return data_loader, epoch_size, modelsave_step


def get_dataset_id(dataset): # --- Modified add ra-ma & ca-ma
    datasetID = ''
    if dataset == 'siw':
        datasetID = 'si'
    elif dataset == 'oulu-npu':
        datasetID = 'ou'
    elif dataset == 'replay-attack':
        datasetID = 'ra'
    elif dataset == 'replay-mobile':
        datasetID = 'rm'
    elif dataset == 'casia':
        datasetID = 'ca'
    elif dataset == 'replay-attack-maged':
        datasetID = 'ra-ma'
    elif dataset == 'casia-maged':
        datasetID = 'ca-ma'
    elif dataset == 'msu':
        datasetID = 'ms'
    else:
        pass
    return datasetID

# --- added ra-ma and ca-ma --- 
def get_part_labels_all(config, config_dl, mode, small_trainset = False, drop_last = False, da_mode = None):
    machine = config['machine']
    net_type = config['net_type']
    pickle_fname = config['pickle_fname']
    num_cls = config_dl['num_cls']
    isFirst = True
    num_exmps_totoal = 0
    dataset_list = None
    if da_mode:
        if da_mode.split('_')[2] == 'source':
            dataset_list = config_dl['da_train_source_dataset_list']
        elif da_mode.split('_')[2] == 'target':
            dataset_list = config_dl['da_train_target_dataset_list']
        else:
            pass
    else:
        dataset_list = config_dl['train_datasets']
    for dataset in dataset_list:
        proto = config_dl[dataset]['proto']
        split = config_dl[dataset]['split']
        dataset_path = config_dl[dataset]['dataset_path_machine{}'.format(machine)]
        datasetID = get_dataset_id(dataset)
        if da_mode:
            sel_every = config_dl[dataset]['sel_every_mergeall']
        else:
            sel_every = config_dl[dataset]['sel_every']

        sel_these_many = config_dl[dataset]['sel_these_many']
        img_path = config_dl[dataset]['full_img_path_machine{}'.format(machine)]

        if datasetID == 'si':
            merge_all_dataset = config_dl[dataset]['merge_all_datasets']
            part, labels, _, _, num_exmps = \
                get_siw_dataset(dataset_path, mode, proto, split, sel_every, sel_these_many, pickle_fname, net_type,
                                small_trainset = small_trainset, datasetID = datasetID, num_cls = num_cls, merge_all_dataset = merge_all_dataset)
        elif datasetID == 'ou':
            eval_on_oulu_devset = config['eval_on_oulu_devset']
            part, labels, _, _, num_exmps = \
                get_oulu_dataset(dataset_path, mode, proto, split, sel_every, sel_these_many, img_path, net_type,
                                 eval_on_oulu_devset = eval_on_oulu_devset, datasetID = datasetID, num_cls = num_cls)
        elif datasetID == 'ra' or datasetID == 'ra-ma':
            part, labels, _, _, num_exmps = \
                get_repatack_dataset(dataset_path, mode, proto, split, sel_every, sel_these_many, img_path, net_type,
                                     datasetID = datasetID, num_cls = num_cls)
        elif datasetID == 'rm':
            part, labels, _, _, num_exmps = \
                get_repmob_dataset(dataset_path, mode, proto, split, sel_every, sel_these_many, img_path, net_type,
                                   datasetID = datasetID, num_cls = num_cls)
        elif datasetID == 'ca' or datasetID == 'ca-ma':
            part, labels, _, _, num_exmps = \
                get_casia_dataset(dataset_path, mode, proto, split, sel_every, sel_these_many, img_path, net_type,
                                  datasetID = datasetID, num_cls = num_cls)

        elif datasetID == 'ms':
            print('IMG PATH IN UTILS: ',img_path)
            part, labels, _, _, num_exmps = \
                get_msu_dataset(dataset_path, mode, proto, split, sel_every, sel_these_many, img_path, net_type,
                                  datasetID = datasetID, num_cls = num_cls)

        else:
            pass
        if not isFirst:
            part_all[mode] += part[mode]
            for key, value in labels.items():
                labels_all[key] = value
        else:
            part_all = part
            labels_all = labels

        num_exmps_totoal += num_exmps
        isFirst = False
    return part_all, labels_all, num_exmps_totoal


def get_data_loader_all(config, config_dl, dataset_name, mode, small_trainset = False, drop_last = False, da_mode = None):
    machine = config['machine']
    dataset = dataset_name
    num_workers = config['num_workers']
    inp_dim = (config['inp_dim'], config['inp_dim'])
    depth_map_size = (config['depth_map_size'], config['depth_map_size'])
    app_feats = config['app_feats']
    batch_size = config['batch_size']
    print(' --- sampling {} examples,labels for dataset {} --- '.format(mode, dataset))
    part_all, labels_all, num_exmps = get_part_labels_all(config, config_dl, mode, small_trainset = small_trainset, drop_last = drop_last, da_mode = da_mode)
    epoch_size = math.floor(num_exmps / batch_size)
    model_save_step = epoch_size
    print('[totoal_training_examples: {}]  [mini-batch size: {}];  [epoch-size: {}];  [model_save_step: {}]'.format(num_exmps, batch_size, epoch_size, model_save_step))
    params = {'app_feats': app_feats,
              'batch_size': batch_size,
              'shuffle': True,
              'num_workers': num_workers,
              'res': inp_dim,
              'dataset_name': dataset,
              'depth_map_size': depth_map_size,
              'net_type': config['net_type']
              }
    my_dataloader = get_loader_all(machine, config_dl, part_all[mode], labels_all, mode, drop_last, **params)
    return my_dataloader, epoch_size, model_save_step

def get_MOT_loader_all(config, config_dl, dataset_name, mode, small_trainset = False, drop_last = False, da_mode = None):
    # data_loader = None
    # epoch_size = None
    # modelsave_step = None

    machine = config['machine']
    dataset = dataset_name
    print('DATASET NAME:{}'.format(dataset_name))
    # split = config['split']
    # dataset_path = config['dataset_path']
    # img_path = config['img_path']
    # depth_map_path = config['depth_map_path']

    num_workers = config['num_workers']
    inp_dim = (config['inp_dim'], config['inp_dim'])
    depth_map_size = (config['depth_map_size'], config['depth_map_size'])
    sel_every = config['sel_every']
    sel_these_many = config['sel_these_many']
    app_feats = config['app_feats']
    if config['net_type'] == 'lstmmot':
        batch_size = config['batch_size_lstm']
    elif config['net_type'] == 'artmot':
        batch_size = config['batch_size_art']

    # eval_on_oulu_devset = config['eval_on_oulu_devset']

    print(' --- sampling {} examples,labels for dataset {} --- '.format(mode, dataset))
    # print(' --- dataset read path {} ---'.format(dataset_path))

    ## this one only gets lists for all image paths
    part_all, labels_all, num_exmps = get_part_labels_all(config, config_dl, mode, small_trainset = small_trainset, drop_last = drop_last, da_mode = da_mode)

    ## epoch_size = math.floor(num_exmps / batch_size)
    ## model_save_step = epoch_size
    

    params = {'app_feats': app_feats,
              'batch_size': batch_size,
              'shuffle': True,
              'num_workers': num_workers,
              'res': inp_dim,
              'dataset_name': dataset,
              'depth_map_size': depth_map_size,
              'net_type': config['net_type']
              }

    ## THIS IS THE PART WHERE YOU GET REAL DATALOADER
    my_dataloader = get_MOT_loader(machine, config_dl, part_all[mode], labels_all, mode, drop_last, **params)
    epoch_size = len(my_dataloader)
    model_save_step = epoch_size

    print('[totoal_training_img_examples: {}]  [mini-batch size: {}];  [epoch-size: {}];  [model_save_step: {}]'.format(num_exmps, batch_size, epoch_size, model_save_step))
    print(' --- NOTE: You can set the value of [snapshot_save_iter] in configs.yaml a value around [model_save_step]  ---')
    return my_dataloader, epoch_size, model_save_step


def get_my_dataloader(config, dataset_name, mode, small_trainset = False, drop_last = False):
    dataset = dataset_name
    protocol = config['protocol']
    split = config['split']
    dataset_path = config['dataset_path']
    pickle_fname = config['pickle_fname']
    img_path = config['img_path']
    depth_map_path = config['depth_map_path']
    num_workers = config['num_workers']
    inp_dim = (config['inp_dim'], config['inp_dim'])
    depth_map_size = (config['depth_map_size'], config['depth_map_size'])
    sel_every = config['sel_every']
    sel_these_many = config['sel_these_many']
    app_feats = config['app_feats']
    batch_size = config['batch_size']
    eval_on_oulu_devset = config['eval_on_oulu_devset']
    print(' --- sampling {} examples,labels for dataset {} --- '.format(mode, dataset))
    print(' --- dataset read path {} ---'.format(dataset_path))
    num_exmps = None
    part = None
    labels = None
    if dataset == 'siw':
        part, labels, gtFlags, scores, num_exmps = get_siw_dataset(dataset_path, mode, protocol, split, sel_every, sel_these_many, pickle_fname, config['net_type'], small_trainset = small_trainset)
    elif dataset == 'oulu-npu':
        part, labels, gtFlags, scores, num_exmps = get_oulu_dataset(dataset_path, mode, protocol, split, sel_every, sel_these_many, img_path, config['net_type'], eval_on_oulu_devset, small_trainset = small_trainset)
    epoch_size = math.floor(num_exmps / batch_size)
    model_save_step = epoch_size
    print('mini-batch size [{}]; epoch-size [{}]; model_save_step [{}]'.format(batch_size, epoch_size, model_save_step))
    print(' --- NOTE: [model_save_step] is not in use, instead use the [snapshot_save_iter] option in configs.yaml ---')

    params = {'app_feats': app_feats,
              'batch_size': batch_size,
              'shuffle': True,
              'num_workers': num_workers,
              'res': inp_dim,
              'dataset_name': dataset,
              'depth_map_size': depth_map_size
              }
    my_dataloader = get_loader(part[mode], labels, img_path, mode, depth_map_path, drop_last, **params)
    return my_dataloader, epoch_size, model_save_step


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('utils.py --> get_model_list() --> folder {} does not exist !'.format(dirname))
        sys.exit()
    print('dir name: ',dirname)
    print(os.listdir(dirname))
    net_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    print('net models ',net_models)
    if net_models is None:
        print('utils.py --> get_model_list() --> net_models is None !')
        sys.exit()
    net_models.sort()
    last_model_name = net_models[-1]
    return last_model_name


def get_exp_name(sub_base_path, net_type):
    if os.path.exists(sub_base_path) is False:
        print('utils.py --> get_exp_name() --> folder {} does not exist !'.format(sub_base_path))
        sys.exit()
   
    existing_exp_names = [f for f in os.listdir(sub_base_path) if net_type in f and 'debug' not in f]
    existing_exp_names.sort()
    print(existing_exp_names)
    if len(existing_exp_names) == 0:
        return '{}_exp_{:03d}'.format(net_type, 0)
    else:
        new_exp_id = int(existing_exp_names[-1].split('_')[2]) + 1
        return '{}_exp_{:03d}'.format(net_type, new_exp_id)


def write_loss(iterations, epoc_cnt, trainer, train_writer, exp_name):
    members = [attr for attr in dir(trainer) if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    print('members: {}'.format(members))
    for m in members:
        train_writer.add_scalar('network/{}'.format(m), getattr(trainer, m), iterations + 1)
        # print('[{}: {:.6f}] '.format(m, getattr(trainer, m)), end=' ') 
        print('[{}: {:.6f}] '.format(m, getattr(trainer, m)))

    print('[iter: {}] [epoch: {}] [{}]'.format(iterations + 1, epoc_cnt + 1, exp_name))



def write2tensorBoardX2(test_dataset, net_type, eval_type, config, iterations, train_writer, best_models):
    strs2match = None
    if 'dnet' in net_type or 'anet' in net_type:
        strs2match = ['eval']
    elif 'fusion' in net_type or 'dadpnet' in net_type:
        strs2match = ['eval_anet', 'eval_dnet', 'eval_fusion']
    protos_splits = config['protos_splits'][test_dataset]
    plot_dict1 = {}
    for str2match in strs2match:
        plot_dict2 = {}
        for proto, splits in protos_splits.items():
            acers = []
            if proto == 2 or proto == 3:
                key21 = '{}-{}-acer_mean_stdv/proto_{}'.format(eval_type, str2match, str(proto))
                plot_dict2[key21] = {}
            for split in splits:
                key1 = '{}-{}-proto_{}/split_{}'.format(eval_type, str2match, str(proto), str(split))
                plot_dict1[key1] = {}
                result = best_models[str(proto)][str(split)][str2match]
                plot_dict1[key1]['EER-Th'] = result[1]
                plot_dict1[key1]['ACER'] = result[3]
                acers += [result[2]]
            if proto == 2 or proto == 3:
                plot_dict2[key21]['mean_acer'] = statistics.mean(acers)
                plot_dict2[key21]['std_acer'] = statistics.stdev(acers)
            for key1, val1 in plot_dict1.items():
                train_writer.add_scalars(key1, val1, iterations + 1)
        for key3, val3 in plot_dict2.items():
            train_writer.add_scalars(key3, val3, iterations + 1)


def writeHardVids2Text(hard_videos, evalOutFile):
    hard_vids = list(set(hard_videos))
    for vid in hard_vids:
        evalOutFile.write('{}\n'.format(vid))


def write2TextFile(evalOutFile, results, config, test_dataset):
    evalOutFile.write('Evaluation on oulu-npu test set\n')
    evalOutFile.write('iterations # eert # eer # apcer1 # apcer2 # apcer # bpcer # acer # proto # split\n')
    evalOutFile.write('OR\n')
    evalOutFile.write('iterations # eert # eer # apcer # bpcer # acer # proto # split\n')
    protos_splits = config['protos_splits'][test_dataset]
    strs2match = config['strs2match']

    for str2match in strs2match:
        evalOutFile.write('>>>> [str2match: {}]\n'.format(str2match))
        for proto, splits in protos_splits.items():
            for split in splits:
                result = results[str(proto)][str(split)][str2match]
                if len(result) == 8:
                    evalOutFile.write('{} # {:3.4f} # {:3.4f} # {:3.4f} # {:3.4f} # {:3.4f} # {:3.4f} # {:3.4f} # {} # {}\n'.
                                       format(result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], proto, split))
                elif len(result) == 6:
                    evalOutFile.write('{} # {:3.4f} # {:3.4f} # {:3.4f} # {:3.4f} # {:3.4f} # {} # {}\n'.
                                      format(result[0], result[1], result[2], result[3], result[4], result[5], proto, split))
                else:
                    pass
            evalOutFile.write('\n')
        evalOutFile.write('\n')


def write2tensorBoardX(test_dataset, eval_type, config, iterations, train_writer, best_models, evalOutFile2):
    evalOutFile2.write('\n')
    evalOutFile2.write(' >>>>>>>>>>> Final Results (ACERs) <<<<<<<<<<<< \n')
    evalOutFile2.write('proto1 # proto2 # proto3-mean # proto3-stdev # proto4-mean # proto4-stdev\n')
    strs2match = config['strs2match']
    protos_splits = config['protos_splits'][test_dataset]
    plot_dict1 = {}
    plot_dict2 = {}
    for str2match in strs2match:
        for proto, splits in protos_splits.items():
            acers = []
            for split in splits:
                key1 = '{}-{}-proto_{}/split_{}'.format(eval_type, str2match, str(proto), str(split))
                plot_dict1[key1] = {}
                result = best_models[str(proto)][str(split)][str2match]
                plot_dict1[key1]['EER-Th'] = result[1]
                plot_dict1[key1]['ACER'] = result[3]
                acers += [result[3]]
            key2 = '{}-{}-acer_mean_stdv/proto_{}'.format(eval_type, str2match, str(proto))
            plot_dict2[key2] = {}
            if proto == 3 or proto == 4:
                plot_dict2[key2]['mean_acer'] = statistics.mean(acers)
                plot_dict2[key2]['std_acer'] = statistics.stdev(acers)
            else:
                plot_dict2[key2]['mean_acer'] = acers[0]
                plot_dict2[key2]['std_acer'] = 0.0
    for key3, val3 in plot_dict1.items():
        train_writer.add_scalars(key3, val3, iterations + 1)
    for key4, val4 in plot_dict2.items():
        train_writer.add_scalars(key4, val4, iterations + 1)
        for k, v in val4.items():
            if 'proto_1' in key4 or 'proto_2' in key4:
                if 'mean' in k:
                    evalOutFile2.write('{} # '.format(v))
            else:
                evalOutFile2.write('{} # '.format(v))
    evalOutFile2.write('\n')


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


def print_scheduler(iterations, net_type, config):
    print('>>>> creating learning rate scheduler for {}  with lr_policy {} '.format(net_type, config['lr_policy']))
    if iterations == -1:
        print('>>>> It seems this is intiating a fresh training regime, so, setting initial learning rate {} as learning rate'.format(config['lr']))
    else:
        print('>>>> It seems this is resuming the training from checkpoint at iterations {}'.format(iterations))


def get_scheduler(optimizer, config, net_type, iterations=-1):
    if 'lr_policy' not in config or config['lr_policy'] == 'constant':
        scheduler = None
    elif config['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'], last_epoch=iterations)
    elif config['lr_policy'] == 'da_lambda':
        alpha = config['da_lr_schedule']['alpha']
        beta = config['da_lr_schedule']['beta']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda p: 1/(1+alpha*p)**beta, last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config['lr_policy'])
    print_scheduler(iterations, net_type, config)
    return scheduler


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def make_dir(dirp):
    if not os.path.exists(dirp):
        os.makedirs(dirp)


def get_unit_dataloader(conf, transconf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    root = conf['data_folder_train']
    file_list = conf['data_list_train']
    new_size = transconf['new_size']
    height = transconf['crop_image_height']
    width = transconf['crop_image_width']
    loader_type = 'train_loader'
    mode = conf['mode']
    normalise = transconf['normalise']
    rand_crop = transconf['rand_crop']
    resize = transconf['resize']
    horiflip = transconf['horiflip']
    train_loader = get_data_loader_list(loader_type, root, file_list, batch_size, mode, new_size, height, width, num_workers, normalise, rand_crop, resize, horiflip)
    return train_loader


def get_data_loader_list(loader_type, root, file_list, batch_size, mode, new_size, height, width, num_workers, normalise, rand_crop, resize, horiflip):
    print('***utils.py --> get_data_loader_list() --> Creating DataLoader for {} ***'.format(loader_type))
    transform_list = []
    transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] + transform_list if normalise else transform_list
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if rand_crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if resize else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if horiflip and mode == 'train' else transform_list
    transform = None
    if transform_list:
        transform = transforms.Compose(transform_list)
    dataset = ImageFilelist(root, file_list, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode == 'train'), drop_last=False, num_workers=num_workers)
    return loader



