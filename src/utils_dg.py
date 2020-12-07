import math
import sys
sys.path.append("src")
import os
print (os.getcwd())
from get_examples_labels_siw import get_examples_labels as get_siw_dataset
from get_examples_labels_oulu import get_examples_labels as get_oulu_dataset
from get_examples_labels_replayattack import get_examples_labels as get_repatack_dataset
# from get_examples_labels_replaymobile import get_examples_labels as get_repmob_dataset
from get_examples_labels_casia import get_examples_labels as get_casia_dataset
from get_examples_labels_msu import get_examples_labels as get_msu_dataset
from get_examples_labels_celebA import get_examples_labels as get_celebA_dataset
from data_loader_anet import get_loader as get_loader_all
from data_loader_anet import get_MOT_loader


def get_data_loader(config, config_dl, dataset_name, mode, small_trainset = False, drop_last = False):
    '''
    Dataloader for images network
    output: 
    Dataloader: Contain list of Images tensor [8,3,224,224],path of images ['ms/live/real_client0...ne01/00121'], labels of images [0,1]
    epoch_size:iteration for this dataset
    '''
    machine = config['machine']
    dataset = dataset_name
    num_workers = config['num_workers']
    inp_dim = (config['inp_dim'], config['inp_dim'])
    depth_map_size = (config['depth_map_size'], config['depth_map_size'])
    app_feats = config['app_feats']
    if config['net_type'] == 'lstmmot' or config['net_type'] == 'cvpr2018dg'  or config['net_type'] == 'resnet18': # --- Because it is the loader to get frame-wise data
        batch_size = config['batch_size_cnn']
    else:
        batch_size = config['batch_size']
    print(' --- sampling {} examples,labels for dataset {} --- '.format(mode, dataset))
    part_all, labels_all, num_exmps = get_part_labels(config, config_dl, mode, small_trainset = small_trainset, drop_last = drop_last, dataset_name=dataset)
    epoch_size = math.floor(num_exmps / batch_size)
    model_save_step = epoch_size
    print('[dataset: {}]  [total_training_examples: {}]  [mini-batch size: {}];  [epoch-size: {}];  [model_save_step: {}]'.format(dataset, num_exmps, batch_size, epoch_size, model_save_step))
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
    '''
    MOT(Motion)/ Video Data loader useful for all train dataset for video network (but use one by one for each data set)
    output: 
    Dataloader: Contain list of Video tensor[1,8,2,224,224],path of Videos ['ca/spoof/003_HR_4.avi'], labels of Videos [0,1]
    Epoch_Size = subvideos=videos*sample_per vid(8), 
    save at step
    '''
    # data_loader = None
    # epoch_size = None
    # modelsave_step = None

    machine = config['machine']
    dataset = dataset_name
    print('DATASET NAME:{}'.format(dataset_name))

    num_workers = config['num_workers']
    inp_dim = (config['inp_dim'], config['inp_dim'])
    depth_map_size = (config['depth_map_size'], config['depth_map_size'])
    sel_every = config['sel_every']
    sel_these_many = config['sel_these_many']
    app_feats = config['app_feats']
    if config['net_type'] == 'lstmmot' or config['net_type'] == 'cvpr2018dg'  or config['net_type'] == 'resnet18': 
        batch_size = config['batch_size_lstm']
    elif config['net_type'] == 'cvpr2019_back':
        batch_size = config['batch_size_cnn']


    # eval_on_oulu_devset = config['eval_on_oulu_devset']

    print(' --- sampling {} examples,labels for dataset {} --- '.format(mode, dataset))
    # print(' --- dataset read path {} ---'.format(dataset_path))

    ## this one only gets lists for all image paths
    part_all, labels_all, num_exmps = get_part_labels(config, config_dl, mode, small_trainset = small_trainset, drop_last = drop_last, dataset_name=dataset)

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

    print('[TOTAL_training_img_examples: {}] [TOTAL videos: {}] [mini-batch size: {}];  [epoch-size: {}];  [model_save_step: {}]'.format(num_exmps,len(my_dataloader)*batch_size,batch_size, epoch_size, model_save_step))
    print(' --- NOTE: You can set the value of [snapshot_save_iter] in configs.yaml a value around [model_save_step]  ---')
    return my_dataloader, epoch_size, model_save_step


def get_dataset_id(dataset):
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
    elif dataset == 'msu':
        datasetID = 'ms'
    elif dataset == 'celebA':
        datasetID = 'ce'
    else:
        pass
    return datasetID

# --- This is used to get train labels & images
def get_part_labels(config, config_dl, mode, small_trainset = False, drop_last = False, dataset_name=None):
    ''' 
    To get each frame/example path and label 
    output: 
    part_all : [ca/live/001_1.avi/00001,...]
    labels : Dictionary of labels with path as keys ['ca/live/001_1.avi/00001':0, ...]
    num_exmps:number of examples
    '''
    machine = config['machine']
    net_type = config['net_type']
    # pickle_fname = config['pickle_fname']
    num_cls = config_dl['num_cls']
    dataset = dataset_name
    proto = config_dl[dataset]['proto']
    split = config_dl[dataset]['split']
    # print('>>>>>>> {} <<<<<<'.format(dataset))
    # print(config_dl[dataset])
    dataset_path = config_dl[dataset]['dataset_path_machine{}'.format(machine)]
    datasetID = get_dataset_id(dataset)
    sel_every = config_dl[dataset]['sel_every_mergeall']
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
            get_oulu_dataset(dataset_path, mode, proto, split, sel_every, sel_these_many, img_path, net_type,eval_on_oulu_devset = eval_on_oulu_devset, datasetID = datasetID, num_cls = num_cls)
    elif datasetID == 'ra':
        part, labels, _, _, num_exmps = \
            get_repatack_dataset(dataset_path, mode, proto, split, sel_every, sel_these_many, img_path, net_type, small_trainset = small_trainset,datasetID = datasetID, num_cls = num_cls)
    elif datasetID == 'rm':
        pass
        # part, labels, _, _, num_exmps = \
        #     get_repmob_dataset(dataset_path, mode, proto, split, sel_every, sel_these_many, img_path, net_type, small_trainset = small_trainset,datasetID = datasetID, num_cls = num_cls)
    elif datasetID == 'ca':
        part, labels, _, _, num_exmps = \
            get_casia_dataset(dataset_path, mode, proto, split, sel_every, sel_these_many, img_path, net_type, small_trainset = small_trainset,datasetID = datasetID, num_cls = num_cls)
    elif datasetID == 'ms':
        part, labels, _, _, num_exmps = \
            get_msu_dataset(dataset_path, mode, proto, split, sel_every, sel_these_many, img_path, net_type, small_trainset = small_trainset,datasetID = datasetID, num_cls = num_cls)
    elif datasetID == 'ce':
        part, labels, _, _, num_exmps = \
            get_celebA_dataset(dataset_path, mode, proto, split, sel_every, sel_these_many, img_path, net_type, small_trainset = small_trainset,datasetID = datasetID, num_cls = num_cls)
    else:
        pass
    return part, labels, num_exmps