import math
from get_examples_labels_casia import get_examples_labels as get_casia_dataset
from get_examples_labels_replayattack import get_examples_labels as get_replayattack_dataset
from utils import get_loader_all
from data_loader_anet import get_MOT_loader

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
    elif dataset == 'casia-maged':
        datasetID = 'ca-ma'
    elif dataset == 'replay-attack-maged':
        datasetID = 'ra-ma'
    elif dataset == 'msu-mfsd':
        datasetID = 'ms'
    else:
        pass
    return datasetID


def get_dataloader_test(config, configdl, debug,dataset_name = None,drop_last = False):
    '''
    Data loader for validation and test set for Replay and Casia data set
    ouput: 
    Dataloader videos [tensor,video path, label],
    num_exmps [number of frames], scores ([[],1],[[],-2]), 
    gtFlags ['ca/live/001_1.avi/00001':1,...], batch_size, num_test_batches, img_path = '/scratch_net/knuffi_third/susaha/apps/datasets/casia/rgb_images_full/train'

    '''
    if debug:
        print(' --- run_eval.py --> get_test_dataloader() --> getting the test dataloader ---')
    
    # dataset_name = config['test_dataset']
    print('>>>> test dataset: ',config['test_dataset'])
    eval_type = config['eval_type']

    if 'Val' in eval_type:
        dataset_name = dataset_name
    elif 'Test' in eval_type:
        dataset_name = config['test_dataset']
    machine = config['machine']
    num_workers = config['num_workers']
    inp_dim = (config['inp_dim'], config['inp_dim'])
    depth_map_size = (config['depth_map_size'], config['depth_map_size'])
    app_feats = config['app_feats']
    if config['net_type'] == 'lstmmot' or config['net_type'] == 'cvpr2018dg':
        batch_size = config['batch_size_lstm']
    else:
        batch_size = config['batch_size']
    print('>>>> dataset_name: ', dataset_name)
    protocol = config['test_dataset_conf'][dataset_name]['protocol']
    split = config['test_dataset_conf'][dataset_name]['split']
    sel_every = config['test_dataset_conf'][dataset_name]['sel_every']
    sel_these_many = config['test_dataset_conf'][dataset_name]['sel_these_many']
    dataset_path = config['test_dataset_conf'][dataset_name]['dataset_path_machine{}'.format(machine)]


    if 'Val' in eval_type:
        if dataset_name == 'replay-attack': 
            img_path = config['test_dataset_conf'][dataset_name]['full_img_path_dev_machine{}'.format(machine)]
            mode = 'val'
        elif dataset_name == 'casia':
            img_path = config['test_dataset_conf'][dataset_name]['full_img_path_machine{}'.format(machine)]
            mode = 'val'
        else:
            print('Not specified dataset')
    elif 'Test' in eval_type:
        img_path = config['test_dataset_conf'][dataset_name]['full_img_path_test_machine{}'.format(machine)]
        mode = 'test'

    # if eval_type == 'Replay2CasiaVal':
    #     img_path = config['test_dataset_conf'][dataset_name]['full_img_path_machine{}'.format(machine)]
    #     mode = 'val'
    # elif eval_type == 'Casia2ReplayVal':
    #     img_path = config['test_dataset_conf'][dataset_name]['full_img_path_dev_machine{}'.format(machine)]
    #     mode = 'val'
    # elif eval_type == 'Replay2CasiaTest' or eval_type == 'Casia2ReplayTest':
    #     img_path = config['test_dataset_conf'][dataset_name]['full_img_path_test_machine{}'.format(machine)]
    #     mode = 'test'
    # else:
    #     pass

    const = config['const_testset']
    num_cls = configdl['num_cls']
    datasetID = get_dataset_id(dataset_name)
    net_type = config['net_type']
    print(' --- sampling {} examples,labels for dataset {} --- '.format('test', dataset_name))
    print(' --- dataset read path {} ---'.format(dataset_path))
    print(' ==== img_path {} ==='.format(img_path))
    print('--- DATA IDS: {}--- '.format(datasetID))

    if datasetID == 'ca' or datasetID == 'ca-ma':
        part, labels, gtFlags, scores, num_exmps = \
            get_casia_dataset(dataset_path, mode, protocol, split, sel_every, sel_these_many, img_path, net_type, datasetID=datasetID, num_cls=num_cls)
    elif datasetID == 'ra' or datasetID == 'ra-ma':
        print('Getting replay ids')
        part, labels, gtFlags, scores, num_exmps = \
            get_replayattack_dataset(dataset_path, mode, protocol, split, sel_every, sel_these_many, img_path, net_type, datasetID=datasetID, num_cls=num_cls)
    # if config['net_type'] == 'lstmmot':
    #     batch_size = config['batch_size_lstm']

    num_test_batches = math.floor(num_exmps / batch_size)
    last_batch_size = num_exmps % batch_size
    if last_batch_size > 0:
        num_test_batches += 1
    print('mini-batch size [{}]; num_test_batches [{}]'.format(batch_size, num_test_batches))

    # print('SCORES: ',scores)
    params = {'app_feats': app_feats,
              'batch_size': batch_size,
              'shuffle': True,
              'num_workers': num_workers,
              'res': inp_dim,
              'dataset_name': dataset_name,
              'depth_map_size': depth_map_size,
              'net_type': net_type, 
              'const_testset': const
              }

    # print('???? config nettype:', config['net_type'])
    if config['net_type'] == 'lstmmot' or 'cvpr2018dg' in config['net_type']:

        print('???? Eval {} lstmmot <<< '.format(datasetID))
        my_dataloader = get_MOT_loader(machine, configdl, part[mode], labels, mode, drop_last, **params)
    else: 
        my_dataloader = get_loader_all(machine, configdl, part[mode], labels, mode, drop_last, **params)
    return my_dataloader, num_exmps, scores, gtFlags, batch_size, num_test_batches, img_path


def get_eval_part_labels(config, configdl,mode,dataset_name = None,drop_last=False):
    '''
    Data loader for validation and test set for Replay and Casia data set
    ouput: 
    Dataloader videos [tensor,video path, label],
    num_exmps [number of frames], scores ([[],1],[[],-2]), 
    gtFlags ['ca/live/001_1.avi/00001':1,...], batch_size, num_test_batches, img_path = '/scratch_net/knuffi_third/susaha/apps/datasets/casia/rgb_images_full/train'

    '''
    # if debug:
    #     print(' --- run_eval.py --> get_test_dataloader() --> getting the test dataloader ---')
    
    # dataset_name = config['test_dataset']
    print('>>>> {:s} dataset: {:s}'.format(mode,dataset_name))
    # eval_type = config['eval_type']

    machine = config['machine']
    num_workers = config['num_workers']
    inp_dim = (config['inp_dim'], config['inp_dim'])
    depth_map_size = (config['depth_map_size'], config['depth_map_size'])
    app_feats = config['app_feats']
    print('>>>> dataset_name: ', dataset_name)
    protocol = config['test_dataset_conf'][dataset_name]['protocol']
    split = config['test_dataset_conf'][dataset_name]['split']
    sel_every = config['test_dataset_conf'][dataset_name]['sel_every']
    sel_these_many = config['test_dataset_conf'][dataset_name]['sel_these_many']
    dataset_path = config['test_dataset_conf'][dataset_name]['dataset_path_machine{}'.format(machine)]


    if 'val' in mode:
        if dataset_name == 'replay-attack': 
            img_path = config['test_dataset_conf'][dataset_name]['full_img_path_dev_machine{}'.format(machine)]
        elif dataset_name == 'casia':
            img_path = config['test_dataset_conf'][dataset_name]['full_img_path_machine{}'.format(machine)]
        else:
            print('Not specified dataset')
    elif 'test' in mode:
        img_path = config['test_dataset_conf'][dataset_name]['full_img_path_test_machine{}'.format(machine)]

    # if eval_type == 'Replay2CasiaVal':
    #     img_path = config['test_dataset_conf'][dataset_name]['full_img_path_machine{}'.format(machine)]
    #     mode = 'val'
    # elif eval_type == 'Casia2ReplayVal':
    #     img_path = config['test_dataset_conf'][dataset_name]['full_img_path_dev_machine{}'.format(machine)]
    #     mode = 'val'
    # elif eval_type == 'Replay2CasiaTest' or eval_type == 'Casia2ReplayTest':
    #     img_path = config['test_dataset_conf'][dataset_name]['full_img_path_test_machine{}'.format(machine)]
    #     mode = 'test'
    # else:
    #     pass

    const = config['const_testset']
    num_cls = configdl['num_cls']
    datasetID = get_dataset_id(dataset_name)
    net_type = config['net_type']
    print(' --- sampling {} examples,labels for dataset {} --- '.format('test', dataset_name))
    print(' --- dataset read path {} ---'.format(dataset_path))
    print(' ==== img_path {} ==='.format(img_path))
    print('--- DATA IDS: {}--- '.format(datasetID))

    if datasetID == 'ca' or datasetID == 'ca-ma':
        part, labels, gtFlags, scores, num_exmps = \
            get_casia_dataset(dataset_path, mode, protocol, split, sel_every, sel_these_many, img_path, net_type, datasetID=datasetID, num_cls=num_cls)
    elif datasetID == 'ra' or datasetID == 'ra-ma':
        print('Getting replay ids')
        part, labels, gtFlags, scores, num_exmps = \
            get_replayattack_dataset(dataset_path, mode, protocol, split, sel_every, sel_these_many, img_path, net_type, datasetID=datasetID, num_cls=num_cls)
    # if config['net_type'] == 'lstmmot':
    #     batch_size = config['batch_size_lstm']

    return part,labels,num_exmps