import math
# from get_examples_labels_casia import get_examples_labels as get_casia_dataset
# from get_examples_labels_replayattack import get_examples_labels as get_replayattack_dataset
from get_examples_labels_oulu import get_examples_labels as get_oulu_dataset
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


def get_dataloader_test(config, configdl, debug, dataset_name = None,drop_last = False):   
    if debug:
        print(' --- run_eval.py --> get_test_dataloader() --> getting the test dataloader ---')
    
    # dataset_name = config['test_dataset']
    dataset_name = 'oulu-npu'
    machine = config['machine']
    num_workers = config['num_workers']
    inp_dim = (config['inp_dim'], config['inp_dim'])
    depth_map_size = (config['depth_map_size'], config['depth_map_size'])
    app_feats = config['app_feats']
    if config['net_type'] == 'lstmmot':
        batch_size = config['batch_size_lstm']
    else:
        batch_size = config['batch_size']
    protocol = config['test_dataset_conf'][dataset_name]['protocol']
    split = config['test_dataset_conf'][dataset_name]['split']
    sel_every = config['test_dataset_conf'][dataset_name]['sel_every']
    sel_these_many = config['test_dataset_conf'][dataset_name]['sel_these_many']
    dataset_path = config['test_dataset_conf'][dataset_name]['dataset_path_machine{}'.format(machine)]
    eval_type = config['eval_type']
    print("????Debug: eval_type: {}".format(eval_type))
    if 'Val' in eval_type:
        # if dataset_name == 'replay-attack': 
        if dataset_name == 'oulu-npu':
            img_path = config['test_dataset_conf'][dataset_name]['img_path_dev_machine{}'.format(machine)]
            mode = 'val'
        # elif dataset_name == 'casia':
        #     img_path = config['test_dataset_conf'][dataset_name]['full_img_path_machine{}'.format(machine)]
        #     mode = 'val'
        else:
            print('Not specified dataset')
    elif 'Test' in eval_type:
        img_path = config['test_dataset_conf'][dataset_name]['img_path_machine{}'.format(machine)]
        mode = 'test'

    else:
        print('Not specify eval type')
        pass
    const = config['const_testset']
    num_cls = configdl['num_cls']
    datasetID = get_dataset_id(dataset_name)
    net_type = config['net_type']
    print(' --- sampling {} examples,labels for dataset {} --- '.format('test', dataset_name))
    print(' --- dataset read path {} ---'.format(dataset_path))
    print(' ==== img_path {} ==='.format(img_path))
    print('--- DATA IDS: {}--- '.format(datasetID))

    if datasetID == 'ou':
        part, labels, gtFlags, scores, num_exmps = \
        get_oulu_dataset(dataset_path, mode, protocol, split, sel_every, sel_these_many, img_path, net_type, eval_on_oulu_devset = False, small_trainset = False, datasetID = datasetID, num_cls = num_cls)
    else: 
        print('>>>> Error: Unrecognized Dataset Type')

    
    # if datasetID == 'ca' or datasetID == 'ca-ma':
    #     part, labels, gtFlags, scores, num_exmps = \
    #         get_casia_dataset(dataset_path, mode, protocol, split, sel_every, sel_these_many, img_path, net_type, datasetID=datasetID, num_cls=num_cls)
    # elif datasetID == 'ra' or datasetID == 'ra-ma':
    #     print('Getting replay ids')
    #     part, labels, gtFlags, scores, num_exmps = \
    #         get_replayattack_dataset(dataset_path, mode, protocol, split, sel_every, sel_these_many, img_path, net_type, datasetID=datasetID, num_cls=num_cls)
    
    print('num exmps: {}; Batch size: {}'.format(num_exmps,batch_size))
    
    num_test_batches = math.floor(num_exmps / batch_size)
    last_batch_size = num_exmps % batch_size
    if last_batch_size > 0:
        num_test_batches += 1
    print('mini-batch size [{}]; num_test_batches [{}]'.format(batch_size, num_test_batches))

    
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


    if config['net_type'] == 'lstmmot' or 'cvpr2018dg' in config['net_type']:
        print('>>> Eval {} lstmmot <<< '.format(datasetID))
        my_dataloader = get_MOT_loader(machine, configdl, part[mode], labels, mode, drop_last, **params)
    else: 
        my_dataloader = get_loader_all(machine, configdl, part[mode], labels, mode, drop_last, **params)
    return my_dataloader, num_exmps, scores, gtFlags, batch_size, num_test_batches, img_path



def get_eval_part_labels(config, configdl,mode,dataset_name = None,drop_last=False):   
    # if debug:
    #     print(' --- run_eval.py --> get_test_dataloader() --> getting the test dataloader ---')
    
    # dataset_name = config['test_dataset']
    dataset_name = 'oulu-npu'
    machine = config['machine']
    num_workers = config['num_workers']
    inp_dim = (config['inp_dim'], config['inp_dim'])
    depth_map_size = (config['depth_map_size'], config['depth_map_size'])
    
    protocol = config['test_dataset_conf'][dataset_name]['protocol']
    split = config['test_dataset_conf'][dataset_name]['split']
    sel_every = config['test_dataset_conf'][dataset_name]['sel_every']
    sel_these_many = config['test_dataset_conf'][dataset_name]['sel_these_many']
    dataset_path = config['test_dataset_conf'][dataset_name]['dataset_path_machine{}'.format(machine)]
 
    if 'val' in mode:
        # if dataset_name == 'replay-attack': 
        if dataset_name == 'oulu-npu':
            img_path = config['test_dataset_conf'][dataset_name]['img_path_dev_machine{}'.format(machine)]
            # mode = 'val'
        # elif dataset_name == 'casia':
        #     img_path = config['test_dataset_conf'][dataset_name]['full_img_path_machine{}'.format(machine)]
        #     mode = 'val'
        else:
            print('Not specified dataset')
    elif 'test' in mode:
        img_path = config['test_dataset_conf'][dataset_name]['img_path_machine{}'.format(machine)]
        # mode = 'test'

    else:
        print('Not specify eval type')
        pass
    const = config['const_testset']
    num_cls = configdl['num_cls']
    datasetID = get_dataset_id(dataset_name)
    net_type = config['net_type']
    print(' --- sampling {} examples,labels for dataset {} --- '.format('test', dataset_name))
    print(' --- dataset read path {} ---'.format(dataset_path))
    print(' ==== img_path {} ==='.format(img_path))
    print('--- DATA IDS: {}--- '.format(datasetID))

    if datasetID == 'ou':
        part, labels, gtFlags, scores, num_exmps = \
        get_oulu_dataset(dataset_path, mode, protocol, split, sel_every, sel_these_many, img_path, net_type, eval_on_oulu_devset = False, small_trainset = False, datasetID = datasetID, num_cls = num_cls)
    else: 
        print('>>>> Error: Unrecognized Dataset Type')

    
    # if datasetID == 'ca' or datasetID == 'ca-ma':
    #     part, labels, gtFlags, scores, num_exmps = \
    #         get_casia_dataset(dataset_path, mode, protocol, split, sel_every, sel_these_many, img_path, net_type, datasetID=datasetID, num_cls=num_cls)
    # elif datasetID == 'ra' or datasetID == 'ra-ma':
    #     print('Getting replay ids')
    #     part, labels, gtFlags, scores, num_exmps = \
    #         get_replayattack_dataset(dataset_path, mode, protocol, split, sel_every, sel_these_many, img_path, net_type, datasetID=datasetID, num_cls=num_cls)
    
    return part,labels,num_exmps