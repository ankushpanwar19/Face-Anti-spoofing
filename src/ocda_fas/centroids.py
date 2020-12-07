import os
import numpy as np
from tqdm import tqdm
import torch

from data_utils import get_domain_list,domain_combined_data_loaders
from source.models.dg_resnet import DgEncoder


def class_count (data):

    try:
        labels = np.array(data.dataset.labels)
    except AttributeError:
        labels = np.array(data.dataset.targets)

    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num


def compute_source_centroids(config,configdl):
    device = config['device']

    source_domain_list,target_domain_list= get_domain_list(config,'mann_net')

    #  source and target Data loaders
    src_data_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='train',net='mann_net',type='src')

    net = DgEncoder(config)
    net.load_init_weights(config['checkpoint_file_path'])
    net.to(device)
    net.eval()

    num_cls=2
    feat_dim=2048
    centroids = np.zeros((num_cls, feat_dim))
    count=np.zeros((num_cls, 1))

    total=int(len(src_data_loader.dataset)/src_data_loader.batch_size)
    with tqdm(total=total) as pbar:
        for idx, (data,_,target) in enumerate(src_data_loader):

            # setup data and target #
            data=data.to(device)
            target = target.to(device)
            data.require_grad = False
            target.require_grad = False

            # forward pass
            x,_,_ = net(data.clone())

            # add feed-forward feature to centroid tensor
            for i in range(len(target)):
                label = target[i]
                centroids[label.item()] += x[i].detach().cpu().numpy()
                count[label.item(),0]+=1.0
            pbar.update(1)
        # break
    # Average summed features with class count
    centroids /= count

    np.save(os.path.join(config['centroids_path'],config['mann_net']['centroid_fname']), centroids)



def evaluation()