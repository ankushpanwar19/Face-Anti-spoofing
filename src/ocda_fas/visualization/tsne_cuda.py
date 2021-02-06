import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

sys.path.append("src")
from utils import get_config, make_dir
from ocda_fas.utils.data_utils import get_domain_list,domain_combined_data_loaders,make_exp_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
config_fname='src/configs/train.yaml'
config= get_config(config_fname)
configdl_fname= 'src/configs/data_loader_dg.yaml'
configdl= get_config(configdl_fname)
config['device']=device
config['net_type']='lstmmot'

source_domain_list,target_domain_list= get_domain_list(config,'da_baseline')
src_train_loader=domain_combined_data_loaders(config,configdl,source_domain_list,mode='val',net='da_baseline',type='src',shuffle=True)
tgt_train_loader=domain_combined_data_loaders(config,configdl,target_domain_list,mode='val',net='da_baseline',type='tgt',shuffle=True)

joint_loader = zip(src_train_loader, tgt_train_loader)
# image_dataset=torch.tensor()
images_tsv=[]
# label_tsv=[['Domain','Dataset']]
label_tsv=[]
path='src/ocda_fas/visualization/vis_tensorboard'
os.makedirs(path, exist_ok=True)
writer = SummaryWriter(path)
flag=0
for batchidx, ((src_images,src_p,src_labels),(tgt_images,tgt_p,tgt_labels)) in enumerate(tqdm(joint_loader)):

    src_live_images=src_images[src_labels==0,:,:,:]
    tgt_live_images=tgt_images[tgt_labels==0,:,:,:]

    src_p=np.array(src_p)
    tgt_p=np.array(tgt_p)
    src_p_live=src_p[src_labels==0]
    tgt_p_live=tgt_p[tgt_labels==0]

    if flag==0:
        images_tsv= torch.cat((src_live_images,tgt_live_images),0)
        label_tsv=label_tsv+[s.split('/')[0] for s in src_p_live] + [t.split('/')[0] for t in tgt_p_live]
        flag=1
    else:

        images_tsv=torch.cat((images_tsv,src_live_images,tgt_live_images),0)
        label_tsv=label_tsv+[s.split('/')[0] for s in src_p_live] + [t.split('/')[0] for t in tgt_p_live]


    if (batchidx+1)%20==0:
        features=images_tsv.reshape((images_tsv.shape[0],-1))
        writer.add_embedding(features,
                        metadata=label_tsv,
                        label_img=images_tsv,
                        global_step=3)
        break
    src_images[0].show()
    # for i in range(src_images.shape[0]):
    #     im=transforms.ToPILImage(mode='RGB')(src_images[i])
    #     im.save("image.png")
    # if batchidx==5:
    #     break

writer.close()
# print(image_dataset.shape)
# # X=X.numpy()
# X=image_dataset.reshape((image_dataset.shape[0],-1))
# # print(X.shape)
# # X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)
# X_embedded = TSNE(n_components=2).fit_transform(X)

# print(X_embedded)
# classes=['live','spoof']
# scatter=plt.scatter(X_embedded[:,0],X_embedded[:,1],c=label_dataset)
# plt.legend(handles=scatter.legend_elements()[0], labels=classes)
# plt.savefig("msu_check.png")
# # plt.show()

print("end")
