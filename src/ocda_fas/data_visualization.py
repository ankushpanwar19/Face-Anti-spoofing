import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.ticker import PercentFormatter


def plot(data,name,type="Spoof"):

    bin_list=np.unique(data)
    bin_list=np.append(bin_list,bin_list[-1]+1)
    N,bins,patches=plt.hist(data, bins=bin_list,weights=np.ones(len(data)) / len(data)) 

    for i, v in enumerate(N):
        plt.text(i + 0.1,v+0.005, str(round(v*100,1))+"%", color='black')
    for i in range(len(bins)-1):
        if i%2==0:
            patches[i].set_facecolor('b')
        else:
            patches[i].set_facecolor('r')
    #axis([xmin,xmax,ymin,ymax])
    plt.xlabel(type+" Type")
    plt.ylabel('Percentage')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title(name)

    plt.savefig("src/ocda_fas/assets/"+name)
    plt.close()

def load_plot_data(file,type="train"):

    with open(file) as f:
        data = json.load(f)

    X= np.array(list(data.values()))
    spoof_type=X[:,40]
    illum_type=X[:,41]
    env_type=X[:,42]

    plot(spoof_type,type+"_spoof.png","Spoof")
    plot(illum_type,type+"_illum.png","Illumination")
    plot(env_type,type+"_env.png","Environment")

train_file="/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/intra_test/myprotocol/train_label_t.json"
val_file="/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/intra_test/myprotocol/train_label_v.json"
test_file="/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/intra_test/myprotocol/test_label.json"

load_plot_data(train_file,"train")
load_plot_data(val_file,"val")
load_plot_data(test_file,"test")
# with open(train_file) as f:
#     data_train = json.load(f)

# train_x= np.array(list(data_train.values()))
# spoof_type=train_x[:,40]
# illum_type=train_x[:,41]
# env_type=train_x[:,42]

# plot(spoof_type,"train_spoof.png")
# plot(illum_type,"train_illum.png")
# plot(env_type,"train_env.png")


