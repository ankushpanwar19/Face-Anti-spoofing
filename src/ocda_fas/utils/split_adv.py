import numpy as np 
import os
from sklearn.model_selection import train_test_split
import json


def split_celeb():
    file_train = "/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/intra_test/train_label.json"
    file_test = "/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/intra_test/test_label.json"

    with open(file_train) as f:
        data_train = json.load(f)
    with open(file_test) as f:
        data_test = json.load(f)


    path_train=np.array(list(data_train.keys()))
    X_train=np.array(list(data_train.values()))
    path_test=np.array(list(data_test.keys()))
    X_test=np.array(list(data_test.values()))

    #train data
    label, labelcount= np.unique(X_train[:,43], return_counts= True)
    total=labelcount.sum()
    print("Train Total:{} live:{} ({}) spoof:{} ({})\n".format(total,labelcount[0],labelcount[0]/total,labelcount[1],labelcount[1]/total))
    spoof, spoofcount= np.unique(X_train[:,40], return_counts= True)
    total=spoofcount.sum()
    print("Train Spoof:{} {}\n\n\n".format(total,spoofcount/total))


    #test data
    label, labelcount= np.unique(X_test[:,43], return_counts= True)
    total=labelcount.sum()
    print("Test Total:{} live:{} ({}) spoof:{} ({})\n".format(total,labelcount[0],labelcount[0]/total,labelcount[1],labelcount[1]/total))

    spoof, spoofcount= np.unique(X_test[:,40], return_counts= True)
    total=spoofcount.sum()
    print("Test Spoof:{}\n".format(spoofcount/total))


    p_train, p_val, xnew_train,xnew_val = train_test_split(path_train,X_train, test_size=0.10, random_state=42,stratify=X_train[:,40])

    #train dict and text
    dict_train=dict(zip(p_train, xnew_train.tolist()))
    path_train=np.char.add(p_train," ")
    text_train=np.char.add(path_train,xnew_train[:,43].astype('<U33'))
    text_train=text_train.tolist()
    #Val dict and text
    dict_val=dict(zip(p_val, xnew_val.tolist()))
    path_val=np.char.add(p_val," ")
    text_val=np.char.add(path_val,xnew_val[:,43].astype('<U33'))
    text_val=text_val.tolist()

    with open('/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/intra_test/myprotocol/train_label_t.json', 'w') as outfile:
        json.dump(dict_train, outfile)


    with open('/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/intra_test/myprotocol/train_label_t.txt', 'w') as outfile:
        outfile.writelines(["%s\n" % item  for item in text_train])

    with open('/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/intra_test/myprotocol/train_label_v.json', 'w') as outfile:
        json.dump(dict_val, outfile)

    with open('/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/intra_test/myprotocol/train_label_v.txt', 'w') as outfile:
        outfile.writelines(["%s\n" % item  for item in text_val])

    print("end")

def sample_from_celeb(name,split=0.2):
    base_path="/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/intra_test/myprotocol"
    file_train = os.path.join(base_path,'train_label_t.txt')

    live_path=[]
    with open(file_train) as f:
        for line in f:
            p=line.split(" ")[0]
            if "live" in p:
               live_path.append(p) 


    _,live_path_test = train_test_split(live_path, test_size=split, random_state=42)

    #train dict and text
    #Val dict and text
    live_sample=np.char.add(live_path_test," ")
    live_sample=np.char.add(live_sample,"1")
    live_sample=live_sample.tolist()

    write_path= os.path.join(base_path,name)
    with open(write_path, 'w') as outfile:
        outfile.writelines(["%s\n" % item  for item in live_sample])

    print("end")

def visualize_celeb():

    print("Celeb")

if __name__ == "__main__":
    # split_celeb()
    sample_from_celeb("train_live.txt")


