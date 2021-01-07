import numpy as np 
import os
from sklearn.model_selection import train_test_split
import json

file = "/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/intra_test/train_label.json"

with open(file) as f:
  data = json.load(f)

path=[]
value=[]
live_cnt=0
spoof_cnt=0
for p,v in data.items():
    path.append(p)
    value.append(v[-1])
    if 'live' in p:
        live_cnt+=1
    else:
        spoof_cnt+=1

total=len(data)
print("total:{} live:{} ({}) spoof:{} ({})".format(total,live_cnt,live_cnt/total,spoof_cnt,spoof_cnt/total))

p_train, p_test = train_test_split(path, test_size=0.10, random_state=42,stratify=value)

train_dict={}
test_dict={}

for k in p_train:
    train_dict[k]=data[k]

for l in p_test:
    test_dict[l]=data[l]


# with open('/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/protocol1/train_label_t.json', 'w') as outfile:
#     json.dump(train_dict, outfile)

# with open('/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/protocol1/train_label_v.json', 'w') as outfile:
#     json.dump(test_dict, outfile)

# print(data)