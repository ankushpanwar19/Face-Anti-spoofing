import numpy as np 
import os
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

test_file = "/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/intra_test/test_label.json"
predict_file = "output/fas_project/DG_exp/lstmmot_exp_013/ocda_fas_files/scheduled_mann_net/scheduled_mann_net_exp_005/score_files/test_score_epoch1.txt"

with open(test_file) as f:
  data = json.load(f)


  pred_data = f.read()

# 'Data/test/6964/spoof/494405.png'
location='output/fas_project/DG_exp/lstmmot_exp_013/ocda_fas_files/scheduled_mann_net/scheduled_mann_net_exp_005/test_spoof.txt'
spoof_writer=open(location,'w')
summary='output/fas_project/DG_exp/lstmmot_exp_013/ocda_fas_files/scheduled_mann_net/scheduled_mann_net_exp_005/spoof_summary.txt'
summary_writer=open(summary,'w')
i=0
dimensions=(11)
correct_pred=np.zeros(dimensions)
wrong_pred=np.zeros(dimensions)
with open(predict_file) as f:
    for strpred in tqdm(f.readlines()):
        pred_arr=strpred.split(',')
        loc=pred_arr[3].strip()[2:]
        key='Data/test'+loc+'.png'
        v=data[key]
        writeline=strpred.strip() + ',' + str(v[40]) +'\n'
        spoof_writer.write(writeline)

        spoof=v[40]
        if pred_arr[1]==pred_arr[2]:
            correct_pred[spoof]+=1
        else:
            wrong_pred[spoof]+=1
        # if i==2:
        #     break
        # i+=1

summary_writer.write(np.array2string(correct_pred))
summary_writer.write(np.array2string(wrong_pred))
summary_writer.close()
spoof_writer.close()


# with open('/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/protocol1/train_label_t.json', 'w') as outfile:
#     json.dump(train_dict, outfile)

# with open('/scratch_net/moustachos/apanwar/CelebA-Spoof/CelebA_Spoof/metas/protocol1/train_label_v.json', 'w') as outfile:
#     json.dump(test_dict, outfile)

# print(data)