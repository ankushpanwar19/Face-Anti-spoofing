import torch
import matplotlib.pyplot as plt

import argparse
import os
import numpy as np
from roc_utils import val_eval_hter
from roc_utils_hter import myeval as myeval_hter

from sklearn.metrics import roc_auc_score, roc_curve

parser = argparse.ArgumentParser(description='Face anti-spoofing evaluation file')
parser.add_argument('--vl_file_path', default='', metavar='DIR', help='path to config file')
parser.add_argument('--iter_num',default = 0,type=int, help= 'number of iterations to be tested')
parser.add_argument('--net_type',default = 'anet',help = 'network type')

args = parser.parse_args()

path_base = ''
txt_name = args.net_type+'_scores_vl_{:08d}.txt'.format(args.iter_num)
score_fname = os.path.join(path_base,args.vl_file_path,txt_name)
ground_truth_list = []
score_list = []

    
print('score_fname: ',score_fname)
with open(score_fname) as f:
    for line in f:
        strs = line.split(',')
        score_list += [float(strs[0])]
        if int(strs[1]) == 1:
            ground_truth_list += [1]
        elif int(strs[1]) == -1 or int(strs[1]) == -2:
            ground_truth_list += [0]
scores_arr = np.asarray(score_list)
ground_truth_arr = np.asarray(ground_truth_list)
ground_truth_arr = ground_truth_arr.astype(int)

auc = roc_auc_score(ground_truth_arr,scores_arr)
print('>>>> AUC Score: {} <<<<'.format(auc*100))

        
