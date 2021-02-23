import os
import glob
import numpy as np
from argparse import ArgumentParser
from sklearn import metrics



def auc_evaluation(args):
    score_folder=os.path.join(args.experiment_path,args.score_files)
    auc_summary= os.path.join(args.experiment_path,args.auc_summary)
    files_path_array=glob.glob(score_folder+"/*.txt")
    files_path_array.sort()
    for p in files_path_array:
        print(p)
        with open(p) as f:
            data = f.readlines()
        
        label_list=[]
        pred_list=[]

        for line in data:
            split=line.split(",")
            if 'val' in p:
                label_list.append(int(split[1]))
            else:
                label_list.append(int(split[2]))
            pred_list.append(float(split[0]))
        fpr, tpr, thresholds = metrics.roc_curve(label_list, pred_list, pos_label=0)
        auc=metrics.auc(fpr, tpr)

        path_split=p.split("/")

        fsum=open(auc_summary,'a')
        fsum.write("\n{}   AUC:{}\n".format(path_split[-1],auc))
        fsum.close()
        
        
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--net_type', type=str, default='lstmmot')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--experiment_path', type=str, default='output/fas_project/ocda_exp')
    parser.add_argument('--score_files', type=str, default='ocda_expand_improve/mann_net/mann_net_exp_001/score_files')
    parser.add_argument('--auc_summary', type=str, default='ocda_expand_improve/mann_net/mann_net_exp_001/auc_summary.txt')

    parser.add_argument('--comments', type=str, default='Train with 0.5 acc thres lr 10-5 with mem')

    args = parser.parse_args()
    auc_evaluation(args)




