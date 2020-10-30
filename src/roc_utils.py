import numpy as np
import sys


def vl_tpfp(labels, scores):
    p = np.sum(labels > 0)
    n = np.sum(labels < 0)
    perm = np.argsort(-scores)
    scores = scores[perm]
    stop = np.max(np.where(scores > np.NINF)) + 1
    perm = perm[:stop]
    labels = labels[perm]
    tp = np.cumsum(labels > 0)
    fp = np.cumsum(labels < 0)
    tp = np.concatenate((np.array([0]), tp))
    fp = np.concatenate((np.array([0]), fp))
    return tp, fp, p, n, perm


def vl_roc(scores, labels):
    info = {}
    tp, fp, p, n, perm = vl_tpfp(labels, scores)
    small = 1e-10
    tpr = tp / max(p, small)
    fpr = fp / max(n, small)
    fnr = 1 - tpr
    tnr = 1 - fpr
    s = np.amax(np.where(tnr > tpr))
    if s == len(tpr):
        info['eer'] = np.NAN
        info['eerThreshold'] = 0
    else:
        if tpr[s] == tpr[s+1]:
            info['eer'] = 1 - tpr[s]
        else:
            info['eer'] = 1 - tnr[s]
        info['eerThreshold'] = scores[perm[s]]
        return tpr, tnr, info


def performances(scores, labels, vids, eer=None, eerth=None):
    if not eer and not eerth:
        TPR, TNR, Info = vl_roc(scores, labels)
        EER = Info['eer']*100
        threashold = Info['eerThreshold']
    else:
        EER = eer
        threashold = eerth
    attacks_labels = np.unique(labels[labels < 0]).tolist()
    perf = {}
    if len(attacks_labels) == 2:
        al = ['replay', 'print']
        perf['replay'] = []
        perf['print'] = []
        perf['max'] = []
    elif len(attacks_labels) == 1:
        if attacks_labels[0] == -2:
            al = ['replay']
            perf['replay'] = []
        elif attacks_labels[0] == -1:
            al = ['print']
            perf['print'] = []
        else:
            print('Error in roc_utils.py --> performances();  gt labels should -1 or -2! it is something else!!!')
            sys.exit()
    else:
        print('Error in roc_utils.py --> performances(), len(attacks_labels) should be 1 or 2, but it is something else!!!')
        sys.exit()
    real_scores = scores[labels > 0]
    BPCER = (np.sum(real_scores < threashold) / len(real_scores)) * 100

    for i in range(len(attacks_labels)):
        attack_scores = scores[labels == attacks_labels[i]]
        perf[al[i]] += [(np.sum(attack_scores >= threashold) / len(attack_scores)) * 100]
        perf[al[i]] += [BPCER]
        perf[al[i]] += [(perf[al[i]][0] + perf[al[i]][1]) / 2.0]
    if len(attacks_labels) > 1:
        perf['max'] += [max(perf['replay'][0], perf['print'][0])]
        perf['max'] += [max(perf['replay'][1], perf['print'][1])]
        perf['max'] += [max(perf['replay'][2], perf['print'][2])]
    return threashold, EER, perf, al


def myeval(score_fname, outfile_name, eer=None, eerth=None, eval_type=None):
    scores = []
    labels = []
    vids = []
    print('score_fname: ',score_fname)
    with open(score_fname) as f:
        for line in f:
            strs = line.split(',')
            scores += [float(strs[0])]
            labels += [int(strs[1])]
            vids += [strs[2].strip()]
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    labels = labels.astype(int)
    eerth, eer, perf, al = performances(scores, labels, vids,  eer=eer, eerth=eerth)
    num_vids = len(vids)
    hard_vids = []
    for i in range(num_vids):
        if labels[i] > 0 and scores[i] < eerth:
            hard_vids += [vids[i]]
        if labels[i] < 0 and scores[i] >= eerth:
            hard_vids += [vids[i]]
    outfile = open(outfile_name, 'w')
    strResult = ''
    heading = ''
    if len(al) == 2:
        heading = 'EER-Th\tEER\tAPCER-REPLAY\tBPCER-REPLAY\tACER-REPLAY\tAPCER-PRINT\tBPCER-PRINT\tACER-PRINT\tAPCER-MAX\tBPCER-MAX\tACER-MAX\n'
        outfile.write(heading)
        strResult = '{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n'.\
                      format(
                             eerth, eer,
                             perf['replay'][0], perf['replay'][1], perf['replay'][2],
                             perf['print'][0], perf['print'][1], perf['print'][2],
                             perf['max'][0], perf['max'][1], perf['max'][2]
                             )
    elif len(al) == 1:
        if al[0] == 'replay':
            heading = 'EER-Th\tEER\tAPCER-REPLAY\tBPCER-REPLAY\tACER-REPLAY\n'
            outfile.write(heading)
            strResult = '{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n'.format(eerth, eer, perf['replay'][0], perf['replay'][1], perf['replay'][2])

        elif al[0] == 'print':
            heading = 'EER-Th\tEER\tAPCER-PRINT\tBPCER-PRINT\tACER-PRINT\n'
            outfile.write(heading)
            strResult = '{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n'.format(eerth, eer, perf['print'][0], perf['print'][1], perf['print'][2])

        else:
            pass
    outfile.write(strResult)
    outfile.close()
    return hard_vids, eer, eerth, strResult, heading
    # if eval_type:
    #     if 'Casia2Replay' in eval_type or 'Replay2Casia' in eval_type or 'Oulu2Oulu' in eval_type  or 'Siw2Siw' in eval_type:
    #         return hard_vids, eer, eerth, strResult, heading
    #     else:
    #         return hard_vids
    # else:
    #     return hard_vids

def perform_hter(scores, labels, vids, eer=None, eerth=None):
    if not eer and not eerth:
        TPR, TNR, Info = vl_roc(scores, labels)
        EER = Info['eer']*100
        EERTh = Info['eerThreshold']
    else:
        EER = eer
        EERTh = eerth

    real_scores = scores[labels > 0]
    FRR = ( np.sum(real_scores < EERTh) / len(real_scores) ) * 100    # FRR -- False Rejection Rate - lives are considered as spoof, and thus the system falsely rejected the actual person

    attack_scores = scores[labels < 0]
    FAR = ( np.sum(attack_scores >= EERTh) / len(attack_scores) ) * 100   # FAR -- False Acceptance Rate -- spoofs are consdiered as live, thus the system falsely accept the sppof
    # print('real num: {}, FN num : {}, FNR: {}'.format(len(real_scores), np.sum(real_scores < EERTh), FRR))
    # print('spoof num: {}, FP num : {}, FPR: {}'.format(len(attack_scores), np.sum(attack_scores >= EERTh), FAR))
    # print()

    HTER = (FAR + FRR) / 2.0
    
    return EER, EERTh, HTER


def val_eval_hter(score_fname,iterations, eer=None, eerth=None): # --- need to specify eer and eerth here
    
    scores = []
    labels = []
    vids = []
    print('score_fname: ',score_fname)
    with open(score_fname) as f:
        for line in f:
            strs = line.split(',')
            scores += [float(strs[0])]
            labels += [int(strs[1])]
            vids += [strs[2].strip()]

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    labels = labels.astype(int)
    eerth, eer, perf, al = performances(scores, labels, vids,  eer=eer, eerth=eerth)
    # print('iter: {}, eer: {}, eer_th:{}'.format(iterations, eer, eerth))
    _, _, hter = perform_hter(scores, labels, vids, eer=eer, eerth=eerth)
    num_vids = len(vids)
    hard_vids = []
    for i in range(num_vids):
        if labels[i] > 0 and scores[i] < eerth:
            hard_vids += [vids[i]]
        if labels[i] < 0 and scores[i] >= eerth:
            hard_vids += [vids[i]]

    return eer, eerth, hter