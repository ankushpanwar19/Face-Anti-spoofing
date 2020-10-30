import numpy as np


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
        EERTh = Info['eerThreshold']
    else:
        EER = eer
        EERTh = eerth
    real_scores = scores[labels > 0]
    FRR = ( np.sum(real_scores < EERTh) / len(real_scores) ) * 100

    attack_scores = scores[labels < 0]
    FAR = ( np.sum(attack_scores >= EERTh) / len(attack_scores) ) * 100
    HTER = (FAR + FRR) / 2.0
    return EER, EERTh, HTER


def myeval(score_fname, outfile_name, eer=None, eerth=None, attacks_labels=-1):
    scores = []
    labels = []
    vids = []
    with open(score_fname) as f:
        for line in f:
            strs = line.split(',')
            scores += [float(strs[0])]
            labels += [int(strs[1])]
            vids += [strs[2].strip()]
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    labels = labels.astype(int)
    EER, EERTh, HTER = performances(scores, labels, vids,  eer=eer, eerth=eerth)
    num_vids = len(vids)
    hard_vids = []
    for i in range(num_vids):
        if labels[i] > 0 and scores[i] < eerth:
            hard_vids += [vids[i]]
        if labels[i] < 0 and scores[i] >= eerth:
            hard_vids += [vids[i]]
    outfile = open(outfile_name, 'w')
    heading = 'EER \t EER-Th \t HTER\n'
    outfile.write(heading)
    strResult = '{:.8f}\t{:.8f}\t{:.8f}\n'.format(EER, EERTh, HTER)
    outfile.write(strResult)
    outfile.close()
    return hard_vids, strResult, heading