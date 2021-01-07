from os.path import join
from os import listdir
import random
import sys

def get_frm_list(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, flag, fldnm, split):
    fpath = join(img_path, fldnm)
    flist = sorted(listdir(fpath))
    num_frms = len(flist)
    frm_ids = list(range(num_frms))
    if mode == 'train' and flag != 1 and (disc_frms > 0):
        if num_frms > (disc_frms+10):
            disc_rids = set(random.sample(range(len(frm_ids)), disc_frms))
            frm_ids = set(frm_ids)
            frm_ids = frm_ids - disc_rids
            frm_ids = list(frm_ids)
        elif sel_these_many > 0:
            frm_ids = set(random.sample(range(len(frm_ids)), sel_these_many))
            frm_ids = list(frm_ids)
        else:
            pass
    elif (mode == 'test' or mode == 'val') and (sel_every > 1) and (sel_these_many == 0) and (len(frm_ids) > 2 * sel_every):
        frm_ids = frm_ids[::sel_every]
    elif (mode == 'test' or mode == 'val') and (sel_every == 1) and (sel_these_many > 0) and (len(frm_ids) > sel_these_many):
        frm_ids = set(random.sample(range(len(frm_ids)), sel_these_many))
        frm_ids = list(frm_ids)
    flist = list(flist[i] for i in frm_ids)
    return flist


def read_proto_file(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, fname, part, labels, gtFlags, scores, split, net_type, datasetID = None, num_cls = 2):
    num_live_exmps = 0
    num_spoof_exmps = 0
    print(">>>> Sel_these_many : {}".format(sel_these_many))
    j1 = 0
    with open(fname) as f:
        for line in f:
            j1 += 1
            sstr = line.split(',')
            flag = int(sstr[0])
            fldnm = sstr[1].strip()
            flist = get_frm_list(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, flag, fldnm, split)
            for i in flist:
                sfname = i.split('.')
                if datasetID:
                    id = join(datasetID, fldnm, sfname[0])
                else:
                    id = join(fldnm, sfname[0])
                part[mode].append(id)
                if flag == 1:
                    labels[id] = 0  # real
                else:
                    if num_cls == 3:
                        if flag == -1:
                            labels[id] = 1 # print spoof
                        elif flag == -2:
                            labels[id] = 2  # replay spoof
                    elif num_cls == 2:
                        labels[id] = 1  # spoof
                    else:
                        print('error found in get_examples_labels_oulu.py --> read_proto_file(): num_cls value is not correct!!!')
                        sys.exit()
                if mode == 'test':
                    gtFlags[id] = flag
            if mode == 'test' or mode == 'val':

                scores[fldnm] = [[], flag]
            if flag == 1:
                num_live_exmps += len(flist)
            elif (flag == -1) or (flag == -2):
                num_spoof_exmps += len(flist)
    print('>>>> Total videos: {}'.format(str(j1)))
    return part, labels, gtFlags, scores, num_live_exmps, num_spoof_exmps


def get_part_labels(mode, protocol, proto_path, img_path, sel_every, sel_these_many, disc_frms, split, net_type, eval_on_oulu_devset = False, small_trainset = False, datasetID = None, num_cls = 2):
    fname = ''
    if mode == 'train':
        if small_trainset:
            fname = join(proto_path, 'Train_debug.txt')
        else:
            fname = join(proto_path, 'Train.txt')
    elif mode == 'val':
        fname = join(proto_path,'Dev_mod.txt')
    elif mode == 'test':

        fname = join(proto_path, 'Test.txt')
    part = {}
    part.setdefault(mode, [])
    labels = {}
    scores = {}
    gtFalgs = {}
    part, labels, gtFalgs, scores, num_live_exmps, num_spoof_exmps = \
        read_proto_file(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, fname, part, labels, gtFalgs, scores, split, net_type, datasetID = datasetID, num_cls = num_cls)
    num_examples = num_live_exmps + num_spoof_exmps

    assert(len(part[mode]) == num_examples )
    assert(len(labels) ==  num_examples)
    if mode == 'test':
        assert(len(gtFalgs) == num_examples)
    return part, labels, gtFalgs, scores, num_examples


def get_examples_labels(datset_path, mode, protocol, split, sel_every, sel_thesemany, img_path, net_type, eval_on_oulu_devset = False, small_trainset = False, datasetID = None, num_cls = 2):
    disc_frms = 0
    sel_these_many = sel_thesemany
    proto_path = ''
    if (protocol == 1) or (protocol == 2) or (protocol == 5) or (protocol == 100):
        proto_path = join(datset_path, 'Protocols', 'Protocol_{}'.format(protocol))
    elif (protocol == 3) or (protocol == 4):
        proto_path = join(datset_path, 'Protocols', 'Protocol_{}'.format(protocol), 'Split_{}'.format(split))
    if mode == 'train' : 
        if protocol == 1:
            disc_frms = 109
            sel_these_many = 40
        elif protocol == 2:
            disc_frms = 69
            sel_these_many = 0
        elif protocol == 3:
            if split == 1 or split == 3 or split == 5:
                disc_frms = 107 #109
                sel_these_many = 40
            elif split == 2:
                disc_frms = 106 #109
                sel_these_many = 40
            elif split == 4:
                disc_frms = 108  # 109
                sel_these_many = 40
            elif split == 6:
                disc_frms = 110  # 109
                sel_these_many = 28
            else:
                pass
        elif protocol == 4:
            if split == 1 or split == 3 or split == 4 or split == 5:
                disc_frms = 69
                sel_these_many = 0
            elif split == 2 or split == 6:
                disc_frms = 71
                sel_these_many = 0
        elif protocol == 5:
            disc_frms = 999999999999999
            # sel_these_many = 33 # I commented it out
            sel_these_many = 0
    part, labels, gtFlags, scores, num_exmps = get_part_labels(mode, protocol, proto_path, img_path, sel_every,
                                                               sel_these_many, disc_frms, split, net_type,
                                                               eval_on_oulu_devset = eval_on_oulu_devset,
                                                               small_trainset = small_trainset,
                                                               datasetID = datasetID, num_cls = num_cls)

    return part, labels, gtFlags, scores, num_exmps