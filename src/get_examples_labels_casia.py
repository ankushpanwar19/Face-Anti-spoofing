from os.path import join
from os import listdir
import random
import sys

def get_frm_list(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, flag, fldnm, split):
    fpath = join(img_path, fldnm)
    flist = sorted(listdir(fpath))
    num_frms = len(flist)
    frm_ids = list(range(num_frms))
    if (mode == 'train' or mode == 'val') and flag != 1 and (disc_frms > 0):
        if num_frms > (disc_frms + 10):
            disc_rids = set(random.sample(range(len(frm_ids)), disc_frms))
            frm_ids = set(frm_ids)
            frm_ids = frm_ids - disc_rids
            frm_ids = list(frm_ids)
        elif (num_frms > sel_these_many) and (sel_these_many > 0):
            frm_ids = set(random.sample(range(len(frm_ids)), sel_these_many))
            frm_ids = list(frm_ids)
        else:
            pass
    elif (mode == 'test') and (sel_every > 1) and (sel_these_many == 0) and (len(frm_ids) > 2 * sel_every):
        frm_ids = frm_ids[::sel_every]
    elif (mode == 'test') and (sel_every == 1) and (sel_these_many > 0) and (len(frm_ids) > sel_these_many):
        frm_ids = set(random.sample(range(len(frm_ids)), sel_these_many))
        frm_ids = list(frm_ids)
    flist = list(flist[i] for i in frm_ids)
    return flist


def read_proto_file(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, fname, part, labels, gtFlags, scores, split, net_type, datasetID = None, num_cls = 2):
    num_live_exmps = 0
    num_spoof_exmps = 0
    if datasetID:
        dsID = datasetID
    else:
        dsID = 'none'
    with open(fname) as f:
        for line in f:
            sstr = line.split(',')
            flag = int(sstr[0])
            fldnm = sstr[1].strip()
            if flag == 1:
                exm_cls = 'live'
                img_path1 = '{}/{}'.format(img_path, exm_cls)
            elif flag == -1 or flag == -2:
                exm_cls = 'spoof'
                img_path1 = '{}/{}'.format(img_path, exm_cls)
            else:
                print('error found in get_examples_labels_casia.py --> read_proto_file(): flag value is not correct!!!')
                sys.exit()
            flist = get_frm_list(mode, protocol, img_path1, sel_every, sel_these_many, disc_frms, flag, fldnm, split)
            for i in flist:
                sfname = i.split('.')
                id = join(dsID, exm_cls, fldnm, sfname[0])
                part[mode].append(id)
                if flag == 1:
                    labels[id] = 0
                else:
                    if num_cls == 3:
                        if flag == -1:
                            labels[id] = 1
                        elif flag == -2:
                            labels[id] = 2
                    elif num_cls == 2:
                        labels[id] = 1
                    else:
                        print('error found in get_examples_labels_casia.py --> read_proto_file(): num_cls value is not correct!!!')
                        sys.exit()
                if mode == 'test' or mode == 'val':
                    gtFlags[id] = flag
            if mode == 'test' or mode == 'val':
                scid = join(exm_cls, fldnm)
                # if net_type in 'anet' or net_type in 'dnet' or net_type in 'resnet' or net_type in 'dadpnet' or net_type in 'lstmmot':
                #     scores[scid] = [[], flag]
                # elif net_type in 'fusion':
                #     scores[scid] = [[], [], flag]
                scores[scid] = [[], flag]
            if flag == 1:
                num_live_exmps += len(flist)
            elif (flag == -1) or (flag == -2):
                num_spoof_exmps += len(flist)
    return part, labels, gtFlags, scores, num_live_exmps, num_spoof_exmps


def get_part_labels(mode, protocol, proto_path, img_path, sel_every, sel_these_many,
                    disc_frms, split, net_type, small_trainset = False, datasetID = None, num_cls = 2):
    fname = ''
    if mode == 'train' or mode == 'val':
        if small_trainset:
            fname = join(proto_path, 'Train_debug.txt')
        else:
            fname = join(proto_path, 'Train.txt')
    elif mode == 'test':
        fname = join(proto_path, 'Test.txt')
    # print('>>>>> get_examples_labels_casia.py --> get_part_labels() --> proto-fname: {} '.format(fname))
    # print('Reading protocol file := [{}]'.format(fname))
    part = {}
    part.setdefault(mode, [])
    labels = {}
    scores = {}
    gtFalgs = {}
    part, labels, gtFalgs, scores, num_live_exmps, num_spoof_exmps = \
        read_proto_file(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, fname, part, labels, gtFalgs, scores, split, net_type, datasetID = datasetID, num_cls = num_cls)
    num_examples = num_live_exmps + num_spoof_exmps
    print('Number of {} examples {}; num live {}; num spoof {}'.format(mode, str(num_examples), str(num_live_exmps), str(num_spoof_exmps)))
    assert(len(part[mode]) == num_examples )
    assert(len(labels) ==  num_examples)
    if mode == 'test':
        assert(len(gtFalgs) == num_examples)
    return part, labels, gtFalgs, scores, num_examples


def get_examples_labels(datset_path, mode, protocol, split, sel_every, sel_thesemany, img_path, net_type, small_trainset = False, datasetID = None, num_cls = 2):

    #I commented it for more images
    if mode == 'train' or mode == 'val':
        disc_frms = 193
        sel_these_many = 52

    # if mode == 'train' or mode == 'val':
    #     disc_frms = 0
    #     sel_these_many = sel_thesemany

    elif mode == 'test':
        disc_frms = 0
        sel_these_many = sel_thesemany
    else:
        pass
    proto_path = '' #find protocol path
    if (protocol == 1):
        proto_path = join(datset_path, 'Protocols', 'Protocol_{}'.format(protocol))
    part, labels, gtFlags, scores, num_exmps = get_part_labels\
        (mode, protocol, proto_path, img_path, sel_every, sel_these_many, disc_frms,
         split, net_type, small_trainset = small_trainset, datasetID = datasetID, num_cls = num_cls)
    return part, labels, gtFlags, scores, num_exmps