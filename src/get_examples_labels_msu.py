from os.path import join
from os import listdir
import random
import os
import sys


def get_frm_list(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, flag, fldnm, split):

    fpath = join(img_path, fldnm)
    flist = sorted(listdir(fpath))
    num_frms = len(flist)
    frm_ids = list(range(num_frms))
    # print('mode: ',mode)
    # print('sel_these_many :{}, sel_every: {}'.format(sel_these_many,sel_every))
    if mode == 'train' and flag != 1 and (disc_frms > 0):
        if num_frms > (disc_frms + 10):
            disc_rids = set(random.sample(range(len(frm_ids)), disc_frms))
            frm_ids = set(frm_ids)
            frm_ids = frm_ids - disc_rids
            frm_ids = list(frm_ids)

        # elif (num_frms > sel_these_many) and (sel_these_many > 0):
        #     frm_ids = set(random.sample(range(len(frm_ids)), sel_these_many))
        #     frm_ids = list(frm_ids)
        # else:
        #     pass

    elif mode == 'val' or mode == 'test':
        frm_ids = frm_ids[::sel_every]
    # elif (mode == 'val') and (sel_every == 1) and (sel_these_many > 0) and (len(frm_ids) > sel_these_many):
    #     frm_ids = set(random.sample(range(len(frm_ids)), sel_these_many))
    #     frm_ids = list(frm_ids)

    # elif (mode == 'test') and (sel_every > 1) and (sel_these_many == 0) and (len(frm_ids) > 2 * sel_every):
    #     frm_ids = frm_ids[::sel_every]
    # elif (mode == 'test') and (sel_every == 1) and (sel_these_many > 0) and (len(frm_ids) > sel_these_many):
    #     frm_ids = set(random.sample(range(len(frm_ids)), sel_these_many))
    #     frm_ids = list(frm_ids)

    flist = list(flist[i] for i in frm_ids)  # the list comprehension approach

    return flist


def read_proto_file(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, proto_file, part, labels, gtFlags, scores, split, net_type, datasetID=None, num_cls=2):
    num_live_exmps = 0
    num_spoof_exmps = 0
    if datasetID:
        dsID = datasetID
    else:
        dsID = 'none'

    with open(proto_file) as f:
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
                        print('error found in get_examples_labels_casia.py --> read_proto_file(): num_cls value is not correct!!!')
                        sys.exit()

                if mode == 'test' or mode == 'val':
                    gtFlags[id] = flag  # saving the actual gt labels (+1, -1, -2), this is for frame wise socre dumping to a text file

            if mode == 'test' or mode == 'val':
                scid = join(exm_cls, fldnm)
                scores[scid] = [[], flag]

            if flag == 1:
                num_live_exmps += len(flist)
            elif (flag == -1) or (flag == -2):
                num_spoof_exmps += len(flist)

    return part, labels, gtFlags, scores, num_live_exmps, num_spoof_exmps


def get_part_labels(mode, protocol, proto_path, img_path, sel_every, sel_these_many, disc_frms, split, net_type, small_trainset = False, datasetID=None, num_cls=2):
    proto_file = ''
    if mode == 'train' or mode == 'val':
        if small_trainset:
            proto_file = join(proto_path, 'Train_debug.txt')
        else:
            proto_file = join(proto_path, 'Train.txt')

    elif mode == 'test':
        proto_file = join(proto_path, 'Test.txt')

    # print('Reading protocol file := [{}]'.format(proto_file))
    part = {}
    part.setdefault(mode, [])
    labels = {}
    scores = {}
    gtFalgs = {}
    part, labels, gtFalgs, scores, num_live_exmps, num_spoof_exmps = \
        read_proto_file(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, proto_file, part, labels, gtFalgs, scores, split, net_type, datasetID=datasetID, num_cls=num_cls)

    num_examples = num_live_exmps + num_spoof_exmps
    print('Number of {} examples {}; num live {}; num spoof {}'.format(mode, str(num_examples), str(num_live_exmps), str(num_spoof_exmps)))
    assert(len(part[mode]) == num_examples )
    assert(len(labels) ==  num_examples)
    if mode == 'test':
        assert(len(gtFalgs) == num_examples)

    return part, labels, gtFalgs, scores, num_examples


def get_examples_labels(datset_path, mode, protocol, split, sel_every, sel_these_many, img_path, net_type, small_trainset = False, datasetID = None, num_cls = 2):

    print('*** get_examples_labels_msu.py --> get_examples_labels() ***')
    
    disc_frms = None
    if mode == 'train':
        disc_frms = 182
        print('*** [mode:{}] [sel_these_many:{}] [disc_frames:{}] ***'.format(mode, sel_these_many, disc_frms))
    elif mode == 'val':
        disc_frms = 0
        print('*** [mode:{}] [sel_every:{}] [disc_frames:{}] ***'.format(mode, sel_every, disc_frms))
    elif mode == 'test':
        disc_frms = 0
        print('*** [mode:{}] [sel_every:{}] [disc_frames:{}] ***'.format(mode, sel_every, disc_frms))
    else:
        print('incorrect mode!!!')

    proto_path = ''
    if (protocol == 1):
        proto_path = join(datset_path, 'Protocols', 'Protocol_{}'.format(protocol))

    part, labels, gtFlags, scores, num_exmps = \
        get_part_labels(mode, protocol, proto_path, img_path, sel_every, sel_these_many, disc_frms,
         split, net_type, small_trainset=small_trainset, datasetID=datasetID, num_cls=num_cls)

    return part, labels, gtFlags, scores, num_exmps