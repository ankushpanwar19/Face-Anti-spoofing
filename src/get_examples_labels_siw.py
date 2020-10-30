from os.path import join
import random
import pickle
import sys

def get_frm_list(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, flag, fldnm, exm_class, subid, sel_every_p3_s1, split, flist_dic, merge_all_dataset = False):
    flist = flist_dic[fldnm]
    if not len(flist) == 0:
        num_frms = len(flist)
        frm_ids = list(range(num_frms))

        if (mode == 'train') and (protocol == 1):
            if num_frms >= 60:
                frm_ids = frm_ids[:60]
                if (flag == -1 or flag == -2) and (disc_frms > 0):
                    if len(frm_ids) > disc_frms+10:
                        disc_rids = set(random.sample(range(len(frm_ids)), disc_frms))
                        frm_ids = set(frm_ids)
                        frm_ids = frm_ids - disc_rids
                        frm_ids = list(frm_ids)
        elif (mode == 'train') and (protocol == 2) and (flag == 1) and (disc_frms > 0):
            if num_frms > (2*disc_frms):
                disc_rids = set(random.sample(range(len(frm_ids)), disc_frms))
                frm_ids = set(frm_ids)
                frm_ids = frm_ids - disc_rids
                frm_ids = list(frm_ids)
        elif (mode == 'train') and (protocol == 3) and (split == 1) and (flag == 1) and (disc_frms > 0):
            if num_frms > (disc_frms):
                disc_rids = set(random.sample(range(len(frm_ids)), disc_frms))
                frm_ids = set(frm_ids)
                frm_ids = frm_ids - disc_rids
                frm_ids = list(frm_ids)
                if (sel_every_p3_s1 > 1) and (len(frm_ids) > 2 * sel_every_p3_s1):
                    frm_ids = frm_ids[::sel_every_p3_s1]
        elif (mode == 'train') and (protocol == 3) and (split == 2) and (flag == -2) and (disc_frms > 0):
            if num_frms > (disc_frms):
                disc_rids = set(random.sample(range(len(frm_ids)), disc_frms))
                frm_ids = set(frm_ids)
                frm_ids = frm_ids - disc_rids
                frm_ids = list(frm_ids)
        elif (mode == 'train') and (protocol == 4):
            if (flag == -1 or flag == -2) and (disc_frms > 0):
                if len(frm_ids) > disc_frms+10:
                    disc_rids = set(random.sample(range(len(frm_ids)), disc_frms))
                    # disc_rids = set(np.random.choice(len(frm_ids), disc_frms))
                    frm_ids = set(frm_ids)
                    frm_ids = frm_ids - disc_rids
                    frm_ids = list(frm_ids)
        elif (mode == 'train') and (protocol == 5):
            if (flag == -1 or flag == -2) and (disc_frms > 0):
                if len(frm_ids) > disc_frms+10:
                    disc_rids = set(random.sample(range(len(frm_ids)), disc_frms))
                    frm_ids = set(frm_ids)
                    frm_ids = frm_ids - disc_rids
                    frm_ids = list(frm_ids)
        elif (mode == 'train') and (protocol == 6):
            if (flag == -1 or flag == -2) and (disc_frms > 0):
                if len(frm_ids) > disc_frms + 10:
                    disc_rids = set(random.sample(range(len(frm_ids)), disc_frms))
                    frm_ids = set(frm_ids)
                    frm_ids = frm_ids - disc_rids
                    frm_ids = list(frm_ids)
                else:
                    pass

        if (mode == 'train') and (protocol == 6) and merge_all_dataset == True and (sel_every > 1):
            randid = random.randint(0, 5)
            frm_ids = frm_ids[randid::sel_every]
        elif (mode == 'test') and (sel_every > 1) and (sel_these_many == 0) and (len(frm_ids) > 2*sel_every):
            frm_ids = frm_ids[::sel_every]
        elif (mode == 'test') and (sel_every == 1) and (sel_these_many > 0) and (len(frm_ids) > sel_these_many):
            frm_ids = set(random.sample(range(len(frm_ids)), sel_these_many))
            frm_ids = list(frm_ids)
        flist = list(flist[i] for i in frm_ids)  # the list comprehension approach
    else:
        print('+++++++++++ this vidid should be 021-2-3-2-1 which has zero frames after filtering ++++++++++')
    return flist


def read_proto_file(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, fname, part, labels, gtFlags, scores, sel_every_p3_s1, split, pickle_fname, net_type, num_cls = 2, datasetID = None, merge_all_dataset = False):
    flist_dic = pickle.load(open(pickle_fname, 'rb'))
    num_live_exmps = 0
    num_spoof_exmps = 0
    with open(fname) as f:
        for line in f:
            sstr = line.split(',')
            fldnm = sstr[1].strip()
            subid = fldnm.split('-')[0]
            if True:
                flag = int(sstr[0])
                if flag == 1:
                    exm_class = 'live'
                elif (flag == -1) or (flag == -2):
                    exm_class = 'spoof'
                flist = \
                    get_frm_list(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, flag,
                                 fldnm, exm_class, subid, sel_every_p3_s1, split, flist_dic, merge_all_dataset = merge_all_dataset)
                if not len(flist)==0:
                    for i in flist:
                        sfname = i.split('.')
                        if datasetID:
                            id = join(datasetID, exm_class, subid, fldnm, sfname[0])
                        else:
                            id = join(exm_class, subid, fldnm, sfname[0])
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
                                print('error found in get_examples_labels_siwk.py --> read_proto_file(): num_cls value is not correct!!!')
                                sys.exit()
                        if mode == 'test':
                            gtFlags[id] = flag
                else:
                    print('++++++++++ this video has no frames {} ++++++++++++++++'.format(fldnm))
                if mode == 'test':
                    scid = join(exm_class, subid, fldnm)
                    if net_type in 'anet' or net_type in 'dnet' or net_type in 'dadpnet' or net_type in 'resnet':
                        scores[scid] = [[], flag]
                    elif net_type in 'fusion':
                        scores[scid] = [[], [], flag]
                if exm_class == 'live':
                    num_live_exmps += len(flist)
                elif exm_class == 'spoof':
                    num_spoof_exmps += len(flist)
    return part, labels, gtFlags, scores, num_live_exmps, num_spoof_exmps


def get_part_labels(mode, protocol, proto_path, img_path, sel_every, sel_these_many, disc_frms, sel_every_p3_s1, split, pickle_fname, net_type, small_trainset = False, num_cls = 2, datasetID = None, merge_all_dataset = False):
    fname = ''
    if mode == 'train':
        if small_trainset:
            fname = join(proto_path, 'Train_debug.txt')
        else:
            fname = join(proto_path, 'Train.txt')
    elif mode == 'test':
        fname = join(proto_path, 'Test.txt')
    print('Reading protocol file := [{}]'.format(fname))
    part = {}
    part.setdefault(mode, [])
    labels = {}
    scores = {}
    gtFalgs = {}
    part, labels, gtFalgs, scores, num_live_exmps, num_spoof_exmps = \
        read_proto_file(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, fname, part, labels, gtFalgs,
                        scores, sel_every_p3_s1, split, pickle_fname, net_type, num_cls = num_cls, datasetID = datasetID, merge_all_dataset = merge_all_dataset)
    num_examples = num_live_exmps + num_spoof_exmps
    # print('Number of {} examples {}; num live {}; num spoof {}'.format(mode, str(num_examples), str(num_live_exmps), str(num_spoof_exmps)))
    assert(len(part[mode]) == num_examples )
    assert(len(labels) == len(part[mode]))
    return part, labels, gtFalgs, scores, num_examples


def get_examples_labels(datset_path, mode, protocol, split, sel_every, sel_these_many, pickle_fname, net_type, small_trainset = False, num_cls = 2, datasetID = None, merge_all_dataset = False):
    disc_frms = 0
    sel_every_p3_s1 = 1
    proto_path = ''
    if (protocol == 1) or (protocol == 4) or (protocol == 5) or (protocol == 6):
        proto_path = join(datset_path, 'Protocols', 'Protocol_{}'.format(protocol))
    elif (protocol == 2) or (protocol == 3):
        proto_path = join(datset_path, 'Protocols', 'Protocol_{}'.format(protocol), 'split_{}'.format(split))
    else:
        pass
    if (protocol == 1) and (mode == 'train'):
        disc_frms = 35
    elif (protocol == 2) and (split == 1) and (mode == 'train'):
        disc_frms = 167 #171
    elif (protocol == 2) and (split == 2) and (mode == 'train'):
        disc_frms = 254 #258
    elif (protocol == 2) and (split == 3) and (mode == 'train'):
        disc_frms = 231 #238
    elif (protocol == 2) and (split == 4) and (mode == 'train'):
        disc_frms = 237 #242
    elif (protocol == 3) and (split == 1) and (mode == 'train'):
        disc_frms = 34 #37
        sel_every_p3_s1 = 3
    elif (protocol == 3) and (split == 2) and (mode == 'train'):
        disc_frms = 14 # 11
    elif (protocol == 4) and (mode == 'train'):
        disc_frms = 131 #130
    elif (protocol == 5) and (mode == 'train'):
        disc_frms = 124 #89
    elif (protocol == 6) and (mode == 'train'):
        disc_frms = 125 #145
    img_path = join(datset_path, 'rgb_images', mode)
    # print(' +++++++++++++++ pickle_fname: {} +++++++++++++++++++'.format(pickle_fname))
    part, labels, gtFlags, scores, num_exmps = \
        get_part_labels(mode, protocol, proto_path, img_path, sel_every, sel_these_many, disc_frms, sel_every_p3_s1, split, pickle_fname, net_type,
                        small_trainset = small_trainset, num_cls = num_cls, datasetID = datasetID, merge_all_dataset = merge_all_dataset)

    return part, labels, gtFlags, scores, num_exmps