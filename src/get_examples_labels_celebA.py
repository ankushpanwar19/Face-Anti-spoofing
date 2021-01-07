from os.path import join,splitext
from os import listdir
import random
import sys
import json

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
    '''currently only for train
    =
    '''
    num_live_exmps = 0
    num_spoof_exmps = 0
    if datasetID:
        dsID = datasetID
    else:
        dsID = 'none'
    with open(fname) as f:
        files=json.load(f)
        for imgpath in files.keys(): #imgpath='Data/train/4980/spoof/000003.jpg'
            # print(imgpath)
            path_split=splitext(imgpath)[0].split('/') 
            path_id=join('ce',path_split[2],path_split[3],path_split[4]) #path_id='4980/spoof/000003' (removed jpg)
            part[mode].append(path_id)
            label_id=files[imgpath][43]
            labels[path_id]=label_id
            gtFlags[path_id]=label_id
            if label_id==0:
                num_live_exmps+=1
            else:
                num_spoof_exmps+=1
    return part, labels, gtFlags, scores, num_live_exmps, num_spoof_exmps

def read_proto_file_txt(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, fname, part, labels, gtFlags, scores, split, net_type, datasetID = None, num_cls = 2):
    '''currently only for train
    =
    '''
    num_live_exmps = 0
    num_spoof_exmps = 0
    if datasetID:
        dsID = datasetID
    else:
        dsID = 'none'
    with open(fname) as f:
        for line in f:
            # print(imgpath)
            sstr = line.split(' ')
            imgpath=sstr[0]
            label_id=int(sstr[1].strip())
            path_split=splitext(imgpath)[0].split('/') 
            path_id=join('ce',path_split[2],path_split[3],path_split[4]) #path_id='4980/spoof/000003' (removed jpg)
            part[mode].append(path_id)
            # label_id=files[imgpath][43]
            labels[path_id]=label_id
            gtFlags[path_id]=label_id
            if label_id==0:
                num_live_exmps+=1
            else:
                num_spoof_exmps+=1
    return part, labels, gtFlags, scores, num_live_exmps, num_spoof_exmps


def get_part_labels(mode, protocol, proto_path, img_path, sel_every, sel_these_many,
                    disc_frms, split, net_type, small_trainset = False, datasetID = None, num_cls = 2):
    fname = ''
    if mode == 'train':
        fname = join(proto_path, 'train_label_t.txt')
    elif mode=='val':
        fname = join(proto_path, 'train_label_v.txt')
    elif mode == 'test':
        fname = join(proto_path, 'test_label.txt')
    # print('>>>>> get_examples_labels_casia.py --> get_part_labels() --> proto-fname: {} '.format(fname))
    # print('Reading protocol file := [{}]'.format(fname))
    part = {}
    part.setdefault(mode, [])
    labels = {}
    scores = {}
    gtFalgs = {}
    part, labels, gtFalgs, scores, num_live_exmps, num_spoof_exmps = \
        read_proto_file_txt(mode, protocol, img_path, sel_every, sel_these_many, disc_frms, fname, part, labels, gtFalgs, scores, split, net_type, datasetID = datasetID, num_cls = num_cls)
    num_examples = num_live_exmps + num_spoof_exmps
    print('Number of {} examples {}; num live {}; num spoof {}'.format(mode, str(num_examples), str(num_live_exmps), str(num_spoof_exmps)))
    assert(len(part[mode]) == num_examples )
    assert(len(labels) ==  num_examples)
    if mode == 'test':
        assert(len(gtFalgs) == num_examples)
    return part, labels, gtFalgs, scores, num_examples


def get_examples_labels(datset_path, mode, protocol, split, sel_every, sel_thesemany, img_path, net_type, small_trainset = False, datasetID = None, num_cls = 2):
    if mode == 'train' or mode == 'val':
        disc_frms = 193
        sel_these_many = 52

    elif mode == 'test':
        disc_frms = 0
        sel_these_many = sel_thesemany
    else:
        pass
    proto_path = ''
    if (protocol == 0):
        proto_path = join(datset_path, 'metas', 'intra_test','myprotocol')
    elif (protocol == 1):
        proto_path = join(datset_path, 'metas', 'protocol{}'.format(protocol))
    part, labels, gtFlags, scores, num_exmps = get_part_labels\
        (mode, protocol, proto_path, img_path, sel_every, sel_these_many, disc_frms,
         split, net_type, small_trainset = small_trainset, datasetID = datasetID, num_cls = num_cls)
    return part, labels, gtFlags, scores, num_exmps



if __name__ == "__main__":
    
    datset_path='/scratch/apanwar/CelebA-Spoof/CelebA_Spoof/'
    mode='train'
    protocol=0
    split=0
    sel_every=0 
    sel_thesemany=0,
    img_path='/scratch/apanwar/CelebA-Spoof/CelebA_Spoof/Data/train'
    net_type='lstmmot' 
    small_trainset = False
    datasetID = 'Ce'
    num_cls = 2
    get_examples_labels(datset_path, mode, protocol, split, sel_every, sel_thesemany, img_path, net_type, small_trainset = False, datasetID = None, num_cls = 2)
