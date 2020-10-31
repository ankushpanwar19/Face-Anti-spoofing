from torch.utils import data
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import time
import torchvision.transforms as transforms
from numpy import random
import os
from os.path import join
import sys
import os.path

class FaceAntiSpoof(Dataset):
    def __init__(self, machine, config_dl, part, labels, mode, res, app_feats, net_type='resnet', transform = None):
        self.machine = machine
        self.config_dl = config_dl
        self.part = part
        self.labels = labels
        self.mode = mode
        self.transform = transform
        self.res = res
        self.app_feats = app_feats
        self.net_type = net_type

        imfname, facial_text_file, dataset = self.get_dataset_paths(self.part[0], self.config_dl, self.machine, self.mode)
        # print('Facial_text_file path: {}'.format(facial_text_file))

    def __len__(self):
        return len(self.part)

    def __getitem__(self, idx):
        # t = time.time()
        ID = self.part[idx]
        imfname, facial_text_file, dataset = self.get_dataset_paths(self.part[idx], self.config_dl, self.machine, self.mode)
        im = Image.open(imfname)

        if self.net_type == 'anet':
            imgh = im.size[0]
            imgw = im.size[1]
            if (self.res[0] != imgh) and (self.res[1] != imgw):
                im = im.resize(self.res, resample=Image.BILINEAR)
        else:
            if facial_text_file and self.mode == 'train':
                if random.randint(2):
                    im = self.getCroppedFace(facial_text_file, im)
            elif (facial_text_file and self.mode == 'val') or (facial_text_file and self.mode == 'test'):
                im = self.getCroppedFace(facial_text_file, im)

        if self.transform is not None:
            im = self.transform(im)

        label = self.labels[ID]
        assert ((label == 0 ) or (label == 1) or (label == 2))
        label = np.array(label)

        return im, self.part[idx], torch.from_numpy(label).long()


    def get_dataset_paths(self, str, configdl, machine, mode):
        dataset = None
        splitStr = str.split('/')
        datasetID = splitStr[0]
        if datasetID == 'si':
            dataset = 'siw'
        elif datasetID == 'ou':
            dataset = 'oulu-npu'
        elif datasetID == 'ra':
            dataset = 'replay-attack'
        elif datasetID == 'rm':
            dataset = 'replay-mobile'
        elif datasetID == 'ca':
            dataset = 'casia'
        elif datasetID == 'ms':
            dataset = 'msu'
        else:
            pass

        imgPath = None
        if mode == 'train':
            if datasetID == 'si' or datasetID == 'ou':
                imgPath = configdl[dataset]['img_path_machine{}'.format(machine)]
            full_img_path = configdl[dataset]['full_img_path_machine{}'.format(machine)]
            if imgPath:
                if configdl['crop_plus_full_frames'] == True:
                    if random.randint(2):
                        img_path = imgPath
                    else:
                        img_path = full_img_path
                else:
                    img_path = imgPath
            else:
                img_path = full_img_path
        elif mode == 'val' and datasetID == 'ou':
            img_path = configdl[dataset]['val_img_path_machine{}'.format(machine)]
        elif mode == 'test' and datasetID == 'ou':
            img_path = configdl[dataset]['test_img_path_machine{}'.format(machine)]
        elif mode == 'test' and datasetID == 'si':
            img_path = configdl[dataset]['test_img_path_machine{}'.format(machine)]
        elif mode == 'val' and datasetID == 'ca':
            img_path = configdl[dataset]['full_img_path_machine{}'.format(machine)]
        elif mode == 'test' and datasetID == 'ca':
            img_path = configdl[dataset]['full_img_path_test_machine{}'.format(machine)]
        elif mode == 'val' and datasetID == 'ra':
            img_path = configdl[dataset]['full_img_path_dev_machine{}'.format(machine)]
        elif mode == 'test' and datasetID == 'ra':
            img_path = configdl[dataset]['full_img_path_test_machine{}'.format(machine)]

        elif mode == 'val' and datasetID == 'ms':
            img_path = configdl[dataset]['full_img_path_machine{}'.format(machine)]
        elif mode == 'test' and datasetID == 'ms':
            img_path = configdl[dataset]['full_img_path_test_machine{}'.format(machine)]
        else:
            print('>>>>>>>>>>>>  Error: from data_loader_anet.py --> get_dataset_paths() <<<<<<<<<<<<<<<<<<<<')
            print(
                'for dataset other than siw, casia, oulu-npu, code has not been implemented...put code here for validating or testing on other dataset')
            print(
                'STEPS to follow: add required path in data_loader_da.yaml under the specific dataset you want to use for testing')
            print(' add code for that dataset here!!')
            sys.exit()

        if datasetID == 'si':
            img_path = join(img_path, splitStr[1], splitStr[2], splitStr[3], '{}.png'.format(splitStr[4]))
        elif datasetID == 'ou':
            img_path = join(img_path, splitStr[1], '{}.png'.format(splitStr[2]))
        else:
            img_path = join(img_path, splitStr[1], splitStr[2], '{}.png'.format(splitStr[3]))

        facialBBoxesPath = None
        if (datasetID == 'ra' or datasetID == 'ca' or datasetID == 'ms') and mode == 'train':
            facialBBoxesPath = configdl[dataset]['facial_bboxes_path_machine{}'.format(machine)]
        elif datasetID == 'ra' and mode == 'val':
            facialBBoxesPath = configdl[dataset]['facial_bboxes_path_dev_machine{}'.format(machine)]
        elif (datasetID == 'ca' or datasetID == 'ms') and mode == 'val':
            facialBBoxesPath = configdl[dataset]['facial_bboxes_path_machine{}'.format(machine)]
        elif (datasetID == 'ra' or datasetID == 'ca' or datasetID == 'ms') and mode == 'test':
            facialBBoxesPath = configdl[dataset]['facial_bboxes_path_test_machine{}'.format(machine)]
        else:
            pass

        facial_text_file = None
        if not facialBBoxesPath:
            pass
            # print('no facial bboxes path specified')
        if facialBBoxesPath and configdl[dataset]['include_cropped_faces_{}'.format(mode)]:
            facial_text_file = join(facialBBoxesPath, splitStr[1], splitStr[2], '{}.txt'.format(splitStr[3]))
            if not os.path.isfile(facial_text_file):
                facial_text_file = None

        return img_path, facial_text_file, dataset

    def getCords(self, facial_text_file):
        file = open(facial_text_file, 'r')
        str = file.readlines()
        x1 = None
        y1 = None
        x2 = None
        y2 = None
        if str:
            str = str[0].strip()
            str = str.split(',')
            x1 = float(str[0].strip())
            y1 = float(str[1].strip())
            x2 = float(str[2].strip())
            y2 = float(str[3].strip())
        else:
            x1 = -1
            y1 = -1
            x2 = -1
            y2 = -1
        file.close()
        return x1, y1, x2, y2

    def getCroppedFace(self, facial_text_file, im):
        x1, y1, x2, y2 = self.getCords(facial_text_file)
        if (x1 > -1) and (y1 > -1) and (x2 > -1) and (y2 > -1):
            im = im.crop((x1, y1, x2, y2))
        return im

class MOT_faceantispoof(Dataset): # THE FILE 'part' HAS TO BE CHANGED
    def __init__(self, machine, config_dl, part, labels, mode, res, app_feats, net_type='resnet', transform = None, const = False):

        self.machine = machine
        self.config_dl = config_dl
        self.part = part    # part
        self.labels = labels        # labels
        self.mode = mode
        self.transform = transform
        self.res = res
        # self.dms = depth_map_size
        self.app_feats = app_feats
        self.net_type = net_type

        self.frames_per_vid = config_dl['frames_per_vid']
        self.sample_per_vid = config_dl['sample_per_vid']
        print('???? Sample per vid: ',config_dl['sample_per_vid'],self.sample_per_vid)
        self.const = const
        self.video_parts = []
        print('++++++++++  self.app_feats : {} ++++++++++++++'.format(self.app_feats))
        print('SELF CONST: ',self.const)
        # self.dmap_path = dmap_path
        print(len(self.part))
        for items in self.part: # --- video names
            split_str = items.split('/')
            if 'ou' in items:
                str_vid = join(split_str[0],split_str[1])
            else:
                str_vid = join(split_str[0],split_str[1],split_str[2])
            if not str_vid in self.video_parts:
                self.video_parts.append(str_vid)
        # print('len vid part: {}'.format(len(self.video_parts)))
        vid_path, facial_text_file, dataset = self.get_vid_paths(self.video_parts[0], self.config_dl, self.machine, self.mode)
        # print('Facial_text_file path: {}'.format(facial_text_file))
        # print('vid path: {}'.format(vid_path))


    def __len__(self):
        return len(self.video_parts*self.sample_per_vid)

    # --- idx should be changed into vid_id
    def __getitem__(self, idx, dim_frame = 0): # --- num_vid_part: decide the starting point of frame number
        # t = time.time()
        vid_id = idx // self.sample_per_vid # --- the video name index 
        num_vid_part = idx % self.sample_per_vid # --- split the video into sample_per_vid subvideos

        # ID = self.video_parts[idx]
        ID = self.video_parts[vid_id]
        for keys in self.part:
            if ID in keys:
                ID = keys
        
        vid_name, facial_text_path, dataset = self.get_vid_paths(self.video_parts[vid_id], self.config_dl, self.machine, self.mode)
        vid_frame = self.frames_per_vid
        
        range_max = len(os.listdir(vid_name)) - 2*vid_frame + 1 # --- maximum value for start_fr
        # try: 
        if range_max <= 2*self.sample_per_vid:
            imf_list = [i for i in range(1,vid_frame+1)]
        else:
            sec_st = np.floor(num_vid_part*range_max/self.sample_per_vid) + 1
            sec_end = np.floor((num_vid_part+1)*range_max/self.sample_per_vid)
            if self.const == False: 
                start_fr = random.randint(sec_st,sec_end)
            else: 
                start_fr = int((sec_st + sec_end)//2) # --- constant 
            imf_list = [start_fr+i for i in range(0,2*vid_frame,2)]
        


        imf_list.sort()
       

        vid_list = [] # --- list of actual frames in videos

        #--- Randomly select 64 frames in the target video and combine them as input
        for index in imf_list:
            imfname = join(vid_name,'{:05d}.png'.format(index))
            flag = 0
            if facial_text_path == None:
                flag = 1
                facial_text_file = None
            if flag == 0:
                facial_text_file = join(facial_text_path,'{:05d}.txt'.format(index))
            im = Image.open(imfname)
            if facial_text_file == None:
                pass
            elif 'msu-mfsd' in facial_text_file:
                while os.path.isfile(facial_text_file) == False:
                    index += 1
                    facial_text_file = join(facial_text_path,'{:05d}.txt'.format(index))
            
            ## First crop, then do the transforms
            if self.net_type == 'anet':
                imgh = im.size[0]
                imgw = im.size[1]
                if (self.res[0] != imgh) and (self.res[1] != imgw):
                    im = im.resize(self.res, resample=Image.BILINEAR)
            else:

                if facial_text_file and self.mode == 'train':
                    ## randomly crop the faces to train
                    if random.randint(2):
                        im = self.getCroppedFace(facial_text_file, im)
                elif (facial_text_file and self.mode == 'val') or (facial_text_file and self.mode == 'test'):
                    im = self.getCroppedFace(facial_text_file, im)


            if self.transform is not None:
                im = self.transform(im)
            im = torch.unsqueeze(im,dim_frame)
            vid_list.append(im)
        vid = torch.cat(vid_list,dim = dim_frame) 
        
        label = self.labels[ID]
        assert ((label == 0 ) or (label == 1) or (label == 2))
        label = np.array(label)

        # print(vid.size())
        # return im, self.part[idx], torch.from_numpy(label).long()

        return vid,self.video_parts[vid_id], torch.from_numpy(label).long()
        # return im.float(), self.part[idx], torch.from_numpy(label).long()


    def get_vid_paths(self, str, configdl, machine, mode):
        dataset = None
        splitStr = str.split('/') ##--- the string should be the file name
        datasetID = splitStr[0]
        if datasetID == 'si':
            dataset = 'siw'
        elif datasetID == 'ou':
            dataset = 'oulu-npu'
        elif datasetID == 'ra':
            dataset = 'replay-attack'
        elif datasetID == 'rm':
            dataset = 'replay-mobile'
        elif datasetID == 'ca':
            dataset = 'casia'
        elif datasetID == 'ca-ma':
            dataset = 'casia-maged'
        elif datasetID == 'ra-ma':
            dataset = 'replay-attack-maged'
        elif datasetID == 'ms':
            dataset = 'msu'
        else:
            pass

        # ----- use this if you want to use only crop faces of siw and oulu
        # if datasetID == 'si' or datasetID == 'ou':
        #     img_path = configdl[dataset]['img_path_machine{}'.format(machine)]
        # else:
        #     img_path = configdl[dataset]['full_img_path_machine{}'.format(machine)]

        # ----- use this if you want to use both cropped and full versions of images of SiW and Oulu

        # imgPath = None
        if mode == 'train':
            if datasetID == 'si' or datasetID == 'ou':
                imgPath = configdl[dataset]['img_path_machine{}'.format(machine)]
            full_img_path = configdl[dataset]['full_img_path_machine{}'.format(machine)]
            if datasetID == 'si' or datasetID == 'ou':
                if configdl['crop_plus_full_frames'] == True:
                    if random.randint(2):
                        img_path = imgPath
                    else:
                        img_path = full_img_path
                else:
                    img_path = imgPath
            else:
                img_path = full_img_path

        elif mode == 'val' and datasetID == 'ou':
            img_path = configdl[dataset]['val_img_path_machine{}'.format(machine)]
        elif mode == 'test' and datasetID == 'ou':
            img_path = configdl[dataset]['test_img_path_machine{}'.format(machine)]
        elif mode == 'test' and datasetID == 'si':
            img_path = configdl[dataset]['test_img_path_machine{}'.format(machine)]
        elif mode == 'val' and (datasetID == 'ca' or datasetID == 'ca-ma' or datasetID == 'ms'):
            img_path = configdl[dataset]['full_img_path_machine{}'.format(machine)]
        elif mode == 'test' and (datasetID == 'ca' or datasetID == 'ca-ma' or datasetID == 'ms'):
            img_path = configdl[dataset]['full_img_path_test_machine{}'.format(machine)]
        elif mode == 'val' and (datasetID == 'ra' or datasetID == 'ra-ma'):
            img_path = configdl[dataset]['full_img_path_dev_machine{}'.format(machine)]
        elif mode == 'test' and (datasetID == 'ra' or datasetID == 'ra-ma'):
            img_path = configdl[dataset]['full_img_path_test_machine{}'.format(machine)]        

        else:
            print('>>>>>>>>>>>>  Error: from data_loader_anet.py --> get_dataset_paths() <<<<<<<<<<<<<<<<<<<<')
            print(
                'for dataset other than siw, casia, oulu-npu, code has not been implemented...put code here for validating or testing on other dataset')
            print(
                'STEPS to follow: add required path in data_loader_resnet_da.yaml under the specific dataset you want to use for testing')
            print(' add code for that dataset here!!')
            sys.exit()

        if datasetID == 'si':
            # img_path = join(img_path, splitStr[1], splitStr[2], splitStr[3], '{}.png'.format(splitStr[4]))
            vid_path = join(img_path, splitStr[1], splitStr[2], splitStr[3])
        elif datasetID == 'ou':
            # img_path = join(img_path, splitStr[1], '{}.png'.format(splitStr[2]))
            vid_path = join(img_path,splitStr[1])
        else:
            # img_path = join(img_path, splitStr[1], splitStr[2], '{}.png'.format(splitStr[3]))
            vid_path = join(img_path, splitStr[1], splitStr[2])

        facialBBoxesPath = None

        if (datasetID == 'ra' or datasetID == 'ca' or datasetID == 'ca-ma' or datasetID == 'ra-ma' or datasetID == 'ms') and mode == 'train':
            
            facialBBoxesPath = configdl[dataset]['facial_bboxes_path_machine{}'.format(machine)]
        elif (datasetID == 'ra' or datasetID == 'ra-ma') and mode == 'val':
            facialBBoxesPath = configdl[dataset]['facial_bboxes_path_dev_machine{}'.format(machine)]
        elif (datasetID == 'ca' or datasetID == 'ca-ma' or datasetID == 'ms') and mode == 'val':
            facialBBoxesPath = configdl[dataset]['facial_bboxes_path_machine{}'.format(machine)]
        elif (datasetID == 'ra' or datasetID == 'ca' or datasetID == 'ca-ma' or datasetID == 'ra-ma' or datasetID == 'ms') and mode == 'test':
            
            facialBBoxesPath = configdl[dataset]['facial_bboxes_path_test_machine{}'.format(machine)]
        else:
            pass


        # if (datasetID == 'ra' or datasetID == 'ca') and mode == 'train':
        #     # print('USING BBOXES FOR CA/RA TRAIN')
        #     facialBBoxesPath = configdl[dataset]['facial_bboxes_path_machine{}'.format(machine)]
        # elif datasetID == 'ra' and mode == 'val':
        #     facialBBoxesPath = configdl[dataset]['facial_bboxes_path_dev_machine{}'.format(machine)]
        # elif datasetID == 'ca' and mode == 'val':
        #     facialBBoxesPath = configdl[dataset]['facial_bboxes_path_machine{}'.format(machine)]
        # elif (datasetID == 'ra' or datasetID == 'ca') and mode == 'test':
        #     # print('USING BBOXES FOR CA/RA TEST')
        #     facialBBoxesPath = configdl[dataset]['facial_bboxes_path_test_machine{}'.format(machine)]
        # else:
        #     pass

        facial_text_file = None
        # if facialBBoxesPath and configdl[dataset]['include_cropped_faces_{}'.format(mode)]:
        #     facial_text_file = join(facialBBoxesPath, splitStr[1], splitStr[2], '{}.txt'.format(splitStr[3]))
        #     if not os.path.isfile(facial_text_file):
        #         facial_text_file = None

        if facialBBoxesPath and configdl[dataset]['include_cropped_faces_{}'.format(mode)]:
            facial_text_path = join(facialBBoxesPath, splitStr[1], splitStr[2])
            if not os.path.isdir(facial_text_path):
                facial_text_path = None
        else:
            facial_text_path = None

        # print('vid path: {}'.format(vid_path))

        ## return information: video path, path to facial bboxes and dataset type
        return vid_path,facial_text_path, dataset


    def getCords(self, facial_text_file):
        file = open(facial_text_file, 'r')
        str = file.readlines()
        x1 = None
        y1 = None
        x2 = None
        y2 = None
        if str:
            str = str[0].strip()
            str = str.split(',')
            x1 = float(str[0].strip())
            y1 = float(str[1].strip())
            x2 = float(str[2].strip())
            y2 = float(str[3].strip())
        else:
            x1 = -1
            y1 = -1
            x2 = -1
            y2 = -1
        file.close()
        return x1, y1, x2, y2


    def getCroppedFace(self, facial_text_file, im):
        # print('getting cropped face')
        x1, y1, x2, y2 = self.getCords(facial_text_file)
        if (x1 > -1) and (y1 > -1) and (x2 > -1) and (y2 > -1):
            im = im.crop((x1, y1, x2, y2))
        return im

def get_MOT_loader(machine, config_dl, part, labels, mode, drop_last, **params): # get_loader_all(config_dl, part_all[mode], labels_all, mode, drop_last, **params)

    print('>>>> data_loader_dnet.py --> get_MOT_loader() --> drop_last: {}'.format(drop_last))

    app_feats = params['app_feats']
    num_workers = params['num_workers']
    dataset_name = params['dataset_name']
    res = params['res']
    shuffle = params['shuffle']
    batch_size = params['batch_size']
    depth_map_size = params['depth_map_size']
    net_type = params['net_type']
    if 'const_testset' in params.keys():
        const = params['const_testset']
    else: 
        const = False
    print('>>> get MOT loader: CONST: {}'.format(const))
    transform_list = []
    transform = None

    if not net_type == 'anet': ## if not using anet, which is most of the cases
    ## do all transforms 
        transform_list = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] + transform_list
        transform_list = [transforms.ToTensor()] + transform_list
        transform_list = [transforms.RandomHorizontalFlip()] + transform_list if mode == 'train' else transform_list
        transform_list = [transforms.RandomResizedCrop(224)] + transform_list if mode == 'train' else transform_list
        transform_list = [transforms.CenterCrop(224)] + transform_list if mode == 'test' or mode == 'val' else transform_list
        transform_list = [transforms.Resize(256)] + transform_list if mode == 'test' or mode == 'val' else transform_list  # RandomResizedCrop RandomSizedCrop
    else:
        transform_list = [transforms.ToTensor()]

    if transform_list:
        transform = transforms.Compose(transform_list)

    dataset = MOT_faceantispoof(machine, config_dl, part, labels, mode, res, app_feats, net_type, transform,const = const)
    # dataset = FaceAntiSpoof(machine, config_dl, part, labels, mode, res, depth_map_size, app_feats, transform)
    # data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last)
    # data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=num_workers, drop_last=drop_last)

    return data_loader

def get_loader(machine, config_dl, part, labels, mode, drop_last, **params):
    print('>>>> data_loader_dnet.py --> get_loader() --> drop_last: {}'.format(drop_last))
    app_feats = params['app_feats']
    num_workers = params['num_workers']
    dataset_name = params['dataset_name']
    res = params['res']
    shuffle = params['shuffle']
    batch_size = params['batch_size']
    depth_map_size = params['depth_map_size']
    net_type = params['net_type']

    transform_list = []
    transform = None

    if not net_type == 'anet':
        transform_list = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] + transform_list
        transform_list = [transforms.ToTensor()] + transform_list
        transform_list = [transforms.RandomHorizontalFlip()] + transform_list if mode == 'train' else transform_list
        transform_list = [transforms.RandomResizedCrop(224)] + transform_list if mode == 'train' else transform_list
        transform_list = [transforms.CenterCrop(224)] + transform_list if mode == 'test' or mode == 'val' else transform_list
        transform_list = [transforms.Resize(256)] + transform_list if mode == 'test' or mode == 'val' else transform_list  # RandomResizedCrop RandomSizedCrop
    else:
        transform_list = [transforms.ToTensor()]

    if transform_list:
        transform = transforms.Compose(transform_list)

    dataset = FaceAntiSpoof(machine, config_dl, part, labels, mode, res, app_feats, net_type, transform)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last) #* num_workers=num_workers,
    return data_loader


