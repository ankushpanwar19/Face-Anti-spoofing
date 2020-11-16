# --- Train DG with ResNetDG model --- 

import torch
from utils import get_config, make_dir, write_loss, get_exp_name, write2tensorBoardX, write2TextFile, writeHardVids2Text
from utils_dg import get_data_loader as get_dataloader_train
from utils_dg import get_MOT_loader_all as get_dataloader_train_mot
from os.path import join
import tensorboardX
import os
import sys
import random
import numpy as np

from init_eval_replay_casia import get_class_scores as get_class_scores_replay_casia
from init_eval_msu import get_class_scores as get_class_scores_msu
from init_eval_oulu import get_class_scores as get_class_scores_oulu
from eval_all_set import get_val_scores_all
import matplotlib.pyplot as plt
import math
import yaml
import torch.nn as nn
from trainer_resnet_dg import Trainer_ResNetMOTDG
import argparse
from trainer_cvpr2018 import TrainerCvpr2018, TrainerCvpr2018Dg#, TrainerCvpr2019_backbone #*


def get_batch(d1, d2):
    img_batch = torch.cat((d1[0], d2[0]), 0)
    sample_id = d1[1] + d2[1]
    gt_labels = torch.cat((d1[2], d2[2]), 0)
    return img_batch, sample_id, gt_labels


# --- Evaluation part
def full_evaluation_lstm(trainer, config, config_dl, min_hter, iterations):
    flag = 0
    print('>>> Lstm Eval <<< ')
    eval_out_path = config['eval_out_path']
    print('>>> Eval out path : {} <<<'.format(eval_out_path))
    names = ['anet', 'lstm', 'comb']
    eval_out_file2 = join(eval_out_path, '{}_dev_{:08d}.txt'.format(config['test_dataset'], iterations + 1))
    eval_out_file3 = join(eval_out_path, '{}_test_HTER_{:08d}.txt'.format(config['test_dataset'], iterations + 1))
    eval_out_file4 = join(eval_out_path, '{}_test_ACER_{:08d}.txt'.format(config['test_dataset'], iterations + 1))
    evalOutFile2 = open(eval_out_file2, 'w')
    evalOutFile3 = open(eval_out_file3, 'w')
    evalOutFile4 = open(eval_out_file4, 'w')
    mode_eval1 = config['eval_type1']
    print('MODE EVAL: ', mode_eval1)
    # --- Get validation scores
    _, eer, eerth, strResult, heading, val_hter_list = get_val_scores_all(trainer, config, config_dl, mode_eval1) #* call eval_all_set.py

    print('eer: {}, eerth: {}'.format(eer, eerth))
    if 'replay-attack' in config['test_dataset'] or 'casia' in config['test_dataset']:
        mode_eval2 = config['eval_type2']
        _, headingHTER, resultHTER, headingACER, resultACER = get_class_scores_replay_casia(trainer, config, config_dl,
                                                                                            mode_eval2, EER=eer,
                                                                                            EERTh=eerth)

    if 'msu' in config['test_dataset']:
        mode_eval2 = config['eval_type2']
        _, headingHTER, resultHTER, headingACER, resultACER = get_class_scores_msu(trainer, config, config_dl,
                                                                                   mode_eval2, EER=eer, EERTh=eerth)

    if 'oulu-npu' in config['test_dataset']:
        mode_eval2 = config['eval_type2']
        _, headingHTER, resultHTER, headingACER, resultACER = get_class_scores_oulu(trainer, config, config_dl,
                                                                                    mode_eval2, EER=eer, EERTh=eerth)

    for i in range(3):
        evalOutFile2.write(heading[i])
        evalOutFile2.write(strResult[i])
        evalOutFile2.write(str(val_hter_list[i]) + '\n')

        print('>>> Iterations: {}; min_hter: {} <<< '.format(iterations + 1, min_hter))
        if val_hter_list[i] <= min_hter:
            min_hter = val_hter_list[i]

        evalOutFile3.write(headingHTER[i])
        evalOutFile3.write(resultHTER[i])
        evalOutFile4.write(headingACER[i])
        evalOutFile4.write(resultACER[i])

        result_hter_list = resultHTER[i].split('\t')
        HTER = float(result_hter_list[2])
        train_writer.add_scalar('{}_{}_eval/{}/val_hter'.format(config['test_dataset'], config['exp_name'], names[i]),
                                val_hter_list[i], iterations + 1)
        train_writer.add_scalar('{}_eval/{}/test_HTER'.format(config['exp_name'], names[i]), HTER, iterations + 1)
        train_writer.add_scalar('{}_eval/{}/eer_th'.format(config['exp_name'], names[i]), eerth[i], iterations + 1)

    evalOutFile2.close()
    evalOutFile3.close()
    evalOutFile4.close()

    if flag == 1:  # --- IF current result HTER is the best
        if config['resume'] == False:
            chkp_loc = trainer.save(checkpoint_directory, iterations)
        else:
            chkp_loc = trainer.save(resume_checkpoint_dir, iterations)
        print('checkpoint saved at: {}'.format(chkp_loc))
    return min_hter

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------main code  start here ----------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --- parser configuration file outside 
parser = argparse.ArgumentParser(description='Face anti-spoofing domain generalization')
parser.add_argument('--config_path', type=str, default='', metavar='DIR', help='path to config file')
parser.add_argument('--lr', type=float, default=0.0003, help='help text here.')
parser.add_argument('--lr_policy', type=str, default='ConstantLR')
parser.add_argument('--da_lambda_cnn', type=float, default=0.2, help='help text here.')
parser.add_argument('--da_lambda_lstm', type=float, default=0.2, help='help text here.')
parser.add_argument('--da_gamma', type=float, default=1, help='help text here.')
parser.add_argument('--batch_size_cnn', type=int, default=8)
parser.add_argument('--batch_size_lstm', type=int, default=1)
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--machine', type=int, default=0)
parser.add_argument('--max_iter', type=int, default=100000)
parser.add_argument('--mute_cnn', type=int, default=0)
parser.add_argument('--no_dg_lstm', type=int, default=0)
parser.add_argument('--out_path', type=str, default='None')
parser.add_argument('--project_path', type=str, default='None')
parser.add_argument('--optim_type', type=str, default='sgd')
parser.add_argument('--set_seed', type=int, default=1)
parser.add_argument('--eval_at_first_iter', type=int, default=0)
parser.add_argument('--net_type', type=str, default='lstmmot')
parser.add_argument('--sample_per_vid', type=int, default=8)
parser.add_argument('--new_convhead', type=int, default=1)
parser.add_argument('--mute_lstm', type=int, default=0)
parser.add_argument('--only_bline', type=int, default=0)
parser.add_argument('--use_cvpr_2019_bbone', type=int, default=0)

# --- New arguments
parser.add_argument('--data_configure', )


args = parser.parse_args()
# --- parser end

config_fname = args.config_path
config = get_config(config_fname)
if args.mute_cnn == 1:
    config['mute_cnn'] = True
else:
    config['mute_cnn'] = False

config['base_out_path'] = args.out_path
config['project_path'] = args.project_path
config['no_dg_lstm'] = args.no_dg_lstm
config['new_convhead'] = args.new_convhead
config['mute_lstm'] = args.mute_lstm
config['only_bline'] = args.only_bline
config['use_cvpr_2019_bbone'] = args.use_cvpr_2019_bbone
if args.eval_at_first_iter == 1:
    config['eval_at_first_iter'] = True
else:
    config['eval_at_first_iter'] = False

print(' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> all command line inputs  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print('[da_lambda_cnn: {}] [da_lambda_lstm: {}] [mute_cnn: {}] [no_dg_lstm: {}]\n [out_path: {}] \n [project_path: {}]'.
      format(args.da_lambda_cnn, args.da_lambda_lstm, args.mute_cnn, args.no_dg_lstm, args.out_path, args.project_path))
print('>>>> [max iteration: {}] <<<<'.format(args.max_iter))
print()
config['da_lambda_cnn'] = args.da_lambda_cnn
config['da_lambda_lstm'] = args.da_lambda_lstm
config['optim']['lr'] = args.lr
config['scheduler']['lr_policy'] = args.lr_policy
config['optim']['da_lambda_constant'] = args.da_lambda_cnn
config['batch_size_cnn'] = args.batch_size_cnn
config['batch_size_lstm'] = args.batch_size_lstm
config['optim']['num_epoch'] = args.num_epoch
config['machine'] = args.machine
config['scheduler']['parameters']['max_iter'] = args.max_iter
config['optim']['optim_type'] = args.optim_type
net_type = args.net_type
config['net_type'] = net_type
# config_fname = '/Users/ankushpanwar/Downloads/course videos/Master Project/fas-code/src/configs/data_loader_dg.yaml' #*
config_fname = 'src/configs/data_loader_dg.yaml' #*
config_dl = get_config(config_fname)  # --- Config file for dataloader
config_dl['sample_per_vid'] = args.sample_per_vid
config['set_seed'] = args.set_seed
# config['base_path_machine{}'.format(0)]="/Volumes/moustachos/" #*
min_hter = 999
num_frames = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['eval_on_oulu_devset'] = False
mode = config['mode']
protocol = config['protocol']
split = config['split']
exp_name_prefix = config['exp_name_prefix']
num_workers = config['num_workers']
train_dataset = config['train_dataset']
project_path = config['project_path']
base_out_path = config['base_out_path'] #*
base_path = config['base_path_machine{}'.format(0)]
out_train_dataset = config['out_train_dataset']

dataset_path = join(base_path, 'datasets', train_dataset)
img_path = join(dataset_path, 'rgb_images', mode)
depth_map_path = join(dataset_path, 'depth_images', 'depth-maps-train')
machine = config['machine']

print(' +++ Net type : {} +++'.format(net_type))
config['net_type'] = net_type
sub_base_path = join(base_out_path, project_path, out_train_dataset)


if ('debug' not in exp_name_prefix) and not config['resume']:
    make_dir(sub_base_path)
    exp_list = [f for f in os.listdir(sub_base_path) if 'exp' in f]
    if not exp_list:
        make_dir(join(sub_base_path, '{}_exp_000'.format(config['net_type'])))
    exp_name = get_exp_name(sub_base_path, config['net_type'])
elif config['resume']:
    exp_name = config['resume_checkpoint_path'].split('/')[-4] + '_resume'
else:
    exp_name = exp_name_prefix
config['exp_name'] = exp_name

out_path = join(sub_base_path, config['exp_name'])
make_dir(out_path)
print('>>>> out path: ', out_path)
checkpoint_path = join(out_path, 'checkpoints')
make_dir(checkpoint_path)
visualresults_path = join(out_path, 'visual_results')
make_dir(visualresults_path)
print('checkpoint_path [{}]'.format(checkpoint_path))

config['device'] = device
config['img_path'] = img_path
config['depth_map_path'] = depth_map_path
config['checkpoint_path'] = checkpoint_path
config['config_fname'] = config_fname
config['dataset_path'] = dataset_path
seed = 123
print("Random Seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
print('>>>>>>>>>>>>>> SETTING random.seed({}) np.random.seed({}) torch.manual_seed({}) <<<<<<<<<<<<<<<<<<'.format(
    seed, seed, seed))
if str(config['device']) == 'cuda':
    if config['multi_gpu']:
        torch.cuda.manual_seed_all(seed)  # seeding for cuda with multiple gpu
        print('>>>>>>>>>>>>>> SETTING torch.cuda.manual_seed_all({})  <<<<<<<<<<<<<<<<<<'.format(seed))
    else:
        torch.cuda.manual_seed(seed)  # seeding for cuda with single gpu
        print('>>>>>>>>>>>>>> SETTING torch.cuda.manual_seed({})  <<<<<<<<<<<<<<<<<<'.format(seed))


trainer = Trainer_ResNetMOTDG(config)

if config['device'] == 'cuda':
    print(' *** model transferred to {} ***'.format(device))
if config['multi_gpu']:
    if torch.cuda.device_count() > 1:
        print(">>>>>>>>>>>>>>>  Let's use", torch.cuda.device_count(), "GPUs! <<<<<<<<<<<<<<<<<<")
        trainer = nn.DataParallel(trainer)
    else:
        print('torch.cuda.device_count()= {}, only these many GPUs are available now!'.format(
            torch.cuda.device_count()))
trainer.to(device)
print(' *** device {} *** '.format(device))
checkpoint_directory = config['checkpoint_path']
# con_strs = config['project_path'].split('/')
con_strs = out_path.split('/') #*

# configs_path = join(base_out_path, con_strs[0], con_strs[1], '{}_configs'.format(con_strs[2]))

configs_path = join(out_path,'{}_configs'.format(con_strs[-1])) #*

# eval_out_path = join(base_out_path, con_strs[0], con_strs[1], '{}_eval'.format(con_strs[2]), out_train_dataset, config['exp_name'])

eval_out_path = join(out_path, '{}_eval'.format(con_strs[-1])) #*

# tnosr_brd_path = join(base_out_path, con_strs[0], con_strs[1], '{}_tnsor_brd'.format(con_strs[2]),out_train_dataset, config['exp_name'])

tnosr_brd_path = join(out_path,'{}_tnsor_brd'.format(con_strs[-1])) #*

train_writer = tensorboardX.SummaryWriter(tnosr_brd_path)

make_dir(configs_path)
make_dir(eval_out_path)
make_dir(tnosr_brd_path)

config['eval_out_path'] = eval_out_path
config_file = join(configs_path, '{}_{}.yaml'.format(out_train_dataset, config['exp_name']))
eval_type = None
eval2run = config['eval2run']
resume_checkpoint_dir = config['resume_checkpoint_path']
iterations = trainer.resume(resume_checkpoint_dir, config) if config['resume'] == True else 0
batch_size_cnn = config['batch_size_cnn']
batch_size_lstm = config['batch_size_lstm']
num_epoch = config['optim']['num_epoch']
da_gamma = config['optim']['da_lr_schedule']['gamma']
trainer.set2train()
epoch = 0
epoc_cnt = 0
start_iters = 0
train_progress = 0

source_domain_list = []  # --- a set of dataset names for training
dset_name_list = ['Ou', 'Ca', 'Ra', 'Ms']
full_dset_name = ['oulu-npu', 'casia', 'replay-attack', 'msu']
for i in range(len(dset_name_list)):
    if dset_name_list[i] in config['train_dataset']:
        source_domain_list.append(full_dset_name[i])

print('>>> Train dataset: {} <<< '.format(source_domain_list))
num_domains = len(source_domain_list)

with open(config_file, 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)

print('>>>> num epoch: {} <<<<'.format(args.num_epoch))

start_epoch=0 # needs to change it for checkpoint load
# ### Main Training part ####
for epoch in range(start_epoch, args.num_epoch):
    vid_train_loaders = []
    vid_epock_sizes = []
    vid_train_iters = []
    
    start_iters = 0

    for domain in source_domain_list:  # --- get a number of training sets and merge it as list
        vid_train_loader, vid_epock_size, _ = get_dataloader_train_mot(config, config_dl, domain, mode,
                                                                       small_trainset=False, drop_last=True)
        vid_train_loaders += [vid_train_loader]
        vid_epock_sizes += [vid_epock_size]
        vid_train_iters += [iter(vid_train_loader)] # store iterable train dataloaders list
    vid_es_max_idx = vid_epock_sizes.index(max(vid_epock_sizes))  # --- the longest dataset
    vid_epoch_size = vid_epock_sizes[vid_es_max_idx]
    total_vid_iterations = args.num_epoch * vid_epoch_size
    print('+++++++++ [num iterations in one VID epoch: {}] [total VID iterations: {}] +++++++++'.format(vid_epoch_size,
                                                                                                        total_vid_iterations))
    
    print('start_iters: {}'.format(start_iters))

    train_loaders = []
    epock_sizes = []
    train_iters = []
    for domain in source_domain_list:  # --- get a number of training sets and merge it as list #* this is for image network
        train_loader, epock_size, _ = get_dataloader_train(config, config_dl, domain, mode,
                                                           small_trainset=False, drop_last=True)
        train_loaders += [train_loader]
        epock_sizes += [epock_size]
        train_iters += [iter(train_loader)]
    es_max_idx = epock_sizes.index(max(epock_sizes))  # --- the longest dataset
    epoch_size = epock_sizes[es_max_idx]
    total_iterations = args.num_epoch * epoch_size
    print('+++++++++ [num iterations in one epoch: {}] [total iterations: {}] +++++++++'.format(epoch_size,
                                                                                                total_iterations))
    
    if args.mute_lstm == 0: # for training video lstm network 

        print('>>> Training on LSTM <<<')
        vid_epoch_size=10
        for i in range(start_iters, vid_epoch_size):

            for riIdx in range(num_domains): #* to check if any domain > biggest domain
                if riIdx != vid_es_max_idx:
                    if i % vid_epock_sizes[riIdx] == 0 and i > 0:
                        vid_train_iters[riIdx] = iter(vid_train_loaders[riIdx])
                        print('re-init VID train_iter for [dataset: {}] [domain-epoch-size: {}]'.format(
                            source_domain_list[riIdx], i))

            video = torch.empty(batch_size_lstm * num_domains, num_frames, 3, 224, 224)  # --- 6*8*3*224*224
            sampIds = ()
            vid_gtLabels = torch.empty(batch_size_lstm * num_domains).long()  # --- live/spoof label for each frame
            sidx = 0  # --- start index in dim(0)
            eidx = batch_size_lstm  # --- end index in dim(0)
            for riIdx in range(num_domains): #* form tensor of video n label from all train domains
                video_single, samp_ids, gtlabels_video = next(vid_train_iters[riIdx])
                video[sidx:eidx, :] = video_single  # --- data input
                sampIds += samp_ids
                vid_gtLabels[sidx:eidx] = gtlabels_video
                sidx = eidx
                eidx += batch_size_lstm

            vid_dom_lab = torch.empty(
                (batch_size_lstm * num_domains)).long()  # --- 24 domain labels, like [0...0,1...1,2...2]
            sidx = 0
            eidx = batch_size_lstm
            for riIdx in range(num_domains): #* form tensor of domain label (which domain it belong to)
                vid_dom_lab[sidx:eidx] = riIdx
                sidx = eidx
                eidx += batch_size_lstm
            liveBinMask = vid_gtLabels.eq(0)  # --- items that equals to 0(live) in gtlabels
            idxLive = liveBinMask.nonzero() #* index of live examples
            liveDomainGTLables = vid_dom_lab.index_select(0, idxLive.squeeze())  # ---varying length #* selecting live domain labels
            spoofBinMask = vid_gtLabels.eq(1)
            idxSpoof = spoofBinMask.nonzero()
            spoofDomainGTLables = vid_dom_lab.index_select(0, idxSpoof.squeeze())  # ---varying length #* selecting spoof train data
            if config['anet_clsnet']['cost_func'] == 'bce':
                gt_labels = vid_gtLabels.float()

            video = video.to(config['device'])
            vid_gtLabels = vid_gtLabels.to(config['device'])
            idxLive = idxLive.to(config['device'])
            liveDomainGTLables = liveDomainGTLables.to(config['device'])
            idxSpoof = idxSpoof.to(config['device'])
            spoofDomainGTLables = spoofDomainGTLables.to(config['device'])

            if args.da_lambda_lstm >= 0:
                da_lambda = args.da_lambda_lstm
            elif args.da_lambda_lstm == -1:
                train_progress = iterations / args.max_iter
                da_lambda = (2 / (1 + math.exp(-args.da_gamma * train_progress))) - 1

            if args.no_dg_lstm == 1:
                trainer.net_update_lstm_no_dg(video, vid_gtLabels, da_lambda, batch_size_lstm, idxLive,
                                              liveDomainGTLables, idxSpoof, spoofDomainGTLables)
            else:
                trainer.net_update_lstm(video, vid_gtLabels, da_lambda, batch_size_lstm, idxLive, liveDomainGTLables,
                                        idxSpoof, spoofDomainGTLables)

            trainer.update_learning_rate()

            if (iterations + 1) % config['log_iter'] == 0:  # --- write loss
                write_loss(iterations, epoc_cnt, trainer, train_writer, config['exp_name'])

            if config['eval_at_first_iter']:
                print('>>>> eval at first iter')
                config['eval_at_first_iter'] = False
                config['epoc_cnt'] = epoc_cnt
                config['iterations'] = iterations
                min_hter = full_evaluation_lstm(trainer, config, config_dl, min_hter, iterations)
            iterations += 1
            if iterations >= args.max_iter: # finish if iterataion exceeds max 100,000
                sys.exit(">>>> Training Finished <<<<")
        config['epoc_cnt'] = epoc_cnt
        config['iterations'] = iterations
        chkp_loc = trainer.save(checkpoint_directory, iterations)
        print('checkpoint saved at: {}'.format(chkp_loc))
        min_hter = full_evaluation_lstm(trainer, config, config_dl, min_hter, iterations)

    if args.mute_cnn == 0:
        for i in range(start_iters, epoch_size):
            for riIdx in range(num_domains):
                if riIdx != es_max_idx:
                    if i % epock_sizes[riIdx] == 0 and i > 0:
                        train_iters[riIdx] = iter(train_loaders[riIdx])
                        print('re-init train_iter for [dataset: {}] [domain-epoch-size: {}]'.format(
                            source_domain_list[riIdx], i))
            imgBatch = torch.empty(batch_size_cnn * num_domains, 3, 224, 224)  # --- the batch size is actually 16*3
            sampIds = ()
            gtLabels = torch.empty(batch_size_cnn * num_domains).long()  # --- live/spoof label for each frame
            sidx = 0  # --- start index in dim(0)
            eidx = batch_size_cnn  # --- end index in dim(0)
            for riIdx in range(num_domains):
                imgs, samp_ids, gtlabels = next(train_iters[riIdx])
                imgBatch[sidx:eidx, :] = imgs  # --- data input
                sampIds += samp_ids
                gtLabels[sidx:eidx] = gtlabels
                sidx = eidx
                eidx += batch_size_cnn
            gtLabelsDomains = torch.empty(
                (batch_size_cnn * num_domains)).long()  # --- 24 domain labels, like [0...0,1...1,2...2]
            sidx = 0
            eidx = batch_size_cnn
            for riIdx in range(num_domains):
                gtLabelsDomains[sidx:eidx] = riIdx
                sidx = eidx
                eidx += batch_size_cnn
            liveBinMask = gtLabels.eq(0)  # --- items that equals to 0(live) in gtlabels
            idxLive = liveBinMask.nonzero()
            liveDomainGTLables = gtLabelsDomains.index_select(0, idxLive.squeeze())  # ---varying length
            spoofBinMask = gtLabels.eq(1)
            idxSpoof = spoofBinMask.nonzero()
            spoofDomainGTLables = gtLabelsDomains.index_select(0, idxSpoof.squeeze())  # ---varying length
            if config['anet_clsnet']['cost_func'] == 'bce':
                gt_labels = gtLabels.float()
            imgBatch = imgBatch.to(config['device'])
            gtLabels = gtLabels.to(config['device'])
            idxLive = idxLive.to(config['device'])
            liveDomainGTLables = liveDomainGTLables.to(config['device'])
            idxSpoof = idxSpoof.to(config['device'])
            spoofDomainGTLables = spoofDomainGTLables.to(config['device'])
            gtLabelsDomains = gtLabelsDomains.to(config['device'])

            if args.da_lambda_cnn >= 0:
                da_lambda = args.da_lambda_cnn
            elif args.da_lambda_cnn == -1:
                train_progress = iterations / args.max_iter
                da_lambda = (2 / (1 + math.exp(-args.da_gamma * train_progress))) - 1
            if args.only_bline == 0:
                trainer.net_update_cnn(imgBatch, gtLabels, da_lambda, batch_size_cnn, idxLive, liveDomainGTLables,
                                       idxSpoof, spoofDomainGTLables)
            elif args.only_bline == 1:
                trainer.net_update_bline(imgBatch, gtLabels, da_lambda, batch_size_cnn, idxLive, liveDomainGTLables,
                                         idxSpoof, spoofDomainGTLables)
            trainer.update_learning_rate()

            if (iterations + 1) % config['log_iter'] == 0:  # --- write loss
                write_loss(iterations, epoc_cnt, trainer, train_writer, config['exp_name'])
            iterations += 1
            if iterations >= args.max_iter:
                sys.exit(">>> Training Finished <<< ")
            if args.mute_lstm == 1 and config['eval_at_first_iter']:
                print('>>>> eval at first iter')
                config['eval_at_first_iter'] = False
                config['epoc_cnt'] = epoc_cnt
                config['iterations'] = iterations
                min_hter = full_evaluation_lstm(trainer, config, config_dl, min_hter, iterations)

        config['epoc_cnt'] = epoc_cnt
        config['iterations'] = iterations
        chkp_loc = trainer.save(checkpoint_directory, iterations)
        print('checkpoint saved at: {}'.format(chkp_loc))
        min_hter = full_evaluation_lstm(trainer, config, config_dl, min_hter, iterations)  # --- Video-wise test

    epoc_cnt += 1
train_writer.close()
