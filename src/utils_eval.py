from os.path import join
import os
# from trainer import Trainer_ANet, Trainer_DNet, Trainer_Fusion, Trainer_FusionV2, Trainer_FusionV3
# from trainer_resnet_Oct19 import Trainer_ResNet
# from trainer_DA import Trainer_DNetDA, Trainer_DNetANetDA
from utils import load_model, load_model_older_version, load_Trainer_DNetANetDA
# from run_eval import eval, get_dataloader_test
import statistics as stat

def get_minmax_vals(config, test_dataset, **best_models):
    protos_splits = config['protos_splits'][test_dataset]
    strs2match = config['strs2match']
    minc = []
    maxc = []

    for str2match in strs2match:
        for proto, splits in protos_splits.items():
            for split in splits:
                dev_results = best_models[str(proto)][str(split)][str2match]
                minc += [dev_results[4]]
                maxc += [dev_results[5]]
    minc = stat.mean(minc)
    maxc = stat.mean(maxc)
    return  minc, maxc



def get_min_max(fname_woext, eval_path, exp_name):

    len1 = len(fname_woext.split('_'))
    if 'dadpnet' in exp_name:
        if len1 == 4:
            minmax_iter = int(fname_woext.split('_')[3])
        elif len1 == 3:
            minmax_iter = int(fname_woext.split('_')[2])
        minmaxfname = join(eval_path, 'dnet_minmax_values_{:08d}.txt'.format(minmax_iter))
    else:
        minmax_iter = int(fname_woext.split('_')[3])
        minmaxfname = join(eval_path, 'pred_scores_frame_level_{}.txt'.format(minmax_iter))
    file = open(minmaxfname, 'r')
    str1 = file.read()
    file.close()
    str2 = str1.split(',')
    str3 = str2[0].strip()
    str4 = str2[1].strip()
    min = float(str3.split(':')[1])
    max = float(str4.split(':')[1])
    return min, max, minmaxfname



def getTheBestResult(eval_out_path, evalOutFile4):
    eval_fnames = [f for f in os.listdir(eval_out_path) if 'oulu_dev' in f]
    eval_fnames.sort()
    iter_list = []
    acer_list = []
    for fname in eval_fnames:
        strIter = fname.split('_')[2]
        strIter = strIter.split('.')[0]


        ffname = join(eval_out_path, fname)
        line_cnt = 0
        num_fields = None
        with open(ffname) as f:
            for line in f:
                line_cnt += 1
                if line_cnt == 1:
                    num_fields = len(line.split('\t'))
                elif line_cnt == 2:
                    splitStr = line.split('\t')
                    if num_fields == 11:
                        acer_list += [float(splitStr[10].strip())]
                    elif num_fields == 8 or num_fields == 5:
                        acer_list += [float(splitStr[4].strip())]
                    iter_list += [strIter]
    # best_acer = min(acer_list)
    best_acer_idx = acer_list.index(min(acer_list))
    iter_best = int(iter_list[best_acer_idx])
    source_file = join(eval_out_path, 'oulu_test_{:08d}.txt'.format(iter_best))
    dest_file = join(eval_out_path, evalOutFile4)
    import shutil
    shutil.copy(source_file, dest_file)
    print()

def pick_best_models(evalOutFile1, evalOutFile2, config, config_dl,  eval_type):

    debug = config['debug']
    machine = config['machine']
    composite_dataset_name = config_dl['composite_dataset_name']
    test_dataset = config_dl['test_dataset']
    config['test_dataset'] = test_dataset
    base_path = config['base_path_machine{}'.format(machine)]
    project_path = config['project_path']
    protos_splits = config['protos_splits'][test_dataset]
    exp_name = config['exp_name']
    sel_every = config_dl['test_dataset_conf'][test_dataset]['sel_every']
    strs2match = config['strs2match']
    fusiontype = config['fusion_types']

    best_models = {}
    res4TextFile = {}

    for proto, splits in protos_splits.items():
        testset_protocol = str(proto)
        best_models[testset_protocol] = {}
        res4TextFile[testset_protocol] = {}
        for split in splits:
            testset_split = str(split)
            best_models[testset_protocol][testset_split] = {}
            res4TextFile[testset_protocol][testset_split] = {}
            for str2match in strs2match:
                best_models[testset_protocol][testset_split][str2match] = []
                res4TextFile[testset_protocol][testset_split][str2match] = []


    for proto, splits in protos_splits.items():
        testset_protocol = proto

        for split in splits:
            testset_split = split

            strTestProto = 'Protocol_{}'.format(testset_protocol)
            strTestSplit = 'Split_{}'.format(testset_split)

            eval_path = join(base_path, project_path, composite_dataset_name, exp_name, 'eval', eval_type, test_dataset,
                             strTestProto, strTestSplit, 'sel_every_{}'.format(sel_every))
            if config['fusion_type'] and config['fusion_type'] > -1:
                eval_path = join(eval_path, fusiontype[int(config['fusion_type'])])

            for str2match in strs2match:

                eval_fnames = [f for f in os.listdir(eval_path) if str2match in f]
                eval_fnames.sort()

                results = []
                ACER = []
                minl2norms = []
                maxl2norms = []

                for fname in eval_fnames:
                    fname_woext = fname.split('.')[0]

                    minl2norm = None
                    maxl2norm = None
                    minmaxfname = None
                    # if 'fusion' in exp_name or 'dnet' in exp_name or ('dadpnet' in exp_name and not config['dadpnet_net_arch'] == 2) or ('dadpnet' in exp_name and not config['dadpnet_net_arch'] == 3):
                    #     minl2norm, maxl2norm, minmaxfname = get_min_max(fname_woext, eval_path, exp_name)


                    ffname = join(eval_path, fname)

                    line_cnt = 0
                    acer = None
                    num_fields = None

                    with open(ffname) as f:
                        for line in f:
                            line_cnt += 1
                            if line_cnt == 1:
                                num_fields = len(line.split('\t'))
                            if line_cnt == 2:
                                splitStr = line.split('\t')
                                if num_fields == 11:
                                    eert = float(splitStr[0])
                                    eer = float(splitStr[1])
                                    apcer1 = float(splitStr[2])
                                    bpcer1 = float(splitStr[3])
                                    acer1 = float(splitStr[4])
                                    apcer2 = float(splitStr[5])
                                    bpcer2 = float(splitStr[6])
                                    acer2 = float(splitStr[7])
                                    apcer = float(splitStr[8])
                                    bpcer = float(splitStr[9])
                                    acer = float(splitStr[10])

                                    # basically BPCR are not computed separately for different PAI (presentation attack Insturments e.g. diff dispaly screens or pritners)
                                    # only APCER is computed separately for different PAIs, so BPCER  should be one single number
                                    assert (bpcer1 == bpcer2 == bpcer)

                                    results.append([fname_woext, eert, eer, apcer1, bpcer1, acer1, apcer2, bpcer2, acer2, apcer, bpcer, acer])
                                    res4TextFile[str(testset_protocol)][str(testset_split)][str2match].append([fname_woext, eert, eer, apcer1, apcer2, apcer, bpcer, acer])

                                elif num_fields == 8 or num_fields == 5:
                                    eert = float(splitStr[0])
                                    eer = float(splitStr[1])
                                    apcer = float(splitStr[2])
                                    bpcer = float(splitStr[3])
                                    acer = float(splitStr[4])
                                    results.append([fname_woext, eert, eer, apcer, bpcer, acer])
                                    res4TextFile[str(testset_protocol)][str(testset_split)][str2match].append([fname_woext, eert, eer, apcer, bpcer, acer])
                                else:
                                    import sys
                                    print('something is wrong in the text file {}!'.format(ffname))
                                    sys.exit()

                            if line_cnt == 2:
                                break

                    ACER.append(acer)
                    # evalOutFile1.write('\n')

                    # if 'fusion' in exp_name or 'dnet' in exp_name or \
                    #         ('dadpnet' in exp_name and not config['dadpnet_net_arch'] == 2) or \
                    #         ('dadpnet' in exp_name and not config['dadpnet_net_arch'] == 3):
                    #     minl2norms.append(minl2norm)
                    #     maxl2norms.append(maxl2norm)

                # if debug:
                #     idx = 1
                #     for result in results:
                #         if len(result)-1 == 11:
                #             print('{} # {} # {:2.4f} # {:2.4f} # {:2.4f} # {:2.4f} # {:2.4f} # {:2.4f} # {:2.4f} # {:2.4f} # {:2.4f} # {:2.4f} # {:2.4f}'.
                #                   format(idx, result[0], result[1],result[2], result[3],result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11]))
                #         elif len(result)-1 == 8 or len(result)-1 == 5:
                #             print('{} # {} # {:2.4f} # {:2.4f} # {:2.4f} # {:2.4f} # {:2.4f}'.
                #                   format(idx, result[0], result[1],result[2], result[3],result[4], result[5]))
                #         idx += 1

                best_acer = min(ACER)
                best_acer_idx = ACER.index(min(ACER))

                len1 = len(results[best_acer_idx][0].split('_'))
                iterations = None
                if 'dadpnet' in exp_name:
                    if len1 == 4:
                        iterations = int(results[best_acer_idx][0].split('_')[3])
                    elif len1 == 3:
                        iterations = int(results[best_acer_idx][0].split('_')[2])
                else:
                    iterations = int(results[best_acer_idx][0].split('_')[2])

                eerth = results[best_acer_idx][1]
                eer = results[best_acer_idx][2]
                min_l2norm = max_l2norm = 1

                # if 'fusion' in exp_name or 'dnet' in exp_name or \
                #         ('dadpnet' in exp_name and not config['dadpnet_net_arch'] == 2) or \
                #         ('dadpnet' in exp_name and not config['dadpnet_net_arch'] == 3):
                #     min_l2norm = minl2norms[best_acer_idx]
                #     max_l2norm = maxl2norms[best_acer_idx]

                if debug:
                    print('*** [exp_name: {}], [min ACER: {:3.4f}] [idx: {}] [iter: {}] [eerth: {:3.4f}] [eer: {:3.4f}]'.
                          format(exp_name, best_acer, best_acer_idx + 1, iterations, eerth, eer))


                # -- results.append([fname_woext, eert, eer, apcer1, bpcer1, acer1, apcer2, bpcer2, acer2, apcer, bpcer, acer])
                # -- results.append([fname_woext, eert, eer, apcer, bpcer, acer])
                best_models[str(testset_protocol)][str(testset_split)][str2match] += [iterations, eerth, eer, best_acer, min_l2norm, max_l2norm]



    evalOutFile1.write('Evaluation results on oulu-npu development set\n')
    evalOutFile1.write('exp_name # eer-th # eer # apcer1 # apcer2 # apcer-max # bpcer # acer-max # proto # split\n')
    evalOutFile1.write('OR\n')
    evalOutFile1.write('exp_name # eer-th # eer # apcer # bpcer # acer # proto # split\n')

    evalOutFile2.write('Evaluation results on oulu-npu development set\n')
    evalOutFile2.write('protocol # split # iterations # eer-th # eer # best-apcer\n')

    for str2match in strs2match:
        evalOutFile1.write('>>>> [str2match: {}]\n'.format(str2match))
        evalOutFile2.write('>>>> [str2match: {}]\n'.format(str2match))
        for proto, splits in protos_splits.items():
            for split in splits:
                result = best_models[str(proto)][str(split)][str2match]
                evalOutFile2.write('{} # {} # {} # {:3.4f} # {:3.4f} # {:3.4f}\n'.format(proto, split, result[0], result[1], result[2], result[3]))

                re4text = res4TextFile[str(proto)][str(split)][str2match]
                for res in re4text:
                    if len(res) == 8:
                        evalOutFile1.write('{} # {:3.4f} # {:3.4f} # {:3.4f} # {:3.4f} # {:3.4f} # {:3.4f} # {:3.4f} # {} # {}\n'.
                                           format(res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], proto, split))
                        # [fname_woext, eert, eer, apcer1, apcer2, apcer, bpcer, acer]
                    elif len(res) == 6:
                        evalOutFile1.write('{} # {:3.4f} # {:3.4f} # {:3.4f} # {:3.4f} # {:3.4f} # {} # {}\n'.format(res[0], res[1], res[2], res[3], res[4], res[5], proto, split))
                        # [fname_woext, eert, eer, apcer, bpcer, acer]
                    else:
                        pass

                # evalOutFile1.write('\n')

            evalOutFile1.write('\n')
            evalOutFile2.write('\n')
        evalOutFile1.write('\n')
        evalOutFile2.write('\n')

    return best_models






def init_triner(net_type, config, fusion_net_arch):
    trainer = None
    if 'anet' in net_type:
        trainer = Trainer_ANet(config)
    elif 'dnet' in net_type:
        trainer = Trainer_DNet(config)
    elif 'fusion' in net_type:
        if fusion_net_arch == 0:
            trainer = Trainer_Fusion(config)
        elif fusion_net_arch == 1:
            trainer = Trainer_FusionV2(config)
        elif config['fusion_net_arch'] == 2:
            trainer = Trainer_FusionV3(config)
    elif 'resnet' in config['net_type']:
        trainer = Trainer_ResNet(config)
    elif 'dadpnet' in config['net_type']:
        # config['dadpnet_net_arch'] = dadpnet_net_arch
        if config['dadpnet_net_arch'] == 0:
            trainer = Trainer_DNetDA(config)
        elif config['dadpnet_net_arch'] == 1:
            if config['anet_arch'] == 1:
                config['convhead']['input_dim2'] = 384
                config['anet_clsnet']['inp_dim'] = 24576
                config['anet_clsnet']['out_dim1'] = 4096
                config['anet_clsnet']['out_dim2'] = 1
            elif config['anet_arch'] == 2:
                config['convhead']['input_dim2'] = 128
                config['anet_clsnet']['inp_dim'] = 8192
                config['anet_clsnet']['out_dim1'] = 1024
                config['anet_clsnet']['out_dim2'] = 1
            elif config['anet_arch'] == 0:
                pass
            trainer = Trainer_DNetANetDA(config)
    else:
        pass
    return trainer


# TODO ---------------------------------------------------
def load_chkpoint(trainer, model_name, device, net_type, fusion_net_arch, load_model_version):
    if load_model_version == 0:
        trainer = load_model_older_version(trainer, model_name, device, net_type)
    elif load_model_version == 1:
        trainer = load_model(trainer, model_name, device, net_type, fusion_net_arch)
    elif load_model_version == 2:
        trainer = load_Trainer_DNetANetDA(trainer, model_name, device, net_type, fusion_net_arch)
    trainer.to(device)
    return trainer


def getRes4tensorBoard(ffname, iterations):
    line_cnt = 0
    num_fields = None
    results4TensorBoard = []
    res4TextFiles = []
    with open(ffname) as f:
        for line in f:
            line_cnt += 1
            if line_cnt == 1:
                num_fields = len(line.split('\t'))
            if line_cnt == 2:
                splitStr = line.split('\t')
                if num_fields == 11:
                    eert = float(splitStr[0])
                    eer = float(splitStr[1])
                    apcer1 = float(splitStr[2])
                    bpcer1 = float(splitStr[3])
                    acer1 = float(splitStr[4])
                    apcer2 = float(splitStr[5])
                    bpcer2 = float(splitStr[6])
                    acer2 = float(splitStr[7])
                    apcer = float(splitStr[8])
                    bpcer = float(splitStr[9])
                    acer = float(splitStr[10])

                    # basically BPCR are not computed separately for different PAI (presentation attack Insturments e.g. diff dispaly screens or pritners)
                    # only APCER is computed separately for different PAIs, so BPCER  should be one single number
                    assert (bpcer1 == bpcer2 == bpcer)
                    results4TensorBoard += [0, eert, 0, acer]

                    res4TextFiles += [iterations, eert, eer, apcer1, apcer2, apcer, bpcer, acer]


                elif num_fields == 8 or num_fields == 5:
                    eert = float(splitStr[0])
                    eer = float(splitStr[1])
                    apcer = float(splitStr[2])
                    bpcer = float(splitStr[3])
                    acer = float(splitStr[4])

                    results4TensorBoard += [0, eert, 0, acer]
                    res4TextFiles += [iterations, eert, eer, apcer, bpcer, acer]

                else:
                    import sys
                    print('something is wrong in the text file {}!'.format(ffname))
                    sys.exit()

            if line_cnt == 2:
                break

    # evalOutFile.write('\n')
    return results4TensorBoard, res4TextFiles