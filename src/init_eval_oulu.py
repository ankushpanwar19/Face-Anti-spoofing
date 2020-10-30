from run_eval_oulu_npu import get_dataloader_test
from os.path import join
from run_eval import eval
from roc_utils import myeval
from roc_utils_hter import myeval as myeval_hter
from roc_utils import val_eval_hter

## THIS IS CURRENTLY USED FOR train_DA.py
def get_class_scores(trainer, config, configdl, eval_type, min_l2norm = None, max_l2norm = None, EER = None, EERTh = None):

    debug = config['debug']
    machine = config['machine']
    device = config['device']
    anet_loss = config['anet_clsnet']['cost_func']  # [3]
    exp_name = config['exp_name']
    train_dataset = config['train_dataset']
    trainset_protocol = config['protocol']
    trainset_split = config['split']
    if config['net_type'] == 'lstmmot':
        batch_size = config['batch_size_lstm']
    else:
        batch_size = config['batch_size']
    epoc_cnt = config['epoc_cnt']
    iterations = config['iterations']
    net_type = config['net_type']
    base_out_path = config['base_out_path']
    project_path = config['project_path']
    test_dataset = config['test_dataset'] ## 'replay-attack' or 'casia'
    sel_these_many = config['test_dataset_conf'][test_dataset]['sel_these_many']

    config['eval_type'] = eval_type

    if debug:
        config['batch_size'] = batch_size #config['batch_size_debug']
        config['device'] = 'cpu'
        config['test_dataset_conf'][test_dataset]['sel_these_many'] = sel_these_many
    else:
        config['batch_size'] = batch_size
        config['device'] = device
        config['test_dataset_conf'][test_dataset]['sel_these_many'] = sel_these_many


    strProto = 'Protocol_{}'.format(trainset_protocol)
    strSplit = 'Split_{}'.format(trainset_split)

    test_loader, num_exmps, scores, gtFlags, batch_size, num_test_batches, img_path = get_dataloader_test(config, configdl, debug)


    # --- this eval_outpath should be modified
    # eval_outpath = join(base_out_path, project_path, train_dataset, exp_name, strProto, strSplit, 'eval', eval_type)
    sub_base_path = join(config['base_out_path'],project_path,config['out_train_dataset'])
    eval_outpath = join(sub_base_path,exp_name,strProto, strSplit, 'eval', eval_type)
    print('>>>> eval_outpath: {}'.format(eval_outpath))

    if config['net_type'] == 'lstmmot' or config['net_type'] == 'cvpr2018dg' or config['net_type'] == 'resnet18':
        score_file_anetsc, score_file_vid_sc,score_file_combsc,evalOutpath = \
        eval(config, trainer, test_dataset, iterations, epoc_cnt, eval_outpath, test_loader, num_exmps, scores, gtFlags, debug,min_l2norm_val_previous=min_l2norm, max_l2norm_val_previous=max_l2norm, anet_loss=anet_loss)
        eval_anetsc_HTER = join(evalOutpath, 'eval_anet_HTER_{:08d}.txt'.format(iterations + 1))
        eval_anetsc_ACER = join(evalOutpath, 'eval_anet_ACER_{:08d}.txt'.format(iterations + 1))
        eval_vidsc_HTER = join(evalOutpath, 'eval_vid_HTER_{:08d}.txt'.format(iterations + 1))
        eval_vidsc_ACER = join(evalOutpath, 'eval_vid_ACER_{:08d}.txt'.format(iterations + 1))
        eval_combsc_HTER = join(evalOutpath, 'eval_comb_HTER_{:08d}.txt'.format(iterations + 1))
        eval_combsc_ACER = join(evalOutpath, 'eval_comb_ACER_{:08d}.txt'.format(iterations + 1))

        hard_vids = [0.0,0.0,0.0]
        headingACER = [0.0,0.0,0.0]
        headingHTER = [0.0,0.0,0.0]
        resultHTER = [0.0,0.0,0.0]
        resultACER = [0.0,0.0,0.0]
        strResult = [0.0,0.0,0.0]
        heading = [0.0,0.0,0.0]
        eer = [0.0,0.0,0.0]
        eerth = [0.0,0.0,0.0]
        if EERTh:

            hard_vids[0], resultHTER[0], headingHTER[0] = myeval_hter(score_file_anetsc, eval_anetsc_HTER, eer=EER[0], eerth=EERTh[0])
            _, _, _, resultACER[0], headingACER[0] = myeval(score_file_anetsc, eval_anetsc_ACER, eer=EER[0], eerth=EERTh[0], eval_type=config['eval_type'])

            hard_vids[1], resultHTER[1], headingHTER[1] = myeval_hter(score_file_vid_sc, eval_vidsc_HTER, eer=EER[1], eerth=EERTh[1])
            _, _, _, resultACER[1], headingACER[1] = myeval(score_file_vid_sc, eval_vidsc_ACER, eer=EER[1], eerth=EERTh[1], eval_type=config['eval_type'])

            hard_vids[2], resultHTER[2], headingHTER[2] = myeval_hter(score_file_combsc, eval_combsc_HTER, eer=EER[2], eerth=EERTh[2])
            _, _, _, resultACER[2], headingACER[2] = myeval(score_file_combsc, eval_combsc_HTER, eer=EER[2], eerth=EERTh[2], eval_type=config['eval_type'])

            return hard_vids, headingHTER, resultHTER, headingACER, resultACER

        else:
            hard_vids[0], eer[0], eerth[0], strResult[0], heading[0] = myeval(score_file_anetsc, eval_anetsc_ACER, eval_type=config['eval_type'])
            hard_vids[1], eer[1], eerth[1], strResult[1], heading[1] = myeval(score_file_vid_sc, eval_vidsc_ACER, eval_type=config['eval_type'])
            hard_vids[2], eer[2], eerth[2], strResult[2], heading[2] = myeval(score_file_combsc, eval_combsc_ACER, eval_type=config['eval_type'])
            # --- write val hter 
            _,_,val_anet_hter = val_eval_hter(score_file_anetsc,iterations, eer= None, eerth = None)
            _,_,val_lstm_hter = val_eval_hter(score_file_vid_sc,iterations, eer= None, eerth = None)
            _,_,val_comb_hter = val_eval_hter(score_file_combsc,iterations, eer= None, eerth = None)
            val_hter_list = [val_anet_hter,val_lstm_hter,val_comb_hter]
            return hard_vids, eer, eerth, strResult, heading, val_hter_list

    else:
        score_file_anetsc, evalOutpath = \
        eval(config, trainer, test_dataset, iterations, epoc_cnt, eval_outpath, test_loader, num_exmps, scores, gtFlags, debug, min_l2norm_val_previous=min_l2norm, max_l2norm_val_previous=max_l2norm, anet_loss=anet_loss)

        eval_anetsc_HTER = join(evalOutpath, 'eval_anet_HTER_{:08d}.txt'.format(iterations + 1))
        eval_anetsc_ACER = join(evalOutpath, 'eval_anet_ACER_{:08d}.txt'.format(iterations + 1))
        if EERTh == 0:
            print('eer Th equals to 0')
            EERTh += 1e-5
        # print('score_file_anetsc: ',score_file_anetsc)
        if EERTh:
            hard_vids, resultHTER, headingHTER = myeval_hter(score_file_anetsc, eval_anetsc_HTER, eer=EER, eerth=EERTh)
            _, _, _, resultACER, headingACER = myeval(score_file_anetsc, eval_anetsc_ACER, eer=EER, eerth=EERTh, eval_type=config['eval_type'])

            return hard_vids, headingHTER, resultHTER, headingACER, resultACER

        else:
            hard_vids, eer, eerth, strResult, heading = myeval(score_file_anetsc, eval_anetsc_ACER, eval_type=config['eval_type'])
            _,_,val_anet_hter = val_eval_hter(score_file_anetsc,iterations, eer= None, eerth = None)
            return hard_vids, eer, eerth, strResult, heading, val_anet_hter

    #  hard_vids, eer, eerth, strResult
    #  hard_vids, eer, eerth, strResult, heading
    # (score_fname, outfile_name, eer=None, eerth=None, eval_type=None):



