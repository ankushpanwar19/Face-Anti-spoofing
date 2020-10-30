from run_eval_msu import get_dataloader_test
from os.path import join
from run_eval import eval
from roc_utils import myeval
from roc_utils_hter import myeval as myeval_hter
from roc_utils import val_eval_hter

def get_class_scores(trainer, config, configdl, eval_type, min_l2norm = None, max_l2norm = None, EER = None, EERTh = None):

    debug = config['debug']
    machine = config['machine']
    anet_loss = config['anet_clsnet']['cost_func']
    exp_name = config['exp_name']
    epoc_cnt = config['epoc_cnt']
    iterations = config['iterations']
    base_path = config['base_path_machine{}'.format(machine)]
    project_path = config['project_path']
    test_dataset = config['test_dataset']
    config['eval_type'] = eval_type

    test_loader, num_exmps, scores, gtFlags, batch_size, num_test_batches, img_path = get_dataloader_test(config, configdl, debug)
    
    sub_base_path = join(config['base_out_path'],project_path,config['out_train_dataset'])
    # eval_outpath = join(sub_base_path,exp_name,strProto, strSplit, 'eval', eval_type)
    eval_outpath = join(sub_base_path,exp_name,'eval', eval_type)
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

        # _, _, _, score_file_anetsc, evalOutpath, _ = \
        score_file_anetsc,evalOutpath = \
        eval(config, trainer, test_dataset, iterations, epoc_cnt, eval_outpath, test_loader, num_exmps, scores, gtFlags, debug, min_l2norm_val_previous=min_l2norm, max_l2norm_val_previous=max_l2norm, anet_loss=anet_loss)

        eval_anetsc_HTER = join(evalOutpath, 'eval_anet_HTER_{:08d}.txt'.format(iterations + 1))
        eval_anetsc_ACER = join(evalOutpath, 'eval_anet_ACER_{:08d}.txt'.format(iterations + 1))

        if EERTh:
            hard_vids, resultHTER, headingHTER = myeval_hter(score_file_anetsc, eval_anetsc_HTER, eer=EER, eerth=EERTh)
            _, _, _, resultACER, headingACER = myeval(score_file_anetsc, eval_anetsc_ACER, eer=EER, eerth=EERTh)

            return hard_vids, headingHTER, resultHTER, headingACER, resultACER

        else:
            hard_vids, eer, eerth, strResult, heading = myeval(score_file_anetsc, eval_anetsc_ACER)
            _,_,val_anet_hter = val_eval_hter(score_file_anetsc,iterations, eer= None, eerth = None)
            return hard_vids, eer, eerth, strResult, heading, val_anet_hter

