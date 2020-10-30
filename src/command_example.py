import os

option = 1
machine = 1

da_lambda_lstm = 0.2
da_lambda_cnn = 0.2
da_gmma = 1
fl_bs = 16
vl_bs = 2
lr_policy = 'ConstantLR'

mute_cnn_0 = 0
only_bline_0 = 0
mute_lstm_0 = 0
no_dg_lstm_0 = 0

mute_cnn_1 = 1
only_bline_1 = 1
mute_lstm_1 = 1
no_dg_lstm_1 = 1

set_seed_0 = 0
set_seed_1 = 1

eval_at_first_iter = 0

net_type = 'lstmmot'
sample_per_vid = 1
num_workers = 2

base_out_path = None # <put your output dir here>

project_path = None # <put your project path here, under base_out_path>

if option == 1:
    configs = [
        ['configs/train.yaml', 0.0003, lr_policy, da_lambda_lstm, da_lambda_cnn, da_gmma, fl_bs, vl_bs, 1000, machine, 100000,base_out_path, project_path, eval_at_first_iter, set_seed_1, net_type, sample_per_vid, num_workers, mute_cnn_0, only_bline_1, mute_lstm_1, no_dg_lstm_1]]

for config in configs:
    cmd = 'python train_dg_final.py --config_path {}' \
            ' --lr {} --lr_policy {} --da_lambda_lstm {} --da_lambda_cnn {} --da_gamma {}' \
            ' --batch_size_cnn {} --batch_size_lstm {} --num_epoch {}' \
            ' --machine {} --max_iter {}' \
            ' --out_path {} --project_path {} --eval_at_first_iter  {}' \
            ' --set_seed {} --net_type {} --sample_per_vid {}' \
            ' --num_workers {} --mute_cnn {} --only_bline {} --mute_lstm {} --no_dg_lstm {}' \
            .format(config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], config[8],
            config[9], config[10], config[11], config[12], config[13], config[14], config[15], config[16],
            config[17], config[18], config[19], config[20], config[21])

    os.system(cmd)





