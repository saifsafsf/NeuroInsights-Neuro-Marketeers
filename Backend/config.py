import argparse


def get_config(case):
    if case == 'train_decoding': 
        # args config for training EEG-To-Text decoder
        parser = argparse.ArgumentParser(description='Specify config args for training EEG-To-Text decoder')
        
        parser.add_argument('-m', '--model_name', default = "BrainTranslator" ,required=True)
        parser.add_argument('-t', '--task_name', default = "task1", required=True)
        
        parser.add_argument('-1step', '--one_step', dest='skip_step_one', action='store_true')
        parser.add_argument('-2step', '--two_step', dest='skip_step_one', action='store_false')

        parser.add_argument('-pre', '--pretrained', dest='use_random_init', action='store_false')
        parser.add_argument('-rand', '--rand_init', dest='use_random_init', action='store_true')
        
        parser.add_argument('-load1', '--load_step1_checkpoint', dest='load_step1_checkpoint', action='store_true')
        parser.add_argument('-no-load1', '--not_load_step1_checkpoint', dest='load_step1_checkpoint', action='store_false')

        parser.add_argument('-ne1', '--num_epoch_step1', type = int, default = 20, required=True)
        parser.add_argument('-ne2', '--num_epoch_step2', type = int, default = 30, required=True)
        parser.add_argument('-lr1', '--learning_rate_step1', type = float, default = 0.00005, required=True)
        parser.add_argument('-lr2', '--learning_rate_step2', type = float, default = 0.0000005, required=True)
        parser.add_argument('-b', '--batch_size', type = int, default = 32, required=True)
        
        parser.add_argument('-s', '--save_path', default = './checkpoints/decoding', required=True)
        parser.add_argument('-subj', '--subjects', default = 'ALL', required=False)
        parser.add_argument('-eeg', '--eeg_type', default = 'GD', required=False)
        parser.add_argument('-band', '--eeg_bands', nargs='+', default = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] , required=False)
        parser.add_argument('-cuda', '--cuda', default = 'cuda:0')
        
        args = vars(parser.parse_args())

    return args