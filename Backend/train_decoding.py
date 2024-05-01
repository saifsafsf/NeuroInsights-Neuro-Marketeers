import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import pickle
import json
import time
import copy
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from data import ZuCo_dataset
from model_decoding import BrainTranslator
from config import get_config


def train_model(dataloaders, device, model, optimizer, scheduler, num_epochs=25, checkpoint_path_best = './checkpoints/decoding/best/temp_decoding.pt', checkpoint_path_last = './checkpoints/decoding/last/temp_decoding.pt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
    print("Check: 01")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for input_embeddings, input_masks, input_mask_invert, target_ids in tqdm(dataloaders[phase]):
                
                # load in batch
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                
                """replace padding ids in target_ids with -100"""
                target_ids_batch[target_ids_batch == tokenizer_phi.pad_token_id] = -100 
              
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                print("Shape:", input_embeddings_batch.shape)
                seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)

                """calculate loss"""
                loss = seq2seqLMoutput.loss # use the BART language modeling loss

                # backward + optimize only if in training phase
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * input_embeddings_batch.size()[0] # batch loss

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f}')

            # deep copy the model
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                '''save checkpoint'''
                torch.save(model.state_dict(), checkpoint_path_best)
                
                print(f'update best on dev checkpoint: {checkpoint_path_best}')
        
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    print(f'Best val loss: {best_loss:4f}')

    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)


if __name__ == '__main__':
    args = get_config('train_decoding')

    ''' config param'''
    dataset_setting = 'unique_sent'
    
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    
    batch_size = args['batch_size']
    model_name = args['model_name']
    task_name = args['task_name']
    save_path = args['save_path']

    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    use_random_init = args['use_random_init']

    if use_random_init and skip_step_one:
        step2_lr = 5*1e-4
        
    print(f'using model: {model_name}')
    
    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'
    
    if use_random_init:
        save_name = 'randinit_' + save_name

    output_checkpoint_name_best = save_path + f'/best/{save_name}.pt' 
    output_checkpoint_name_last = save_path + f'/last/{save_name}.pt' 

    subject_choice = args['subjects']
    eeg_type_choice = args['eeg_type']
    bands_choice = args['eeg_bands']
    
    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():
        dev = args['cuda'] 
    
    else:  
        dev = "cpu"

    device = torch.device(dev)
    
    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = './dataset/ZuCo/task1-SR/task1-SR-dataset.pickle' 
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = './dataset/ZuCo/task2-NR/task2-NR-dataset.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))


    """save config"""
    with open(f'./config/decoding/{save_name}.json', 'w+') as out_config:
        json.dump(args, out_config, indent = 4)

    model_name = 'google/pegasus-xsum'
    tokenizer_phi = PegasusTokenizer.from_pretrained(model_name)
    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer_phi, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer_phi, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}

    print('train_set size: ', len(train_set))
    print('dev_set size: ', len(dev_set))
    
    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=0)
    
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size = 1, shuffle=False, num_workers=0)
    
    # dataloaders
    dataloaders = {'train': train_dataloader, 'dev': val_dataloader}

    ''' set up model '''
    model_name = "google/pegasus-xsum"
    model_phi = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    
    model = BrainTranslator(model_phi, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    model.to(device)
    
    ''' training loop '''
    ######################################################
    
    for name, param in model.named_parameters():
        if param.requires_grad and 'pretrained' in name:
            if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                continue
            else:
                param.requires_grad = False

    if skip_step_one:
        if load_step1_checkpoint:
            stepone_checkpoint = 'path_to_step_1_checkpoint.pt'
            print(f'skip step one, load checkpoint: {stepone_checkpoint}')
            model.load_state_dict(torch.load(stepone_checkpoint))
        else:
            print('skip step one, start from scratch at step two')

    else:
        ''' set up optimizer and scheduler'''
        optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)
        exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)

        print('=== start Step1 training ... ===')
        
        show_require_grad_layers(model)
        model = train_model(dataloaders, device, model, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)

    ######################################################
    '''step two trainig: update whole model for a few iterations'''
    ######################################################

    for name, param in model.named_parameters():
        param.requires_grad = True

    ''' set up optimizer and scheduler'''
    optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)
    exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=30, gamma=0.1)
    
    print()
    print('=== start Step2 training ... ===')
    show_require_grad_layers(model)
    
    '''main loop'''
    trained_model = train_model(dataloaders, device, model, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)