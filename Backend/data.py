import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_1d(input_tensor):
    
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean) / std

    return input_tensor


def get_input_sample(sent_obj, tokenizer, eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], max_len = 56, add_CLS_token = False):
    
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):

        frequency_features = []
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])

        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != 105*len(bands):
            return None

        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)


    def get_sent_eeg(sent_obj, bands):
        
        sent_eeg_features = []
        for band in bands:
            key = 'mean'+band
            sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])
        
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)


    if sent_obj is None:
        return None

    input_sample = {}

    target_string = sent_obj['content']
    target_tokenized = tokenizer(
        target_string, 
        padding='max_length', 
        max_length=max_len, 
        truncation=True, 
        return_tensors='pt', 
        return_attention_mask = True
    )
    
    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    
    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        return None

    # get input embeddings
    word_embeddings = []

    """add CLS token embedding at the front"""
    if add_CLS_token:
        word_embeddings.append(torch.ones(105*len(bands)))

    for word in sent_obj['word']:
        
        # add each word's EEG embedding as Tensors
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands = bands)
        
        # check none, for v2 dataset
        if word_level_eeg_tensor is None:
            return None
        
        # check nan:
        if torch.isnan(word_level_eeg_tensor).any():
            return None
            
        word_embeddings.append(word_level_eeg_tensor)
    
    # pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))

    input_sample['input_embeddings'] = torch.stack(word_embeddings)

    # mask out padding tokens
    input_sample['input_attn_mask'] = torch.zeros(max_len)

    if add_CLS_token:
        input_sample['input_attn_mask'][:len(sent_obj['word'])+1] = torch.ones(len(sent_obj['word'])+1) 
    else:
        input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word'])) 
    

    input_sample['input_attn_mask_invert'] = torch.ones(max_len) # 1 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])+1] = torch.zeros(len(sent_obj['word'])+1) # 0 is not masked
    else:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word'])) # 0 is not masked
    
    # clean 0 length data
    if len(sent_obj['word']) == 0:
        return None

    return input_sample


class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject = 'ALL', eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], setting = 'unique_sent', is_add_CLS_token = False):
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts,list):
            input_dataset_dicts = [input_dataset_dicts]
            
        for input_dataset_dict in input_dataset_dicts:
            
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
            else:
                subjects = [subject]
            
            total_num_sentence = len(input_dataset_dict[subjects[0]])
            
            train_divider = int(0.8*total_num_sentence)
            dev_divider = train_divider + int(0.1*total_num_sentence)

            if setting == 'unique_sent':
                
                if phase == 'train':

                    for key in subjects:
                        for i in range(train_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            
                            if input_sample is not None:
                                self.inputs.append(input_sample)

                elif phase == 'dev':

                    for key in subjects:
                        for i in range(train_divider,dev_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            
                            if input_sample is not None:
                                self.inputs.append(input_sample)

                elif phase == 'test':

                    for key in subjects:
                        for i in range(total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            
                            if input_sample is not None:
                                self.inputs.append(input_sample)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            input_sample['input_embeddings'],
            input_sample['input_attn_mask'], 
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids']
        )