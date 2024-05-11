import numpy as np
import torch
from torch.utils.data import DataLoader
from util.construct_dataset_mat_to_pickle_v1 import mat_to_dict
from dotenv import dotenv_values
from data import ZuCo_dataset
from model_decoding import BrainTranslator
from openai import OpenAI
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
env_vars = dotenv_values(".env")


def eval_model(dataloaders, device, tokenizer, model):

    model.eval()   # Set model to evaluate mode
    running_loss = 0.0

    # Iterate over data.
    sample_count = 0
    
    pred_tokens_list = []
    pred_string_list = []
    for input_embeddings, input_masks, input_mask_invert, target_ids in dataloaders['test']:
        
        input_embeddings_batch = input_embeddings.to(device).float()
        input_masks_batch = input_masks.to(device)
        target_ids_batch = target_ids.to(device)
        input_mask_invert_batch = input_mask_invert.to(device)
        target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

        # forward
        seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)

        """calculate loss"""
        loss = seq2seqLMoutput.loss 

        logits = seq2seqLMoutput.logits
        probs = logits[0].softmax(dim = 1)
        
        _, predictions = probs.topk(1)
        
        predictions = torch.squeeze(predictions)
        predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>','',)
        predictions = predictions.tolist()

        truncated_prediction = []
        for t in predictions:
        
            if t != tokenizer.eos_token_id:
                truncated_prediction.append(t)
            else:
                break
        
        pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)
        
        pred_tokens_list.append(pred_tokens)
        pred_string_list.append(predicted_string)
        
        sample_count += 1
        running_loss += loss.item() * input_embeddings_batch.size()[0]

    predicted_para = ' '.join(pred_string_list)
    API_KEY = env_vars.get('API_KEY')
    client = OpenAI(api_key=API_KEY)

    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": """I'm going to give you a paragraph that does not make sense.
         I want you to use the same words and phrases as present in the original paragraph, and turn the paragraph
         into one in which the sentences make sense. Return this new continuous paragraph back to me. 
         Do not add any new information from yourself about anything. Use the same words and phrase, just change their
         arrangement so it makes more sense.
         """},
        {"role": "user", "content": predicted_para},
      ]
    )
    legible_predicted_para = response.choices[0].message.content
    print(legible_predicted_para)

    return legible_predicted_para


def getEegResults(file_path):
    subject_choice = "ALL"
    eeg_type_choice = "GD"
    bands_choice = ["_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"]
    
    dataset_setting = 'unique_sent'
    model_name = "BrainTranslator"

    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = torch.device('cpu')

    ''' set up dataloader '''
    whole_dataset_dicts = []
    whole_dataset_dicts.append(mat_to_dict(file_path))
        
    model_name = 'google/pegasus-xsum'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)

    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    
    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=0)

    dataloaders = {'test': test_dataloader}

    ''' set up model '''
    checkpoint_path = "./checkpoints/decoding/best/pegasus.pt"
    pretrained_bart = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    
    model = BrainTranslator(pretrained_bart, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.to(device)

    ''' eval '''
    predicted_para = eval_model(dataloaders, device, tokenizer, model)
    return predicted_para

if __name__ == "__main__":
    print(getEegResults("./dataset/ZuCo/task2-NR/mat_files/resultsZAB1_NR_0-5.mat"))