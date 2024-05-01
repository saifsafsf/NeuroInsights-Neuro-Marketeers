import torch.nn as nn
import torch.nn.functional as F


class BrainTranslator(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
        super(BrainTranslator, self).__init__()
        
        self.pretrained = pretrained_layers
        
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)

        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)


    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):

        encoded_embedding = self.additional_encoder(input_embeddings_batch, src_key_padding_mask = input_masks_invert)
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        out = self.pretrained(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch, return_dict = True, labels = target_ids_batch_converted)                    
        
        return out
