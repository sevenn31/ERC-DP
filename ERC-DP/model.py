import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoModel, AutoTokenizer
from transformers import RobertaTokenizer, RobertaModel

class bertaModel(nn.Module):
    def __init__(self, dropout_rate=0.1,nums=6):
        super(bertaModel, self).__init__()
        self.roberta = AutoModel.from_pretrained('')#simcse
        self.tokenizer = AutoTokenizer.from_pretrained('')#simcse
        
        self.dropout = nn.Dropout(dropout_rate)
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        for i in range(8):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(3072,3072)
        self.tanh = nn.Tanh()
        self.fc_cls = nn.Linear(3072, nums)

    
    def forward(self, utterance_input_ids, utterance_attention_mask):
        outputs = self.roberta(utterance_input_ids, attention_mask=utterance_attention_mask)
        all_token_vectors = outputs[0]  # shape: (batch_size, seq_length, hidden_size)

        # Find the positions of start and end tokens for each sentence in the batch
        start_positions = (utterance_input_ids == 0).nonzero(as_tuple=True)
        end_positions = (utterance_input_ids == 2).nonzero(as_tuple=True)

        part1_vectors_list = []
        part2_vectors_list = []
        part3_vectors_list = []
        for i in range(utterance_input_ids.size(0)):  # iterate over the batch size
            # Find the positions of start and end tokens for each sentence in the batch
            start_position = start_positions[1][start_positions[0] == i][0]
            end_position = end_positions[1][end_positions[0] == i][0]

            # Check if the sentence has part1 or part3
            if start_position > 0:
                # Split the tensor into three parts: before <s>, between <s> and </s>, after </s>
                part1_vectors = all_token_vectors[i, :start_position, :]
                part1_vectors = part1_vectors.mean(dim=0)
                part1_vectors_list.append(part1_vectors)
            else:
                part1_vectors = torch.zeros(all_token_vectors.size(-1)).to(utterance_input_ids.device)
                part1_vectors_list.append(part1_vectors)

            if end_position + 1 < all_token_vectors.size(1):
                part3_vectors = all_token_vectors[i, end_position + 1:, :]
                part3_vectors = part3_vectors.mean(dim=0)
                part3_vectors_list.append(part3_vectors)
            else:
                part3_vectors = torch.zeros(all_token_vectors.size(-1)).to(utterance_input_ids.device)
                part3_vectors_list.append(part3_vectors)
                
            # Split the tensor into part2: between <s> and </s>
            part2_vectors = all_token_vectors[i, start_position:end_position+1, :]
            part2_vectors = part2_vectors.mean(dim=0)
            part2_vectors_list.append(part2_vectors)

    
        part1_vectors_batch = torch.stack(part1_vectors_list, dim=0)
        part2_vectors_batch = torch.stack(part2_vectors_list, dim=0)
        part3_vectors_batch = torch.stack(part3_vectors_list, dim=0)

        combined_output = torch.cat([part1_vectors_batch, part2_vectors_batch, part3_vectors_batch], dim=-1)  # shape: (batch_size, 3 * hidden_size)
        pooled_output = self.dropout(self.tanh(self.fc1(combined_output)))
        pooled_output = self.dropout(self.tanh(self.fc1(pooled_output)))

        logits = self.fc_cls(pooled_output)  
        return logits

