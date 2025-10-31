import torch.nn as nn
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import math
from models.basics import timestep_embedding

SEG_DICT = {
    '1': [14],
    '2': [8, 6],
    '4': [4, 4, 3, 3],
}

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerSegmentEncoder(nn.Module):    
    def __init__(self, input_dim, hidden_dim, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.1):
        super(TransformerSegmentEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.output_map = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x, mask=None):

        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        output = self.output_map(encoded)
        
        return output

class TransformerNetwork(nn.Module):    
    def __init__(self, input_dim, hidden_dim, model_name="transformer", 
                 dropout=0.1, num_seg=4, learning_type='grad'):
        super(TransformerNetwork, self).__init__()
        
        self.num_seg = num_seg
        self.seg_list = SEG_DICT[str(num_seg)]
        self.seg_cumsum = [2] + (np.cumsum(self.seg_list) + 2).tolist()
                
        self.input_dim = input_dim - 2
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self.dropout_rate = dropout
        
        self.learning_type = learning_type

        if self.learning_type == 'diff':
            mixer_input_dim = self.hidden_dim * 3
        elif self.learning_type == 'grad':
            mixer_input_dim = self.hidden_dim
        
        self.mlp_is = nn.ModuleList()
        for i, seq_len in enumerate(self.seg_list):
            self.mlp_is.append(nn.Linear(seq_len, self.hidden_dim))

        mixer = TransformerSegmentEncoder(
                    input_dim=mixer_input_dim,
                    hidden_dim=self.hidden_dim,
                    nhead=4,
                    num_layers=1,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=dropout
                )
        
        self.mixers = nn.ModuleList()
        for _ in range(len(self.seg_list)):
            self.mixers.append(mixer)
        
        self.pos_encoder = PositionalEncoding(mixer_input_dim)
        
        self.mlp_o = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_seg, self.input_dim),
            nn.Sigmoid()
        )
        
        self.hnn_dropout = nn.Dropout(self.dropout_rate)
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_seg, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.hidden_dim, 2),
        )
    
    def init_hidden(self, batch_size, device):
        return None
    
    def detach_hidden(self, hidden):
        return None
    
    def process(self, x_list, device):
        lengths = torch.tensor([x.size(0) for x in x_list])
        sorted_lens, sort_idx = torch.sort(lengths, descending=True)
        sorted_x = [x_list[i] for i in sort_idx]
        
        padded_x = pad_sequence(sorted_x, batch_first=True).to(device)
        
        batch, seq_len, n_feas = padded_x.size()
        padded_x = padded_x.reshape(-1, n_feas)
        
        # ------------------------------------------------------------------
        tc_emb = timestep_embedding(padded_x[:, 0], self.hidden_dim)
        if self.learning_type == 'diff':
            tp_emb = timestep_embedding(padded_x[:, 1], self.hidden_dim)
            t_emb = torch.concat([tc_emb, tp_emb], dim=-1)
        elif self.learning_type == 'grad':
            t_emb = tc_emb
         # ------------------------------------------------------------------
        
        embs = []
        for i in range(len(self.seg_list)):
            emb = padded_x[:, self.seg_cumsum[i] : self.seg_cumsum[i + 1]]
            emb = self.mlp_is[i](emb)
            if self.learning_type == 'diff':
                emb = torch.concat([t_emb, emb], dim=-1).unsqueeze(1)
            embs.append(emb)
        
        embs = torch.concat(embs, dim=1).reshape(batch, len(self.seg_list), seq_len, -1)
        
        padding_masks = []
        for i in range(batch):
            mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            if sorted_lens[i] < seq_len:
                mask[sorted_lens[i]:] = True
            padding_masks.append(mask)
        padding_mask = torch.stack(padding_masks)
        
        return embs, sorted_lens, sort_idx, padding_mask
    
    def fusion_module(self, hidden_state):
        out = hidden_state.reshape(hidden_state.shape[0], -1)
        
        return out
    
    def encode(self, x, device):
        self.batch_size = len(x)
        embs, sorted_lens, sort_idx, padding_mask = self.process(x, device)
        
        hidden_states = []
        
        for i in range(len(self.seg_list)):
            segment_emb = embs[:, i]  # [batch, seq_len, input_dim]
            segment_emb = self.pos_encoder(segment_emb)
            
            
            encoded = self.mixers[i](segment_emb, padding_mask)
            
            encoded = self.hnn_dropout(encoded)
            
            hidden_states.append(encoded)
        
        batch_idx = torch.arange(self.batch_size, device=device)
        final_hidden_states = []
        
        for i in range(len(self.seg_list)):
            final_indices = sorted_lens - 1
            final_hidden = hidden_states[i][batch_idx, final_indices]
            final_hidden_states.append(final_hidden)
        
        combined_hidden = torch.stack(final_hidden_states, dim=1)  # [batch, num_segments, hidden_dim]
        
        _, inverse_idx = torch.sort(sort_idx)
        
        hidden_state = combined_hidden.reshape(combined_hidden.shape[0], -1)
        
        hidden_state = hidden_state[inverse_idx]
        
        return hidden_state
    
    def predict(self, hidden_state):
        out = self.mlp_o(hidden_state)
        return out
    
    def domain_classification(self, hidden_state):
        labels = self.domain_classifier(hidden_state)
        return labels
    
    def forward(self, x, device):
        hidden_state = self.encode(x, device)
        out = self.predict(hidden_state)
        return out