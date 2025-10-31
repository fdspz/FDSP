import torch.nn as nn
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.basics import timestep_embedding
from models.GRLF import ReverseLayerF


SEG_DICT = {
    '1': [14],
    '2': [8, 6],
    '4': [4, 4, 3, 3],
}

def grad_reverse(x, alpha=1.0):
    return ReverseLayerF.apply(x, alpha)

class rnns(nn.Module):
    def __init__(self, input_dim, hidden_dim, model_name, dropout, num_seg, learning_type):
        super(rnns, self).__init__()

        self.num_seg = num_seg
        self.seg_list = SEG_DICT[str(num_seg)]
        self.seg_cumsum = [2] + (np.cumsum(self.seg_list) + 2).tolist()

        self.initialize = 'rand'
        
        self.input_dim = input_dim - 2
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self.dropout_rate = dropout
        self.learning_type = learning_type

        if self.learning_type == 'diff':
            mixer_input_dim = self.hidden_dim * 3
        elif self.learning_type == 'grad':
            mixer_input_dim = self.hidden_dim

        if 'lstm' in model_name:
            mixer = nn.LSTMCell(mixer_input_dim, self.hidden_dim)
        elif 'gru' in model_name:
            mixer = nn.GRUCell(mixer_input_dim, self.hidden_dim)
        else:
            raise ModuleNotFoundError
        
        self.mlp_is = nn.ModuleList()
        for i, seq_len in enumerate(self.seg_list):
            self.mlp_is.append(nn.Linear(seq_len, self.hidden_dim))
        
        self.mixers = nn.ModuleList()
        for i, seq_len in enumerate(self.seg_list):
            self.mixers.append(mixer)
        
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
        initializer = torch.rand if self.initialize == 'rand' else torch.zeros

        if 'lstm' in self.model_name:
            h0 = initializer(batch_size, self.hidden_dim)
            c0 = initializer(batch_size, self.hidden_dim)
            hidden = (h0, c0)
        elif 'gru' in self.model_name:
            h0 = initializer(batch_size, self.hidden_dim)
            hidden = h0
        else:
            raise ModuleNotFoundError
    
        def cudify_hidden(h):
            if isinstance(h, tuple):
                return tuple([cudify_hidden(h_) for h_ in h])
            else:
                return h.to(device)
            
        hidden = cudify_hidden(hidden)

        return hidden
    
    def detach_hidden(self, hidden):
        if isinstance(hidden, tuple):
            return tuple([self.detach_hidden(h_) for h_ in hidden])
        else:
            return hidden.detach()

    def process(self, x_list, device):
        """
        x: (batch, seq_len, input_dim)
        """
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
                emb = torch.concat([emb, t_emb], dim=-1).unsqueeze(1)
            embs.append(emb)
        embs = torch.concat(embs, dim=1).reshape(batch, len(self.seg_list), seq_len, -1)

        return embs, sorted_lens, sort_idx

    def encode(self, x, device):
        """
        x: (batch, seq_len, input_dim)
        """

        self.batch_size = len(x)
        emb, sorted_lens, sort_idx = self.process(x, device)   # (batch, len(self.seg_list), seq_len, input_dim)
        seq_len = int(emb.size(2))
        
        hidden_states = []
        for i in range(len(self.seg_list)):
            hidden = self.init_hidden(self.batch_size, device)
            sub_hidden_states = []
            for j in range(seq_len):
                hidden = self.mixers[i](emb[:, i, j], hidden)
                if self.model_name == 'gru':
                    sub_hidden_states.append(hidden)
                else:
                    sub_hidden_states.append(hidden[0])
            
            sub_hidden_states = torch.stack(sub_hidden_states, dim=1)
            sub_hidden_states = self.hnn_dropout(sub_hidden_states)

            hidden_states.append(sub_hidden_states)

        hidden_states = torch.stack(hidden_states, dim=-2)

        # use sorted_lens to get the effective last step of each sequence
        batch_idx = torch.arange(self.batch_size, device=device)
        final_indices = sorted_lens - 1

        hidden_state = hidden_states[batch_idx, final_indices]  # [batch * particles, hidden_dim] or [batch, hidden_dim]

        _, inverse_idx = torch.sort(sort_idx)

        hidden_state = hidden_state.reshape(hidden_state.shape[0], -1)

        inverse_idx = inverse_idx
        hidden_state = hidden_state[inverse_idx]

        return hidden_state
            
    def predict(self, hidden_state):
        """
        hidden_state: (batch, hidden_dim)
        """
        out = self.mlp_o(hidden_state)
        return out

    def domain_classification(self, hidden_state):
        """
        hidden_state: (batch, hidden_dim)
        """
        alpha = 1.0
        reversed_hidden_state = grad_reverse(hidden_state, alpha)
        labels = self.domain_classifier(reversed_hidden_state)
        return labels
    
    def forward(self, x, device):
        """
        x: (batch, seq_len, input_dim)
        """

        hidden_state = self.encode(x, device)

        out = self.predict(hidden_state)
        
        return out