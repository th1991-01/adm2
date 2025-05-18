import torch
import torch.nn as nn
from torch.nn import functional as F

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        activation=Swish(),
        layer_norm=True,
        with_residual=True,
        dropout=0.1
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.with_residual = with_residual
    
    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.dropout is not None:
            y = self.dropout(y)
        if self.with_residual:
            y = x + y
        if self.layer_norm is not None:
            y = self.layer_norm(y)
        return y

class SADModel(nn.Module):
    """ Self-transition Any-step Dynamics Model (SADM) """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=200,
        rnn_num_layers=3,
        dropout=0.1,
        device="cuda:0"
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rnn_num_layers = rnn_num_layers
        self.hidden_dim = hidden_dim
        self.device = device
        
        # obs to h
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, hidden_dim*rnn_num_layers)
        )
        # rnn with any-step action sequence as input
        self.rnn_layer = nn.GRU(
            input_size=action_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True
        )
        # h to obs
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim*rnn_num_layers, hidden_dim),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, self.obs_dim)
        )
        # h to delta obs
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim*rnn_num_layers, hidden_dim),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            ResBlock(hidden_dim, hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, (self.obs_dim+1)*2)
        )
        self.to(device)

    def forward(self, obs, act_seq):
        self.rnn_layer.flatten_parameters()
        h_state = self.encoder(obs)
        h_state = h_state.view(-1, self.hidden_dim, self.rnn_num_layers).permute(2, 0, 1)
        h_state = h_state.contiguous()
        rnn_out, h_state = self.rnn_layer(act_seq, h_state)
        # rnn_out = rnn_out[:, -1]
        next_in = h_state.permute(1, 2, 0).reshape(-1, self.hidden_dim*self.rnn_num_layers)
        output = self.out_layer(next_in)
        return output, h_state
    
    def encode_obs(self, obs):
        # obs: (bs, -1)
        return self.encoder(obs)
    
    def decode_h(self, h_state):
        # h_state: (bs, -1)
        return self.decoder(h_state)
    
    def init_hiddens(self, obs_seq, act_seq):
        # obs_seq: (bs, m, -1)
        # act_seq: (bs, m-1, -1)
        hiddens = []
        bs, m, _ = obs_seq.shape
        for i in range(m-1):
            _, hidden = self.forward(obs_seq[:, i], act_seq[:, i:])
            hidden = hidden.permute(1, 2, 0).reshape(bs, -1)
            hiddens.append(hidden)
        hiddens.append(self.encode_obs(obs_seq[:, -1]))
        # (m, bs, -1)
        hiddens = torch.stack(hiddens, dim=0)
        return hiddens
    
    def set_hiddens(self, hiddens, env_ids=None):
        if env_ids is None:
            self.hiddens = hiddens
            self.n_hiddens, self.n_parallels, _ = hiddens.shape
        else:
            self.hiddens[:, env_ids] = hiddens
            
    def update_hiddens(self, hiddens, env_ids):
        self.hiddens[:, env_ids] = torch.cat((self.hiddens[1:, env_ids], hiddens[None]), dim=0)
        
    def transition(self, action):
        # action: (bs, -1)
        action = torch.cat([action]*self.n_hiddens, dim=0)
        h_state = self.hiddens.view(-1, self.hidden_dim, self.rnn_num_layers).permute(2, 0, 1)
        h_state = h_state.contiguous()
        rnn_out, h_state = self.rnn_layer(action[:, None], h_state)
        self.hiddens = h_state.permute(1, 2, 0).reshape(self.n_hiddens, self.n_parallels, -1)
        # rnn_out = rnn_out[:, -1]
        next_in = h_state.permute(1, 2, 0).reshape(-1, self.hidden_dim*self.rnn_num_layers)
        output = self.out_layer(next_in).view(self.n_hiddens, self.n_parallels, -1)
        return output
