import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class SimpleLSTM(nn.Module):
    def __init__(self, n_feature, n_hidden, n_out, nb_layers, on_gpu):
        super(SimpleLSTM, self).__init__()
        self.n_feature = n_feature
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.nb_layers = nb_layers
        self.on_gpu = on_gpu
        self.LSTM = nn.LSTM(input_size=self.n_feature,
                           hidden_size=self.n_hidden,
                           num_layers=self.nb_layers,
                           batch_first=True,)
        self.out_layer = nn.Linear(self.n_hidden, self.n_out)
        
    def forward(self, seqs, lens):
        self.batch_size, self.seq_len, _ = seqs.size() # batch size
        self.h = self.init_hidden() # initialize hidden state of GRU
        
        unpad_seqs = pack_padded_sequence(seqs, lens, batch_first=True) # unpad
        
        LSTM_out, self.h= self.LSTM(unpad_seqs, self.h)
        
        LSTM_out, _ = pad_packed_sequence(LSTM_out, batch_first=True)
        batch_seq_len = LSTM_out.size()[1]
        LSTM_out = LSTM_out.contiguous()
        LSTM_out = LSTM_out.view(-1, self.n_hidden)
        
        out  = self.out_layer(LSTM_out)
        
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, N_OUT)
        out = out.view(self.batch_size, batch_seq_len, self.n_out)
        if self.seq_len != batch_seq_len:
            if self.on_gpu:
                out = torch.cat([out, torch.zeros([self.batch_size, self.seq_len - batch_seq_len, self.n_out]).cuda()], dim=1)
            else:
                out = torch.cat([out, torch.zeros([self.batch_size, self.seq_len - batch_seq_len, self.n_out])], dim=1)
        return out
    
    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_o = torch.randn(self.nb_layers, self.batch_size, self.n_hidden)
        hidden_c = torch.randn(self.nb_layers, self.batch_size, self.n_hidden)

        if self.on_gpu:
            hidden_o = hidden_o.cuda()
            hidden_c = hidden_c.cuda()

        hidden_o = Variable(hidden_o)
        hidden_c = Variable(hidden_c)

        return (hidden_o, hidden_c)
    
    def loss(self, Y, Y_hat, lens):
        # y: (batch_size, seq_len, n_out)
        # assume using l1 loss, padding is in the tail
        loss = 0
        loss_function = nn.functional.mse_loss
        for y_preds, y_trues,length in zip(Y_hat, Y, lens):
            y_true = y_trues[:length]
            y_pred = y_preds[:length]
            loss += loss_function(y_true, y_pred) # loss calculated for each utterance
        loss = loss/self.batch_size
        return loss