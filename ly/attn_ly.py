import math
import torch.nn.functional as F
import torch.nn as nn
import torch


class TanhAttention(nn.Module):
    """
    https://github.com/Renovamen/Text-Classification/blob/master/models/AttBiLSTM/attention.py
    """

    def __init__(self, rnn_size):
        super(TanhAttention, self).__init__()
        self.w = nn.Linear(rnn_size, 1)

    def forward(self, H):
        """
        :param H: output of LSTM (batch_size, word_pad_len, hidden_size)
        :return:
            r: attn_output (batch_size, word_pad_len, hidden_size)
            alpha: attn_weights (batch_size, word_pad_len)
        """
        seq_len = H.size(1)
        M = H.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # M: (batch_size, word_pad_len, word_pad_len, hidden_size)
        # H: (batch_size, word_pad_len, hidden_size)
        
        # M = M + M.transpose(1, 2) 
        MT = M.transpose(1, 2)
        temp = torch.zeros(M.shape)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                temp[i][j] = torch.add(M[i][j], MT[i][j])
        M = temp.to(torch.device('cuda:0'))

        M = torch.tanh(M)
        # M: (batch_size, word_pad_len, word_pad_len, hidden_size)
        alpha = self.w(M).squeeze(3)  # (batch_size, word_pad_len, word_pad_len)
        alpha = F.softmax(alpha, -1)  # (batch_size, word_pad_len, word_pad_len)
        r = torch.matmul(alpha, H)  # (batch_size, word_pad_len, rnn_size)
        return r, alpha

        # M = torch.tanh(H)  # eq.9:  M, H (batch_size, word_pad_len, rnn_size)
        # # eq.10: α = softmax(w^T M)
        # alpha = self.w(M).squeeze(2)  # (batch_size, word_pad_len)
        # alpha = F.softmax(alpha, -1)  # (batch_size, word_pad_len)
        # r = H * alpha.unsqueeze(2)  # (batch_size, word_pad_len, rnn_size)
        # # r = r.sum(dim=1)  # (batch_size, rnn_size)
        # return r, alpha



class DotAttention(nn.Module):
    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, value):
        return self.dot_attention(value, value, value)

    def dot_attention(self, query, key, value, mask=None, dropout=None):
        """
          Implementation of Scaled dot product attention
        :param query: batch_size, pad_text_length, embedding_size
        :param key:
        :param value:
        :param mask:
        :param dropout:
        :return:
        """
        # query:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


if __name__ == '__main__':
    a = torch.randn((1, 5, 3))
    model = TanhAttention(3)
    print(a)
    x, y = model(a)
    print('========x')
    print(x.shape)
    print('========y')
    # print(y)

    model2 = DotAttention()
    x, y = model2(a)

    print("==========x")
    print(x.shape)
    print("==========y")
    # print(y)