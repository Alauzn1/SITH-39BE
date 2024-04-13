import copy
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, decoder_block, layer_de, norm):
        super(Decoder, self).__init__()
        self.n_layer = layer_de
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])
        self.norm = norm

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        de_layer_out = []
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
            de_layer_out.append(out)
   
        out = self.norm(out)


        return out, de_layer_out
