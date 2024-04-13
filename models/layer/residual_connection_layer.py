import torch.nn as nn


class ResidualConnectionLayer(nn.Module):

    def __init__(self, norm, dr_rate=0):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(p=dr_rate)


    # def forward(self, x, sub_layer):
    #     out = x
    #     out = self.norm(out)
    #     out = sub_layer(out)
    #     out = self.dropout(out)
    #     out = out + x
    #     return out

    def forward(self, x, sub_layer):
        out1 = x
        sub_layer_out = self.dropout(sub_layer(x))
        # print(out1.shape)
        # print(self.norm(out1).shape)
        # print(sub_layer_out.shape)
        out = self.norm(out1 + sub_layer_out)
        return out

