"""
Position feed-forward network from "Attention is All You Need"
"""
import torch
import torch.nn as nn
from c2nl.modules.util_class import LayerNorm


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        '''self.intermediate = nn.Linear(d_model, 6*d_ff)
        self.output = nn.Linear(d_ff, 6*d_model)
        self.layer_norm = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)'''

        self.d_model = d_model
        self.d_ff = d_ff

        self.layer_norm1 = LayerNorm(d_model)
        self.intermediate1 = nn.Linear(d_model, d_ff)
        self.output1 = nn.Linear(d_ff, d_model)
        self.relu1 = nn.ReLU()
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)

    def forward(self, x, code_type5=None):
        """
        Layer definition.
        Args:
            input: [ batch_size, input_len, model_dim ]
        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        # 先生成一个B x len x d大小的矩阵从0到d-1 arange生成并用expand扩展，然后code_type5 expand 成B x len x d再乘d，加到前面的矩阵上面去
        #if code_type5 is not None:
        '''if False:
            # B x len x d_ff
            dff = torch.arange(0, self.d_ff).cuda(non_blocking=True)
            code_type_onehot_dff = dff.expand(code_type5.shape[0], code_type5.shape[1], self.d_ff)
            code_type5_dff = code_type5.unsqueeze(2).expand(code_type5.shape[0], code_type5.shape[1], self.d_ff)
            code_type_onehot_dff = code_type_onehot_dff + self.d_ff*code_type5_dff
            code_type_onehot_dff = code_type_onehot_dff.long()
            # B x len x d_model
            dmodel = torch.arange(0, self.d_model).cuda(non_blocking=True)
            code_type_onehot_dmodel = dmodel.expand(code_type5.shape[0], code_type5.shape[1], self.d_model)
            code_type5_dmodel = code_type5.unsqueeze(2).expand(code_type5.shape[0], code_type5.shape[1], self.d_model)
            code_type_onehot_dmodel = code_type_onehot_dmodel + self.d_model*code_type5_dmodel
            code_type_onehot_dmodel = code_type_onehot_dmodel.long()

            inter = self.dropout_1(self.relu(self.intermediate(self.layer_norm(x)))) # B x len x 6*d_ff
            inter_type = inter.gather(dim=2, index=code_type_onehot_dff) # B x len x d_ff
            output = self.dropout_2(self.output(inter_type)) # B x len x 6*d_model
            output_type = output.gather(dim=2, index=code_type_onehot_dmodel) # B x len x d_model
            return output_type + x
        else:'''
        inter = self.dropout_3(self.relu1(self.intermediate1(self.layer_norm1(x))))
        output = self.dropout_4(self.output1(inter))
        return output + x