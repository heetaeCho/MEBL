import torch
from collections import OrderedDict

class Model(torch.nn.Module):
    def __init__(self, num_layers=8):
        super(Model, self).__init__()
        # self.code_encoder = SourceCodeEncoder()

        nl_encoders = OrderedDict()
        pl_encoders = OrderedDict()

        for i in range(num_layers):
            nl_encoders['cte_#{}'.format(i+1)] = CrossTrEncoder()
            pl_encoders['cte_#{}'.format(i+1)] = CrossTrEncoder()
        
        self.nl_encoders = torch.nn.Sequential(nl_encoders)
        self.pl_encoders = torch.nn.Sequential(pl_encoders)

        # self.nl_encoder = CrossTrEncoder()
        # self.pl_encoder = CrossTrEncoder()

        self.classifier = B_Classifier()
        
    def forward(self, inp):
        nl, sc_nl, sc_pl = inp
        # sc_nl = self.code_encoder(sc_nl)
        # sc_pl = self.code_encoder(sc_pl)

        nl_sc_nl = sc_nl
        pl_sc_nl = sc_nl
        
        nl_inp = (nl, nl_sc_nl, nl_sc_nl)
        pl_inp = (sc_pl, pl_sc_nl, pl_sc_nl)
        _, _, nl_sc_nl = self.nl_encoders(nl_inp)
        _, _, pl_sc_nl = self.pl_encoders(pl_inp)

        out = self.classifier(nl_sc_nl, pl_sc_nl)

        # nl_out = self.nl_encoder(nl, sc_nl, sc_nl)
        # pl_out = self.pl_encoder(sc_pl, sc_nl, sc_nl)
        # out = self.classifier(nl_out, pl_out)

        return out

class B_Classifier(torch.nn.Module):
    def __init__(self):
        super(B_Classifier, self).__init__()
        self.mhca = MHCA()
        self.linear = torch.nn.Linear(768, 2)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, nl, pl):
        att_out, att_w = self.mhca(pl, nl, nl)
        out = att_out[:, 0, :]
        out = self.linear(out)
        out = self.softmax(out)
        return out

class MHCA(torch.nn.Module):
    def __init__(self, dim=768, nhead=8):
        super(MHCA, self).__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, batch_first=True)

    def forward(self, q, k, v):
        attn_output, attn_output_weights = self.mha(q, k, v)
        return attn_output, attn_output_weights

class CrossTrEncoder(torch.nn.Module):
    def __init__(self, dim=768, nhead=8):
        super(CrossTrEncoder, self).__init__()
        self.attention = MHCA(dim, nhead) # output.shape = [batch, max_len, dim]

        self.norm1 = torch.nn.LayerNorm(dim, 1e-5)
        self.dropout1 = torch.nn.Dropout(0.1)
        
        self.linear1 = torch.nn.Linear(dim, 2048)
        self.dropout = torch.nn.Dropout(0.1)
        self.linear2 = torch.nn.Linear(2048, dim)

        self.norm2 = torch.nn.LayerNorm(dim, 1e-5)
        self.dropout2 = torch.nn.Dropout(0.1)
    
    def forward(self, inp):
        q, k, v = inp
        attention_out, _ = self.attention(q, k, v)
        out = self.norm1(v + self.dropout1(attention_out))
        out = self.linear2(self.dropout(self.linear1(out)))
        out = self.norm2(out + self.dropout2(out))
        return q, out, out

# class SourceCodeEncoder(torch.nn.Module):
#     def __init__(self, input_size=768, hidden_size=768, batch_first=True):
#         super(SourceCodeEncoder, self).__init__()
#         self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=batch_first)
#     def forward(self, x):
#         print(x.shape)
#         out, (hn, cn) = self.lstm(x)
#         return out

# if __name__ == '__main__':
# #     print(MHCA())
#     model = Model()
#     print(model)
#     a = torch.rand(16, 512, 768)
#     b = torch.rand(16, 512, 768)
#     c = torch.rand(16, 512, 768)
#     inp = (a, b, c)
#     print(model(inp))


    # encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
    # transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
    # src = torch.rand(10, 32, 512)
    # out = transformer_encoder(src)
    # print(transformer_encoder)
    # print(out.shape)
    # print('----------------------------')
    # a = torch.rand(16, 512, 768)
    # b = torch.rand(16, 512, 768)
    # c = torch.rand(16, 512, 768)
    # m = CrossTrEncoder()
    # print(m)
    # out = m(a, b, b)
    # out2 = m(c, b, b)
    # print('_------------------')
    # print(out.shape)
    # print("-=---=-=-=-=-=-=-=-=-=")   
    # print(out2.shape)
    # print("-=---=-=-=-=-=-=-=-=-=")
    # out = out[:, 0, :]   
    # out2 = out2[:, 0, :]
    # print('_------------------')
    # print(out.shape)
    # print("-=---=-=-=-=-=-=-=-=-=")   
    # print(out2.shape)
    # print('_------------------')
    # print(out[0])
    # print("-=---=-=-=-=-=-=-=-=-=")
    # print(out2[0])
    # print("-=---=-=-=-=-=-=-=-=-=")
    # cos = torch.nn.CosineSimilarity()
    # print(cos(out, out2))