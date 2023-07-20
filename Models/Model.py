import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.nl_encoder = CrossTrEncoder()
        self.pl_encoder = CrossTrEncoder()
        self.classifier = B_Classifier()
        
    def forward(self, nl, sc_nl, sc_pl):
        nl_out = self.nl_encoder(nl, sc_nl, sc_nl)
        pl_out = self.pl_encoder(sc_pl, sc_nl, sc_nl)

        out = self.classifier(nl_out, pl_out)
        return out

class B_Classifier(torch.nn.Module):
    def __init__(self):
        super(B_Classifier, self).__init__()
        self.mhca = MHCA()
        self.linear = torch.nn.Linear(768, 1)
        self.sig = torch.nn.Sigmoid()
    
    def forward(self, nl, pl):
        att_out, att_w = self.mhca(pl, nl, nl)
        out = att_out[:, 0, :]
        out = self.linear(out)
        out = self.sig(out)
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
    
    def forward(self, q, k, v):
        attention_out, _ = self.attention(q, k, v)
        out = self.norm1(v + self.dropout1(attention_out))
        out = self.linear2(self.dropout(self.linear1(out)))
        out = self.norm2(out + self.dropout2(out))
        return out


# if __name__ == '__main__':
#     print(MHCA())
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