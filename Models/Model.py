from sklearn.naive_bayes import GaussianNB
import torch
from collections import OrderedDict

class Model(torch.nn.Module):
    def __init__(self, num_layers=4):
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

        print("\nnl_sc_nl: {}".format(nl_sc_nl))
        print("pl_sc_nl: {}\n".format(pl_sc_nl))

        # exit()
        out = self.classifier(nl_sc_nl, pl_sc_nl)        

        # nl_out = self.nl_encoder(nl, sc_nl, sc_nl)
        # pl_out = self.pl_encoder(sc_pl, sc_nl, sc_nl)
        # out = self.classifier(nl_out, pl_out)

        return out

class B_Classifier(torch.nn.Module):
    def __init__(self):
        super(B_Classifier, self).__init__()
        # self.mhca = MHCA()
        # self.linear1 = torch.nn.Linear(768, 2048)
        # self.linear0 = torch.nn.Linear(768, 1)
        self.lstm_nl = torch.nn.LSTM(input_size = 768, hidden_size = 768, batch_first=True)
        self.lstm_pl = torch.nn.LSTM(input_size = 768, hidden_size = 768, batch_first=True)

        self.linear1 = torch.nn.Linear(768*2, 2048)
        # self.norm1 = torch.nn.LayerNorm(2048, 1e-5)
        # self.act1 = torch.nn.GELU()

        self.linear2 = torch.nn.Linear(2048, 2048)
        # self.norm2 = torch.nn.LayerNorm(2048, 1e-5)
        # self.act2 = torch.nn.GELU()

        self.linear3 = torch.nn.Linear(2048, 2)
        # self.sig = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, nl, pl): # (1, N, 768), (1, N, 768)
        # print("\nnl: {}".format(nl))
        # print("nl.size(): {}".format(nl.size()))

        # print("\npl: {}".format(pl))
        # print("pl.size(): {}\n".format(pl.size()))

        nl_out, (_, _) = self.lstm_nl(nl)
        pl_out, (_, _) = self.lstm_pl(pl)

        nl_out = nl_out[:, -1, :] # (1, 768)
        pl_out = pl_out[:, -1, :] # (1, 768)

        # print("\nnl_out: {}".format(nl_out))
        # print("pl_out: {}\n".format(pl_out))

        # print("\nnl_out.size(): {}".format(nl_out.size()))
        # print("pl_out.size(): {}\n".format(pl_out.size()))

        out = torch.concatenate((nl_out, pl_out), dim=1) # (1, 768*2)
        # print('out: {}\n'.format(out))
        # print("out.size(): {}".format(out.shape))

        # m = torch.nn.AdaptiveAvgPool2d((1, 768))
        # m = torch.nn.AdaptiveMaxPool2d((1, 768))
        # out = m(out).view(1, -1)

        # exit()
        # exit()
        # out = self.linear0(out).view(1, -1)

        out = self.linear1(out)
        # out = self.norm1(out)
        # out = self.act1(out)

        out = self.linear2(out)
        # out = self.norm2(out)
        # out = self.act2(out)

        out = self.linear3(out)
        # out = self.sig(out)
        out = self.softmax(out)
        # print(out)
        return out

class MHCA(torch.nn.Module):
    def __init__(self, dim=768, nhead=8):
        super(MHCA, self).__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, batch_first=True)

    def forward(self, q, k, v):
        # print("q: {}\n".format(q))
        # print("k: {}\n".format(k))
        # print("v: {}\n".format(v))

        # attn_mask = torch.tensor([[False, False, False, True], \
        #                           [False, False, False, True], \
        #                           [False, False, False, True], \
        #                           [False, False, False, True]])
        # # k_attn_mask = torch.tensor([[False, False, False, True]])
        
        # print("attn_mask: {}\n".format(attn_mask))
        # print("attn_mask: {}\n".format(attn_mask.shape))
        attn_output, attn_output_weights = self.mha(q, k, v)
        # attn_output, attn_output_weights = self.mha(q, k, v, attn_mask=attn_mask)
        # attn_output, attn_output_weights = self.mha(q, k, v, attn_mask=attn_mask, key_padding_mask=k_attn_mask)

        m = torch.nn.AdaptiveAvgPool2d((k.size(1), k.size(2)))
        attn_output = m(attn_output)

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
        # print("q: {}\n".format(q))
        
        attention_out, _ = self.attention(q, k, v)
        # print("attention_out: {}\n".format(attention_out))

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
#     torch.manual_seed(0)
#     model = MHCA()
#     # cri = torch.nn.BCELoss()
#     # optim = torch.optim.Adam(model.parameters(), lr=1e-5)

#     # for name, child in model.named_children():
#     #     for param in child.parameters():
#     #         print(name, param.shape, param)

#     # optim.zero_grad()
#     a = torch.rand(1, 2, 768)
#     b = torch.rand(1, 3, 768)
#     c = torch.rand(1, 3, 768)
    
#     # a = torch.cat((a, torch.zeros(1, 2, 768)), dim=1)
#     # b = torch.cat((b, torch.zeros(1, 1, 768)), dim=1)
#     # a = torch.cat((a, torch.ones(1, 2, 768, dtype=torch.float)*torch.finfo(torch.float).min), dim=1)
#     # b = torch.cat((b, torch.ones(1, 1, 768, dtype=torch.float)*torch.finfo(torch.float).min), dim=1)
    
#     c = torch.cat((c, torch.zeros(1, 1, 768)), dim=1)
#     import copy
#     c = copy.deepcopy(b)
#     inp = (a, b, c)
#     out, wt = model(a, b, c)
#     # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
#     print(c.shape)
#     print(out.shape)

#     print(c + out)
# #     print(torch.matmul(c, out))
#################
    # # target output size of 5x7
    # m = torch.nn.AdaptiveAvgPool2d((5, 7))
    # input = torch.randn(1, 64, 8, 9)
    # print(input.shape)
    # output = m(input)
    # print(output.shape)
    # # target output size of 7x7 (square)
    # m = torch.nn.AdaptiveAvgPool2d(7)
    # input = torch.randn(1, 64, 10, 9)
    # output = m(input)
    # # target output size of 10x7
    # m = torch.nn.AdaptiveAvgPool2d((None, 7))

    # input = torch.randn(1, 64, 10, 9)
    # print(input.shape)
    # output = m(input)
    # print(output.shape)

    # print(m)

    # loss = cri(c, out)
    # print("\ncos = {}\n".format(cos(c, out)))
    # print("\nloss = {}\n".format(loss))
    # loss.backward()
    # optim.step()

    # for name, child in model.named_children():
    #     for param in child.parameters():
    #         print(name, param.shape, param)

    # model = B_Classifier()
    # cri = torch.nn.BCELoss()
    # optim = torch.optim.Adam(model.parameters(), lr=1e-5)

    # for name, child in model.named_children():
    #     for param in child.parameters():
    #         print(name, param.shape, param)

    # optim.zero_grad()
    # a = torch.rand(16, 512, 768)
    # b = torch.rand(16, 512, 768)

    # out = model(a, b)
    # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    # t = torch.randint(0, 2, out.shape, dtype=torch.float)

    # out = torch.squeeze(out)
    # t = torch.squeeze(t)
    # print(out)
    # print(t)

    # loss = cri(t, out)
    # print("\ncos = {}".format(cos(t, out)))
    # print("loss = {}\n".format(loss))
    # loss.backward()
    # optim.step()

    # for name, child in model.named_children():
    #     for param in child.parameters():
    #         print(name, param.shape, param)

    # model = Model()
    # cri = torch.nn.BCELoss()
    # optim = torch.optim.Adam(model.parameters(), lr=1e-5)

    # for name, child in model.named_children():
    #     for param in child.parameters():
    #         print(name, param.shape, param)

    # optim.zero_grad()
    # a = torch.rand(16, 512, 768)
    # b = torch.rand(16, 512, 768)
    # c = torch.rand(16, 512, 768)
    # inp = (a, b, c)
    # out = model(inp)
    # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    # # print(out.shape)
    # t = torch.randint(0, 2, out.shape, dtype=torch.float)
    # # print(t.shape)

    # loss = cri(t, out)
    # print("\ncos = {}\n".format(cos(t, out)))
    # print("\nloss = {}\n".format(loss))
    # loss.backward()
    # optim.step()

    # for name, child in model.named_children():
    #     for param in child.parameters():
    #         print(name, param.shape, param)