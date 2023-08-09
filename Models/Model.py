from sklearn.naive_bayes import CategoricalNB
import torch
from collections import OrderedDict

import pickle

class Model(torch.nn.Module):
    def __init__(self, num_layers=4):
        super(Model, self).__init__()
        nl_encoders = OrderedDict()
        pl_encoders = OrderedDict()

        for i in range(num_layers):
            nl_encoders['cte_#{}'.format(i+1)] = CrossTrEncoder()
            pl_encoders['cte_#{}'.format(i+1)] = CrossTrEncoder()
        
        self.nl_encoders = torch.nn.Sequential(nl_encoders)
        self.pl_encoders = torch.nn.Sequential(pl_encoders)

        self.lstm_nl = torch.nn.LSTM(input_size = 768, hidden_size = 768, batch_first=True)
        self.lstm_pl = torch.nn.LSTM(input_size = 768, hidden_size = 768, batch_first=True)
        
    def forward(self, inp):
        nl, sc_nl, sc_pl = inp

        nl_sc_nl = sc_nl
        pl_sc_nl = sc_nl

        nl_inp = (nl, nl_sc_nl, nl_sc_nl)
        pl_inp = (sc_pl, pl_sc_nl, pl_sc_nl)
        _, _, nl_sc_nl = self.nl_encoders(nl_inp)
        _, _, pl_sc_nl = self.pl_encoders(pl_inp)

        nl_out, (_, _) = self.lstm_nl(nl_sc_nl)
        pl_out, (_, _) = self.lstm_pl(pl_sc_nl)

        nl_out = nl_out[:, -1, :] # (1, 768)
        pl_out = pl_out[:, -1, :] # (1, 768)
        
        return nl_out, pl_out

class NV_Classifier():
    def __init__(self):
        self.classifier = CategoricalNB()

    def fit(self, x, y):
        self.classifier.fit(x, y)

    def pred(self, x):
        return self.classifier.predict(x)

    def save(self):
        with open('./nv_classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
    
    def load(self):
        with open('./nv_classifier.pkl', 'rb') as f:
            self.classifier = pickle.load(f)


class B_Classifier(torch.nn.Module):
    def __init__(self):
        super(B_Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(768*2, 2048)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.norm1 = torch.nn.LayerNorm(2048, 1e-5)

        self.linear2 = torch.nn.Linear(2048, 2048)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.norm2 = torch.nn.LayerNorm(2048, 1e-5)

        self.linear3 = torch.nn.Linear(2048, 2)
        # self.sig = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, nl, pl): # (1, 768), (1, 768)
        out = torch.concatenate((nl, pl), dim=1) # (1, 768*2)

        out = self.linear1(out)
        out = self.dropout1(out)
        out = self.norm1(out)

        out = self.linear2(out)
        out = self.dropout2(out)
        out = self.norm2(out)

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
        attn_output, attn_output_weights = self.mha(q, k, v)

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
        
        attention_out, _ = self.attention(q, k, v)

        out = self.norm1(v + self.dropout1(attention_out))
        out = self.linear2(self.dropout(self.linear1(out)))
        out = self.norm2(out + self.dropout2(out))

        return q, out, out

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