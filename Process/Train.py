import torch

from Models.Model import Model, B_Classifier
from utils import saveModel

from tqdm import tqdm

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class Train:
    def __init__(self):
        self.model = Model().train().to(device)
        self.classifier = B_Classifier().train().to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-5)

        self.criterion_1 = torch.nn.CosineEmbeddingLoss(margin=-1)
        self.criterion_2 = torch.nn.BCELoss()
        '''
            # CrossEntropy는 class indices 또는 class probabilities 를 사용 가능.
            #
            # class indices 사용 시, = [#sample] each value = class number
            # model.output = [#sample, #class]
            #
            # class probabilities 사용 시, = [#sample, #class] each #class value = class proba
            # model.output = [#sample, #class]
        '''
        # self.criterion = torch.nn.CrossEntropyLoss()
        # lambda_fnc = lambda epoch: 0.95 ** epoch
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lambda_fnc, verbose=False)


    def train(self, train_dataloader, m_save_path, c_save_path, epoch):
        # print("Current learning rate = ", self.scheduler.get_last_lr())
        # try:
        #     self.model = loadModel(load_path)
        #     print('===== Load model =====')
        # except:
        #     print('===== Generate model =====')

        total_loss = 0

        for batch in tqdm(train_dataloader, desc="{}-th epoch now training".format(epoch), ncols=70, leave=True):
            nl, sc_nl, sc_pl, label = batch

            nl = torch.tensor(nl, dtype=torch.float)
            sc_nl = torch.tensor(sc_nl, dtype=torch.float)
            sc_pl = torch.tensor(sc_pl, dtype=torch.float)
            # label = torch.LongTensor([label]) # [1]
            label = torch.FloatTensor([label]) # [1]
            sim_label = torch.FloatTensor([label]) if label == 1 else torch.FloatTensor([-1])
            # label = makeLabel(label)

            # print(nl.shape) # [1, min(#tokens, max_len), 768]
            # print(sc_nl.shape) # [1, #chunks, 768]
            # print(sc_pl.shape) # [1, #chunks, 768]

            self.optim.zero_grad()
            inp = (nl.to(device), sc_nl.to(device), sc_pl.to(device))
            nl_out, pl_out = self.model(inp)

            loss1 = self.criterion_1(nl_out, pl_out, sim_label.to(device))

            result = self.classifier(nl_out, pl_out)

            result = result.view(-1) #[2]
            result = result[1].view(1) # softmax
        #     # result = result[0].view(1) # sigmoid

            loss2 = self.criterion_2(result, label.to(device))

            loss = loss1 + loss2
            loss.backward()

            # print("\nlabel: {}".format(label))
            # print("result: {}".format(result))
            # print("sim_label: {}".format(sim_label))
            # print("\nLoss1: {}".format(loss1))
            # print("Loss2: {}\n".format(loss2))

            self.optim.step()

            total_loss += loss.item()

        # self.scheduler.step()
        
        g_loss = total_loss / len(train_dataloader)
        saveModel(self.model, m_save_path)
        saveModel(self.classifier, c_save_path)
        return g_loss