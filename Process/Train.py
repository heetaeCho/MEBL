import torch

from DataProcessor.ShapeProcessor import makeLabel, squeeze
from Models.Model import Model
from utils import loadModel, saveModel

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
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = torch.nn.BCELoss()
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

    def train(self, train_dataloader, load_path, save_path, epoch):
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
            # label = makeLabel(label)

            # print(nl.shape) # [1, min(#tokens, max_len), 768]
            # print(sc_nl.shape) # [1, #chunks, 768]
            # print(sc_pl.shape) # [1, #chunks, 768]

            self.optim.zero_grad()
            inp = (nl.to(device), sc_nl.to(device), sc_pl.to(device))
            result = self.model(inp)

            result = result.view(-1) #[2]
            result = result[1].view(1) # softmax
            # result = result[0].view(1) # sigmoid

            # print("\nlabel: {}".format(label))
            # print("result: {}\n".format(result))
            
            loss = self.criterion(result, label.to(device))
            # print("\nLoss: {}\n".format(loss))
            loss.backward()

            self.optim.step()

            total_loss += loss.detach().cpu().numpy()

        # self.scheduler.step()
        
        g_loss = total_loss / len(train_dataloader)
        saveModel(self.model, save_path)
        return g_loss