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
        # self.criterion = torch.nn.BCELoss()
        self.criterion = torch.nn.CrossEntropyLoss()
        lambda_fnc = lambda epoch: 0.95 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lambda_fnc, verbose=False)

    def train(self, train_dataloader, load_path, save_path, epoch):
        print("Current learning rate = ", self.scheduler.get_last_lr())
        try:
            self.model = loadModel(load_path)
            print('===== Load model =====')
        except:
            print('===== Generate model =====')

        total_loss = 0

        for batch in tqdm(train_dataloader, desc="{}-th epoch now training".format(epoch), ncols=70, leave=True):

            # for param in self.model.parameters():
            #     print(param)

            nl, sc_nl, sc_pl, labels = batch
            # print("nl: {}\n".format(nl))
            # print("sc_nl: {}\n".format(sc_nl))
            # print("sc_pl: {}\n".format(sc_pl))
            # print("labels: {}\n".format(labels))
            # exit()

            nl, sc_nl, sc_pl = squeeze(nl, sc_nl, sc_pl)
            # nl.view(-1, 512, 768)
            # sc_nl.view(-1, 512, 768)
            # sc_pl.view(-1, 512, 768)

            self.optim.zero_grad()
            
            inp = (nl.to(device), sc_nl.to(device), sc_pl.to(device))

            results = self.model(inp)

            labels = makeLabel(labels, bin=True) # return [#sample]
            # labels = makeLabel(labels) # return [#sample, #class]

            '''
            # CrossEntropy는 class indices 또는 class probabilities 를 사용 가능.
            #
            # class indices 사용 시, = [#sample] each value = class number
            # model.output = [#sample, #class]
            #
            # class probabilities 사용 시, = [#sample, #class] each #class value = class proba
            # model.output = [#sample, #class]
            '''

            # results = torch.squeeze(results)
            # labels = torch.squeeze(labels)

            print("\nlabels: {}".format(labels))
            print("results: {}\n".format(results))
            if len(results.shape) == 1:
                results = results.view(1, -1)
            loss = self.criterion(results, labels.to(device))
            # print("\nLoss: {}\n".format(loss))
            loss.backward()

            self.optim.step()

            total_loss += loss.detach().cpu().numpy()

        self.scheduler.step()
        
        g_loss = total_loss / len(train_dataloader)
        saveModel(self.model, save_path)
        return g_loss