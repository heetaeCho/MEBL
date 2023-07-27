import torch

from DataProcessor.ShapeProcessor import makeLabel, squeeze
from Models.Model import Model
from utils import loadModel, saveModel

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
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.criterion = torch.nn.BCELoss()
        lambda_fnc = lambda epoch: 0.95 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lambda_fnc, verbose=False)

    def train(self, train_dataloader, load_path, save_path, epoch):
        print("Current learning rate = ", self.scheduler.get_last_lr())
        try:
            self.model = loadModel(load_path)
            print('===== Load model =====')
        except:
            print('===== Generating model =====')

        total_loss = 0

        for batch in train_dataloader:
            nl, sc_nl, sc_pl, labels = batch
            nl, sc_nl, sc_pl = squeeze(nl, sc_nl, sc_pl)

            self.optim.zero_grad()
            
            results = self.model(nl, sc_nl, sc_pl)
            labels = makeLabel(labels)
            loss = self.criterion(results, labels.to(device))
            loss.backward()

            self.optim.step()

            total_loss += loss.detach().cpu().numpy()

        self.scheduler.step()
        
        g_loss = total_loss / len(train_dataloader)
        saveModel(self.model, save_path)
        return g_loss