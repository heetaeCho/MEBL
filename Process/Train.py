from Models.Model import Model
from DataProcessor.ShapeProcessor import squeeze, makeLabel
from utils import loadModel, saveModel
import torch

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def train(train_dataloader, load_path, save_path):
    model = Model().to(device)
    try:
        model = loadModel(load_path)
        print('===== Load model =====')
    except:
        print('===== Generating model =====')
    
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCELoss()
    total_loss = 0

    for batch in train_dataloader:
        nl, sc_nl, sc_pl, labels = batch
        nl, sc_nl, sc_pl = squeeze(nl, sc_nl, sc_pl)

        results = model(nl, sc_nl, sc_pl)
        labels = makeLabel(labels)
        
        loss = criterion(results, labels.to(device))

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.detach().cpu().numpy()

    g_loss = total_loss / len(train_dataloader)
    saveModel(model, save_path)
    
    return g_loss