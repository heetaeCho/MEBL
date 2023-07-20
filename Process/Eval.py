import torch
from Models.Model import Model
from utils import loadModel
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def validation(valid_dataloader, load_path):
    model = Model().to(device)
    try:
        model = loadModel(load_path)
        print('===== Load model =====')
    except:
        print('===== Generating model =====')

    criterion = torch.nn.BCELoss()
    total_loss = 0
    
    for batch in valid_dataloader:
        nl, sc_nl, sc_pl, labels = batch
        nl = torch.squeeze(nl)

        results = model(nl, sc_nl, sc_pl)
        loss = criterion(results, labels.view(-1, 1).float().to(device))

        total_loss += loss
    return total_loss/len(valid_dataloader)