import torch
from Models.Model import Model
from utils import loadModel
from DataProcessor.ShapeProcessor import getDataLoader, squeeze, makeLabel

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
        nl, sc_nl, sc_pl = squeeze(nl, sc_nl, sc_pl)

        results = model(nl, sc_nl, sc_pl)
        labels = makeLabel(labels)
        loss = criterion(results, labels.to(device))

        total_loss += loss.detach().cpu().numpy()
    return total_loss/len(valid_dataloader)

def prediction(bug_reports, load_path):
    model = Model().to(device)
    try:
        model = loadModel(load_path)
        print('===== Load model =====')
    except:
        print('===== Generating model =====')

    test_dataloader = getDataLoader(bug_reports, test=True)

    for batch in test_dataloader:
        nl, sc_nl, sc_pl, labels = batch
        nl, sc_nl, sc_pl = squeeze(nl, sc_nl, sc_pl)

        results = model(nl, sc_nl, sc_pl)
        print(results)
        print(labels)