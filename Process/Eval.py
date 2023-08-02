import torch
import numpy as np
from Models.Model import Model
from utils import loadModel
from DataProcessor.ShapeProcessor import getDataLoader, squeeze, makeLabel

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def validation(valid_dataloader, load_path):
    model = Model().eval().to(device)
    try:
        model = loadModel(load_path)
        print('===== Load model =====')
    except:
        print('===== Generating model =====')

    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    
    for batch in valid_dataloader:
        nl, sc_nl, sc_pl, labels = batch
        nl, sc_nl, sc_pl = squeeze(nl, sc_nl, sc_pl)

        inp = (nl.to(device), sc_nl.to(device), sc_pl.to(device))
        results = model(inp)

        labels = makeLabel(labels, bin=True)
        # labels = makeLabel(labels)
        
        loss = criterion(results, labels.to(device))

        total_loss += loss.detach().cpu().numpy()
    return total_loss/len(valid_dataloader)

def prediction(bug_reports, load_path):
    model = Model().eval().to(device)
    try:
        model = loadModel(load_path)
        print('===== Load model =====')
    except:
        print('===== Generating model =====')

    # print("==========================================================")
    # print(model)
    # print("==========================================================")
    # print("#params : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print("==========================================================")

    test_dataloader = getDataLoader(bug_reports, test=True)

    full_res = []
    full_labels = []
    for batch in test_dataloader:
        nl, sc_nl, sc_pl, labels = batch
        nl, sc_nl, sc_pl = squeeze(nl, sc_nl, sc_pl)

        inp = (nl.to(device), sc_nl.to(device), sc_pl.to(device))
        results = model(inp)
        full_res.extend(results.detach().cpu().numpy())
        full_labels.extend(labels.detach().cpu().numpy())
    return full_res, full_labels

def evaluation(bug_report, results, labels):
    print(bug_report.getBugId())
    
    results = [ (res[1], lab) for res, lab in zip(results, labels) ]
    results = sorted(results, key=lambda x:x[0], reverse=True)

    print("Total Candidates: ", len(labels))
    print(results)
    topk(results)

def topk(results):
    top_n = []
    for i, (probability, label) in enumerate(results):
        if label == 1:
            top_n.append(i+1)
    print(top_n)