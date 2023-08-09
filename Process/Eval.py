import torch
import numpy as np
from Models.Model import Model, B_Classifier
from utils import loadModel
from DataProcessor.ShapeProcessor import getDataLoader

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def validation(valid_dataloader, m_load_path, c_load_path):
    with torch.no_grad():
        model = Model().eval().to(device)
        classifier = B_Classifier().eval().to(device)
        
        model = loadModel(m_load_path)
        classifier = loadModel(c_load_path)

        criterion_1 = torch.nn.CosineEmbeddingLoss()
        criterion_2 = torch.nn.BCELoss()
        # criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0

        for batch in valid_dataloader:
            nl, sc_nl, sc_pl, label = batch

            nl = torch.tensor(nl, dtype=torch.float)
            sc_nl = torch.tensor(sc_nl, dtype=torch.float)
            sc_pl = torch.tensor(sc_pl, dtype=torch.float)
            # label = torch.LongTensor([label])
            label = torch.FloatTensor([label]) # [1]
            sim_label = torch.FloatTensor([label]) if label == 1 else torch.FloatTensor([-1])
            # label = makeLabel(label)

            inp = (nl.to(device), sc_nl.to(device), sc_pl.to(device))
            nl_out, pl_out = model(inp)

            loss1 = criterion_1(nl_out, pl_out, sim_label.to(device))

            result = classifier(nl_out, pl_out)
            result = result.view(-1) #[2]
            result = result[1].view(1) # softmax

            loss2 = criterion_2(result, label.to(device))

            loss = loss1 + loss2

            total_loss += loss.item()
        return total_loss/len(valid_dataloader)

def prediction(bug_reports, m_load_path, c_load_path, project):
    with torch.no_grad():
        model = Model().eval().to(device)
        classifier = B_Classifier().to(device)
        
        model = loadModel(m_load_path)
        classifier = loadModel(c_load_path)

        # print("==========================================================")
        # print(model)
        # print("==========================================================")
        # print("#params : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        # print("==========================================================")

        test_dataloader = getDataLoader(bug_reports, project, test=True)

        full_res = []
        full_labels = []
        classifier = B_Classifier().to(device)
        for batch in test_dataloader:
            nl, sc_nl, sc_pl, label = batch

            nl = torch.tensor(nl, dtype=torch.float)
            sc_nl = torch.tensor(sc_nl, dtype=torch.float)
            sc_pl = torch.tensor(sc_pl, dtype=torch.float)
            # label = torch.LongTensor([label])
            label = torch.FloatTensor([label]) # [1]
            # label = makeLabel(label)

            inp = (nl.to(device), sc_nl.to(device), sc_pl.to(device))
            nl_out, pl_out = model(inp)

            result = classifier(nl_out, pl_out)

            result = result.view(-1) #[2]
            result = result[1].view(1) # softmax

            full_res.extend(result.detach().cpu().numpy())
            full_labels.extend(label.detach().cpu().numpy())
            
        return full_res, full_labels

def evaluation(bug_report, results, labels):
    print(bug_report.getBugId())
    
    results = [ (res, lab) for res, lab in zip(results, labels) ]
    results = sorted(results, key=lambda x:x[0], reverse=True)

    print("Total Candidates: ", len(labels))
    # print(results)
    topk(results)
    print()

def evaluation_sim(bug_report, results):
    print(bug_report.getBugId())
    results = [ (res, lab) for res, lab in results ]
    results = sorted(results, key=lambda x:x[0], reverse=True)

    print("Total Candidates: ", len(results))
    # print(results)
    return topk(results)

def topk(results):
    top_n = []
    for i, (probability, label) in enumerate(results):
        if label == 1:
            top_n.append(i+1)
    return top_n