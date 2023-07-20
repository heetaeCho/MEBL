import os
import pickle

def generateDirs(projects):
    projects = list(projects.keys())

    br_path = './Dataset/BugReports/'
    os.makedirs(br_path, exist_ok=True)

    sc_path = './Dataset/SourceCodes/{}/'
    for project in projects:
        os.makedirs(sc_path.format(project), exist_ok=True)

def loadBugReport(path):
    with open(path, 'rb') as fr:
        return pickle.load(fr)
    
def saveBugReport(bug_reports, path):
    with open(path, 'wb') as fw:
        pickle.dump(bug_reports, fw)

def loadSourceCode(path):
    with open(path, 'rb') as fr:
        return pickle.load(fr)

def saveSourceCode(source_code, path):
    with open(path, 'wb') as fw:
        pickle.dump(source_code, fw)

def divideData(data):
    total = len(data)

    train_valid_size = int(total * 0.9)
    train_valid = data[:train_valid_size]
    test = data[train_valid_size:]

    train_size = int(len(train_valid) * 0.9)
    train = train_valid[:train_size]
    valid = train_valid[train_size:]

    return train, valid, test

import torch
def loadModel(path):
    return torch.load(path)

def saveModel(model, path):
    torch.save(model, path)