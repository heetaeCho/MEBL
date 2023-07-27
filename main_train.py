import torch
from utils import loadBugReport, divideData
from DataProcessor.ShapeProcessor import getDataLoader
from Process.Train import Train
from Process.Eval import validation

train_loss = []
valid_loss = []

def pipeline(train_bug_reports, valid_bug_reports):
    epoch = 100
    model_path = "./SavedModel/AspectJ_{}_lr_lambda.pt"
    trainer = Train()

    for e in range(1, epoch+1):
        load_path = model_path.format(e-1)
        save_path = model_path.format(e)

        train_dataloader = getDataLoader(train_bug_reports)
        loss = trainer.train(train_dataloader, load_path, save_path, e)
        train_loss.append(loss)
        print("{} epoch loss = {}".format(e, loss))

        valid_dataloader = getDataLoader(valid_bug_reports)
        loss = validation(valid_dataloader, save_path)
        valid_loss.append(loss)
        print("{} epoch validation loss = {}".format(e, loss))

if __name__ == "__main__":
    project = "AspectJ"
    path = './dataset/BugReports/{}_with_code.plk'.format(project)

    bug_reports = loadBugReport(path)
    train_br, valid_br, _ = divideData(bug_reports)

    pipeline(train_br, valid_br)

    # with open('./train_loss.txt', 'w', encoding='utf-8') as f:
    #     for i in range(len(train_loss)):
    #         f.write('{},{}\n'.format(i+1, train_loss[i]))
    
    # with open('./valid_loss.txt', 'w', encoding='utf-8') as f:
    #     for i in range(len(valid_loss)):
    #         f.write('{},{}\n'.format(i+1, valid_loss[i]))