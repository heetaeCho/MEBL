from utils import loadBugReport, divideData, saveBugReport
from DataProcessor.ShapeProcessor import getDataLoader, setAllCandidates
from Process.Train import Train
from Process.Eval import validation

train_loss = []
valid_loss = []


def writeLoss(path, epoch, loss):
    with open(path, 'a', encoding='utf-8') as f:
        f.write('{},{}\n'.format(epoch+1, loss))


def pipeline(train_bug_reports, valid_bug_reports, project):
    epoch = 100
    model_path = "./SavedModel/{}_bs0_#layers4_{}_{}.pt"
    trainer = Train()

    for e in range(1, epoch+1):
        m_load_path = model_path.format(project, e-1, 'model')
        m_save_path = model_path.format(project, e, 'model')

        c_load_path = model_path.format(project, e-1, 'clf')
        c_save_path = model_path.format(project, e, 'clf')

        train_dataloader = getDataLoader(train_bug_reports, project)
        loss = trainer.train(train_dataloader, m_save_path, c_save_path, e)
        t_path = './train_loss.txt'
        writeLoss(t_path, e, loss)
        print("{} epoch loss = {}".format(e, loss))
        del train_dataloader

        valid_dataloader = getDataLoader(valid_bug_reports, project)
        loss = validation(valid_dataloader, m_save_path, c_save_path)
        v_path = './valid_loss.txt'
        writeLoss(v_path, e, loss)
        print("{} epoch validation loss = {}".format(e, loss))
        del valid_dataloader

if __name__ == "__main__":
    project = "AspectJ"
    project = "Tomcat"
    path = './dataset/BugReports/{}_with_code.plk'.format(project)
    bug_reports = loadBugReport(path)

    setAllCandidates(bug_reports)
    new_path = "./Dataset/BugReports/{}_with_All_Candidates.plk".format(project)
    saveBugReport(bug_reports, new_path)
    bug_reports = loadBugReport(new_path)

    train_br, valid_br, _ = divideData(bug_reports)

    pipeline(train_br, valid_br, project)