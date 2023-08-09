import torch

from utils import loadBugReport, saveBugReport
from DataProcessor.ShapeProcessor import getCodeBertDataLoader, setAllCandidates
from Process.Eval import evaluation_sim

train_loss = []
valid_loss = []

def writeLoss(path, epoch, loss):
    with open(path, 'a', encoding='utf-8') as f:
        f.write('{},{}\n'.format(epoch+1, loss))

def pipeline(bug_reports, project):
    print("Total #bug_reports: ", len(bug_reports))

    br_id_2_br = {}

    results = {}
    for bug_report in bug_reports:
        dataloader = getCodeBertDataLoader([bug_report], project, test=True)

        br_id_2_br[bug_report.getBugId()] = bug_report

        results[bug_report.getBugId()] = []

        for batch in dataloader:
            nl, sc_nl, _, label = batch

            nl = torch.mean(torch.tensor(nl, dtype=torch.float), dim=1)
            sc_nl = torch.mean(torch.tensor(sc_nl, dtype=torch.float), dim=1)
            label = torch.FloatTensor([label]) # [1]

            sim = torch.cosine_similarity(nl, sc_nl).mean()
            results[bug_report.getBugId()].append( (sim, label) )

    total = 0
    hit_1 = 0
    hit_10 = 0
    hit_100 = 0
    hit_1000 = 0
    hit_over_1000 = 0
    for key in results:
        top_n = evaluation_sim(br_id_2_br[key], results[key])
        for n in top_n:
            total += 1
            if n > 1000:
                hit_over_1000 += 1
            if n <= 1000:
                hit_1000 += 1
            if n <= 100:
                hit_100 += 1
            if n <= 10:
                hit_10 += 1
            if n == 1:
                hit_1 += 1

    print('=====================================================')
    print("Project: ", project)
    print('=====================================================')
    print("Total: ", total)
    print("hit_1: ", hit_1)
    print("hit_10: ", hit_10)
    print("hit_100: ", hit_100)
    print("hit_1000: ", hit_1000)
    print("hit_over_1000: ", hit_over_1000)
    print('=====================================================')
    print("hit_1_ratio: {} %".format(hit_1/total*100))
    print("hit_10_ratio: {} %".format(hit_10/total*100))
    print("hit_100_ratio: {} %".format(hit_100/total*100))
    print("hit_1000_ratio: {} %".format(hit_1000/total*100))
    print("hit_over_1000_ratio: {} %".format(hit_over_1000/total*100))
    print('=====================================================')
    
    with open('./{}_sim_res.txt'.format(project), 'w', encoding='utf-8') as f:
        f.write('=====================================================\n')
        f.write("Total: {}\n".format(total))
        f.write("hit_1: {}\n".format(hit_1))
        f.write("hit_10: {}\n".format(hit_10))
        f.write("hit_100: {}\n".format(hit_100))
        f.write("hit_1000: {}\n".format(hit_1000))
        f.write("hit_over_1000: {}\n".format(hit_over_1000))
        f.write('=====================================================\n')
        f.write("hit_1_ratio: {} %\n".format(hit_1/total*100))
        f.write("hit_10_ratio: {} %\n".format(hit_10/total*100))
        f.write("hit_100_ratio: {} %\n".format(hit_100/total*100))
        f.write("hit_1000_ratio: {} %\n".format(hit_1000/total*100))
        f.write("hit_over_1000_ratio: {} %\n".format(hit_over_1000/total*100))
        f.write('=====================================================\n')

if __name__ == "__main__":
    # project = "AspectJ"
    # project = "Tomcat"
    projects = ["AspectJ", "Tomcat"]

    for project in projects:
        path = './dataset/BugReports/{}_with_code.plk'.format(project)
        bug_reports = loadBugReport(path)

        setAllCandidates(bug_reports)
        new_path = "./Dataset/BugReports/{}_with_All_Candidates.plk".format(project)
        saveBugReport(bug_reports, new_path)
        bug_reports = loadBugReport(new_path)

        pipeline(bug_reports, project)