from DataProcessor.BenchmarkProcessor import readBenchmarkData, generateBugReports
from DataProcessor.GitProcessor import getRepo, getAllCommits, copy_sc

from utils import generateDirs, loadBugReport, saveBugReport
import pickle

def updateBugReportsCommit(bug_reports, repo):
    all_commits = getAllCommits(repo)

    prev_commit = None
    for bug_report in bug_reports:
        commit = bug_report.getFixedCommit()
        for i in range(len(all_commits)):
            this_commit = all_commits[i]
            if str(this_commit).startswith(str(commit)):
                if bug_report.getBuggyCommit() is None:
                    bug_report.setFixedCommit(str(this_commit))
                    bug_report.setBuggyCommit(str(prev_commit))
                else:
                    print(this_commit)
                break
            else:
                prev_commit = this_commit

def checkSaved(path):
    with open(path, 'rb') as fr:
        bug_reports = pickle.load(fr)
    
    buggy_commits = []
    fixed_commits = []
    print("======== New Files ========")
    for bug_report in bug_reports:
        print(bug_report.getNewFiles())
        buggy_commits.append(bug_report.getBuggyCommit())
        fixed_commits.append(bug_report.getFixedCommit())

    from collections import Counter
    buggy_commits = Counter(buggy_commits)
    fixed_commits = Counter(fixed_commits)

    print("==== buggy_commit ====")
    for com, cnt in buggy_commits.items():
        if cnt > 1 or len(com) < 8:
            print(cnt, com)

    print("==== fixed_commit ====")
    for com, cnt in fixed_commits.items():
        if cnt > 1 or len(com) < 8:
            print(cnt, com)

def sourceCodeCopyAndUpdateBR(repo, bug_reports, path):
    copy_sc(repo, bug_reports, path)


if __name__ == "__main__":
    # projects = {'AspectJ':"org.aspectj", 'Birt':'birt', 'Eclipse_Platform_UI':'eclipse.platform.ui', 'JDT':'eclipse.jdt.ui', 'SWT':'eclipse.platform.swt', 'Tomcat':'tomcat'}
    # projects = {'AspectJ':"org.aspectj", 'Birt':'birt', 'Eclipse_Platform_UI':'eclipse.platform.ui', 'JDT':'eclipse.jdt.ui', 'SWT':'eclipse.platform.swt'}
    # projects = {'Tomcat':'tomcat'}
    projects = {'AspectJ':"org.aspectj"}
    generateDirs(projects)

    benchmark_path = "./Dataset/benchmark/{}.xlsx"
    bug_report_save_path = "./Dataset/BugReports/{}.plk"
    sourcecode_save_path = "./Dataset/SourceCodes/{}/"
    project_git_path = "C:/datasets/2023TR/projects/{}/"

    for project in projects.keys():
        print("\n============= {} ============".format(project))
        benchmark_data = readBenchmarkData(benchmark_path.format(project))
        bug_reports = generateBugReports(benchmark_data)
        # print(bug_reports[0])

        repo = getRepo(project_git_path.format(projects.get(project)))

        ## take some minute because n^2 searching
        updateBugReportsCommit(bug_reports, repo)

        sourceCodeCopyAndUpdateBR(repo, bug_reports, sourcecode_save_path.format(project))

        saveBugReport(bug_reports, bug_report_save_path.format(project))
        checkSaved(bug_report_save_path.format(project))