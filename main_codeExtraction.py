from DataProcessor.SourceCodeProcessor import readAndUpdateCodeData
from utils import loadBugReport, saveBugReport

def checker(bug_report):
    for code_file in bug_report.getCodeData():
        for code_chunk in code_file.getCodeChunks():
            print(code_chunk.getCode())

if __name__ == "__main__":
    project = "AspectJ"
    project = "Tomcat"

    path = './dataset/BugReports/{}.plk'.format(project)
    bug_reports = loadBugReport(path)
    readAndUpdateCodeData(bug_reports)

    path = './dataset/BugReports/{}_with_code.plk'.format(project)
    saveBugReport(bug_reports, path)

    
    bug_reports = loadBugReport(path)
    for bug_report in bug_reports[-3:]:
        checker(bug_report)