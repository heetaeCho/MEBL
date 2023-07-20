from DataStructure.BugReport import BugReport
import pandas as pd

def readBenchmarkData(path):
    df = pd.read_excel(path)
    return df.sort_values(by=["commit_timestamp"]).reset_index()

def generateBugReports(data_frame):
    bug_reports = []
    for i in range(len(data_frame)):
        bug_reports.append(_generateBugReport(data_frame.iloc[i]))
    return bug_reports

def _generateBugReport(serise):
    bug_report = BugReport()
    bug_report.setBugId(serise["bug_id"])
    bug_report.setFixedCommit(serise["commit"])
    
    if pd.isna(serise["summary"]):
        bug_report.setSummary("")
    else:
        bug_report.setSummary(serise["summary"])
    
    if pd.isna(serise["description"]):
        bug_report.setDescription("")
    else:
        bug_report.setDescription(serise["description"])
    
    bug_report.setFiles(_split_file_string(serise["files"]))
    return bug_report

def _split_file_string(files_string):
    new_files = []
    files = files_string.split('.java ')
    for file in files:
        if file.endswith('.java'):
            new_files.append(file)
        else:
            new_files.append(file+'.java')

    return new_files

# def generateSourceCodes(bug_reports):
#     source_codes = []
#     for bug_report in bug_reports:
#         for file in bug_report.getFiles():
#             source_codes.append(SourceCode(commit=bug_report.getFxiedCommit(), file=file))
#     return source_codes

# def generateBugReportSourceCodeMap(bug_reports):
#     return {bug_report.getBugId(): bug_report.getFiles() for bug_report in bug_reports}