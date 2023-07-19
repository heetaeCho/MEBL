import os

def generateDirs(projects):
    projects = list(projects.keys())

    br_path = './Dataset/BugReports/'
    os.makedirs(br_path, exist_ok=True)

    sc_path = './Dataset/SourceCodes/{}/'
    for project in projects:
        os.makedirs(sc_path.format(project), exist_ok=True)
