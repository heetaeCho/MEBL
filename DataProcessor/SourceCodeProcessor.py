from DataStructure.Code import Code
from DataStructure.CodeFile import CodeFile

import subprocess
import json
from tqdm import tqdm

def readCode(file):
    jar_path = './Java/CodeExtractor_Jar/target/CodeExtractor_Jar-0.0.1-jar-with-dependencies.jar'
    jar_data = subprocess.check_output('java -jar {} "{}"'.format(jar_path, file)).decode('cp949')
    jar_data = json.loads(jar_data)
    
    if jar_data["error"]:
        return True, True

    codeFileData = CodeFile()
    codeFileData.setCommit(file.split('_')[0])
    codeFileData.setQualifiedName(file)
    codeFileData.setError(jar_data['error'])

    codes = []
    for code in jar_data["codeChunks"]:
        codeData = Code(**code)
        codes.append(codeData)
    codeFileData.setCodeChunks(codes)
    return codeFileData, False

def readAndUpdateCodeData(bug_reports):
    jar_path = './Java/CodeExtractor_Jar/target/CodeExtractor_Jar-0.0.1-jar-with-dependencies.jar'

    for bug_report in tqdm(bug_reports, total=len(bug_reports), ncols=100, leave=True):
        bug_report.setCodeData()

        for file in bug_report.getNewFiles():
            jar_data = subprocess.check_output('java -jar {} "{}"'.format(jar_path, file)).decode('cp949')
            jar_data = json.loads(jar_data)

            codeFileData = CodeFile()
            codeFileData.setCommit(file.split('_')[0])
            codeFileData.setQualifiedName(file)
            codeFileData.setError(jar_data['error'])

            codes = []
            for code in jar_data["codeChunks"]:
                codeData = Code(**code)
                codes.append(codeData)
            codeFileData.setCodeChunks(codes)

            bug_report.addCodeData(codeFileData)