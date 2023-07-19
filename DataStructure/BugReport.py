class BugReport:
    def __init__(self, bug_id=None, summary=None, description=None, buggy_commit=None, fixed_commit=None, \
                 files=None, new_files=[]):
        self.bug_id = bug_id
        self.summary = summary
        self.description = description
        self.buggy_commit = buggy_commit
        self.fixed_commit = fixed_commit
        self.files = files
        self.new_files = new_files

    def setBugId(self, bug_id):
        self.bug_id = bug_id
    def setSummary(self, summary):
        self.summary = summary
    def setDescription(self, description):
        self.description = description
    def setBuggyCommit(self, buggy_commit):
        self.buggy_commit = buggy_commit
    def setFixedCommit(self, fixed_commit):
        self.fixed_commit = fixed_commit
    def setFiles(self, files): ## only file_name
        self.files = files

    def addNewFile(self, new_file): ## file_name with commit
        self.new_files.append(new_file)

    def getBugId(self):
        return self.bug_id
    def getSummary(self):
        return self.summary
    def getDescription(self):
        return self.description
    def getBuggyCommit(self):
        return self.buggy_commit
    def getFixedCommit(self):
        return self.fixed_commit
    def getFiles(self):
        return self.files
    def getNewFiles(self):
        return self.new_files
    
    def __str__(self):
        return "bug_id: {}\nsummary: {}\ndescription: {}\nbuggy_commit: {}\nfixed_commit: {}\nfiles: {}\n"\
            .format(self.bug_id, self.summary, self.description, self.buggy_commit, self.fixed_commit, self.files)