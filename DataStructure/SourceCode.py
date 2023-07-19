class SourceCode:
    def __init__(self, commit=None, file=None):
        self.commit = commit
        self.file = file
    
    def setCommit(self, commit):
        self.commit = commit
    def setFile(self, file):
        self.file = file

    def getCommit(self):
        return self.commit
    def getFile(self):
        return self.file
    
    def __str__(self):
        return "commit: {}\nfile: {}\n"\
            .format(self.commit, self.file)