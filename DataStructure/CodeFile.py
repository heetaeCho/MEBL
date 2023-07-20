class CodeFile:
    def __init__(self, commit=None, qualified_name=None,\
                 error=None, code_chunks=None):
        self.commit = commit
        self.qualified_name = qualified_name
        self.error = error
        self.code_chunks = code_chunks

    def setCommit(self, commit):
        self.commit = commit
    def setQualifiedName(self, qualified_name):
        self.qualified_name = qualified_name
    def setError(self, error):
        self.error = error
    def setCodeChunks(self, code_chunks):
        self.code_chunks = code_chunks

    def getCommit(self): return self.commit
    def getQualifiedName(self): return self.qualified_name
    def getError(self): return self.error
    def getCodeChunks(self): return self.code_chunks