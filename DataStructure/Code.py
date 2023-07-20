class Code:
    def __init__(self, position=None, declaration=None, code=None):
        self.position = position
        self.declaration = declaration
        self.code = code

    def setPosition(self, position):
        self.position = position
    def setDeclaration(self, declaration):
        self.declaration = declaration
    def setCode(self, code):
        self.code = code

    def getPosition(self): return self.position
    def getDeclaration(self): return self.declaration
    def getCode(self): return self.code