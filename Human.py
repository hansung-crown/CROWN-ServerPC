class Human():

    name = ''
    max = 15
    count = 0
    recognition = False
    check = False
    groupName = ""

    #def __init__(self, name):
    #    self.name = name
    #    self.max = 15
    #    self.count = 0
    #    self.recognition = False
    #    self.check = False
    #    self.groupName = ""

    def getName(self):
        return self.name

    def getGroupName(self):
        return self.groupName

    def getMax(self):
        return self.max

    def getCount(self):
        return self.count

    def getRecognition(self):
        return self.recognition

    def getCheck(self):
        return self.check

    def setCount(self, count):
        self.count = count

    def setRecognition(self, recognition):
        self.recognition = recognition

    def setCheck(self, check):
        self.check = check