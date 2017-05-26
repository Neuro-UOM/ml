import numpy as np

class csvImporter: 

    def __init__(self, pathName):
        self.data = np.loadtxt(pathName , delimiter=',', dtype=np.float32, skiprows=1)

    def getData(self):
        return self.data
   