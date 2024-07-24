import math
# from config import the

class SYM:
    # Create: Creates the initial object of type SYM 
    def __init__(self, s=" ", n=0):    
        self.txt = s
        self.at = n
        self.n = 0
        self.has = {}
        self.mode = None
        self.most = 0

    # Update: Used when update the SYM type cols when data rows are added
    def add(self, x):
        if x != "?":
            self.n = self.n + 1
            self.has[x] = 1 + self.has.get(x, 0)
            if self.has[x] > self.most:
                self.most, self.mode = self.has[x], x

    #Query to return the mode of the column
    def mid(self):
        return self.mode
    
    #Diversity of the a sym class calculated with entropy
    def div(self, e=0):
        for v in self.has.values():
            e -= v / self.n * math.log(v / self.n, 2)
        return e

    def small(self, _):
        return 0

    #Calculate teh likelihood of the sym col
    def like(self, x, prior, the):
        tmp = 0
        if x in self.has:
            tmp = self.has[x]
        if the['m']==0:
            the['m']=0.000001
        return ((tmp) + the['m'] * prior) / (self.n + the['m'])