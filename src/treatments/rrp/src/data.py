#Read data file
from helper import *
from row import ROW
from cols import COLS
from sym import SYM
import random
from config import the
from Node import NODE

class DATA:

    def __init__(self, src=[], fun=None):
        self.rows = []
        self.cols = None
        if isinstance(src, str):
            for _, x in csv(src):
                self.add(x, fun)
        else:
            for row in src:
                self.add(row, fun)

    def add(self, t, fun=None):
        row = t if isinstance(t, ROW) else ROW(t)
        if self.cols:
            if fun:
                fun(self, row)
            self.cols.add(row)
            self.rows.append(row)
        else:
            self.cols = COLS(row)

    def stats(self, fun = None, ndivs = None):
        u = {".N": len(self.rows)}
        for col in self.cols.all:
            if(isinstance(col, SYM)):
                u[col.txt] = int(col.mid())
            else:
                u[col.txt] = roundoff(col.mid())
        return u
    
    def classes_data(self):
        table = {}
        col = self.cols.klass
        table["N"] = len(self.rows)
        table["klasses"] = len(col.has)
        for key, val in col.has.items():
            table[key] = val
            table[key+"%"] = roundoff(val*100/len(self.rows), 2)
        return table
    
    def mid(self, cols=None):
        u = []
        for col in cols or self.cols.all:
            u.append(col.mid())
        return ROW(u)
    
    def div(self, cols=None):
        u = []
        for col in cols or self.cols.all:
            u.append(col.div())
        return ROW(u)

    def small(self):
        u = []
        for col in self.cols.all:
            u.append(col.small(the))
        return ROW(u)
    
    def clone(self, rows=None, newData=None):
        newData = DATA([self.cols.names])
        for row in rows or []:
            newData.add(row)
        return newData

    def farapart(self,randomSeed,input, rows, sortp, a=None, b=None, far=None, evals=0):
        far = int(len(rows) * 0.95) + 1
        evals = 1 if a is not None else 2
        
        a = a or random.choice(rows)
    
        sorted_neighbors = a.neighbors(self, rows)
        a = a or sorted_neighbors[0]
        b = sorted_neighbors[min(far, len(sorted_neighbors) - 1)]
        
        print(a.d2h(input,randomSeed,self), b.d2h(input,randomSeed,self))
        if sortp and b.d2h(input,randomSeed,self) < a.d2h(input,randomSeed,self):
            a, b = b, a
        
        return a, b, a.dist(b, self), evals
    
    def half(self,randomSeed, input, rows, sortp, before):
        the_half = min(len(rows) // 2, len(rows))
        
        some = random.sample(rows, the_half)
        a, b, C, evals = self.farapart(randomSeed, input,some, sortp, before)
        def d(row1, row2):
            return row1.dist(row2, self)
        
        def project(r):
            return (d(r, a)**2 + C**2 - d(r, b)**2) / (2 * C)
    
        rows_sorted = sorted(rows, key=project)
        mid_point = len(rows) // 2
        as_ = rows_sorted[:mid_point]
        bs = rows_sorted[mid_point:]
        return as_, bs, a, b, C, d(a, bs[0]), evals
    
    
    def branch(self,randomSeed,input, stop=None, rest=None, _branch=None, evals=None):
        evals, rest = 1, []
        stop = stop or (2 * (len(self.rows) ** 0.5))
        random.seed(randomSeed)
        
        def _branch(data, above=None, left=None, lefts=None, rights=None):
            nonlocal evals, rest

            if len(data.rows) > stop:
                lefts, rights, left, _, _, _, _  = self.half(randomSeed,input,data.rows, True, above)
                evals += 1
                for row1 in rights:
                    rest.append(row1)

                return _branch(self.clone(lefts), left)
            else:
                return self.clone(data.rows), self.clone(rest), evals

        return _branch(self)