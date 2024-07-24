#Read data file
from helper import *
from row import ROW
from cols import COLS
from sym import SYM
import random
from config import the



class DATA:

    def __init__(self, src=[], fun=None):
        self.rows = []
        self.cols = None
        if isinstance(src,str):
            for _,x in csv(src):
                self.add(x, fun)
        else:
            #for _,x in enumerate(src):
            self.add(src, fun)
    
    def add(self, t, fun=None):
        row = t if type(t) == ROW else ROW(t)
        # row = ROW(t) if type(t) == list else t
        if self.cols:
            if fun:
                fun(self, row)
            self.rows.append(self.cols.add(row))
        else:
        #    print('here', row)
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
    
    
    def gate(self, randomSeed, budget0, budget, some,input):
        random.seed(randomSeed)

        rows = random.sample(self.rows, len(self.rows))

        lite = rows[:budget0]
        dark = rows[budget0:]
        
        stats, bests = [], []
        for i in range(budget):
            best, rest = self.best_rest(randomSeed,lite, len(lite)**some,input)
            todo, selected = self.split(best, rest, lite, dark)

            stats.append(selected.mid())
            bests.append(best.rows[0])
            lite.append(dark.pop(todo))

        lite.sort(key=lambda a: a.d2h(input,randomSeed,data=None),reverse=False)


        return lite[0], lite[0].cells[len(lite[0].cells)-1]

    def split(self, best, rest, lite, dark):
        selected = DATA(self.cols.names)
        max_value = float('-inf')
        out = 0
        for i, row in enumerate(dark):
            b = row.like(best, len(lite), 2, the)
            r = row.like(rest, len(lite), 2, the)
            if b > r:
                selected.add(row)
            tmp = abs(b + r) / abs(b - r + 1E-300)
            if tmp > max_value:
                out, max_value = i, tmp
        return out, selected

    def best_rest(self,randomSeed,  rows, want,input):
        rows.sort(key=lambda a: a.d2h(input,randomSeed,self), reverse=False)
        print("===========jfhajhfjahfadsfajdfhajsfdhasdhgfasf=========")
        for row in rows:
            print(row.cells[19])
        best = DATA(self.cols.names)
        rest = DATA(self.cols.names)
        for i, row in enumerate(rows):
            if i < want:
                best.add(row)
            else:
                rest.add(row)
        return best, rest
    
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