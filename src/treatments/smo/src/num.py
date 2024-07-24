
class NUM:
    # Inititalizing  NUM instance
    def __init__(self, s = "", n=0):
        self.txt = s
        self.at = n
        self.n = 0
        self.mu = 0
        self.m2 = 0
        self.hi = -1E30
        self.lo = 1E30
        self.heaven = 0 if s.endswith("-") else 1
    
    # Updating class variables
    def add(self, x):
        if x != "?":
            self.n += 1
            d = x - self.mu
            self.mu += d/self.n
            self.m2 += d*(x - self.mu)
            self.lo = min(x, self.lo)
            self.hi = max(x, self.hi)

    # Class methods (queries)
            
    # Method to return mean of NUM
    def mid(self):
        return self.mu
    # Method to calculate standard deviation of NUM
    def div(self):
        return 0 if self.n < 2 else (self.m2 / (self.n - 1)) ** 0.5
    # Method to calcuate small value of NUM based on cohen's factor given in the input
    def small(self, the):
        return the['cohen'] * self.div()
    # Method to normalize the value
    def norm(self, x):
        return x if x == "?" else (x - self.lo) / (self.hi - self.lo + 1E-30)
    # Method to find the likelihood based on normal distribution
    def like(self, x, _, nom=None, denom=None):
        mu, sd = self.mid(), self.div() + 1E-30
        nom = 2.718**(-0.5 * (x - mu)**2 / (sd**2))
        denom = sd * 2.5 + 1E-30
        return nom / denom 
