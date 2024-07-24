import math, re
import sys, ast
import random


def coerce(s):
    def fun(s2):
        return None if s2.lower() in ["null", "nil", "none"] else s2.lower() == "true" or (s2.lower() != "false" and s2)
    try:
        return float(s)
    except ValueError:
        return fun(re.match(r'^\s*(.*\S)', s).group(1)) if isinstance(s, str) else s

def cells(s):
    t = []
    for s1 in s.split(','):
        t.append(coerce(s1))
    return t

def csv(src):
    i=0
    src = sys.stdin if src == "−" else open(src ,"r")
    for s in src:
        i += 1
        yield i, cells(s)
    src.close()

def roundoff(n, ndecs=None):
    if type(n) == str:
        return n
    if math.floor(n) == n:
        return n
    if not ndecs: 
        ndecs = 2
    mult = 10**(ndecs)
    return math.floor(n * mult + 0.5) / mult

def shuffle(t):
    u = list(t)
    for i in range(len(u) - 1, 0, -1):
        j = random.randint(0,i)
        u[i], u[j] = u[j], u[i]
    return u

def slice(t, go=None, stop=None, inc=None):
    if go is not None and go < 0:
        go = len(t) + go
    if stop is not None and stop < 0:
        stop = len(t) + stop
    u = []
    for j in range((go or 1)//1, (stop or len(t))//1, (inc or 1)//1):
        u.append(t[j])
    return u

def o(t, n=None, u=None):
    if isinstance(t, (int, float)):
        return str(roundoff(t, n))
    if not isinstance(t, dict) and not isinstance(t, list):
        return str(t)
    u = []
    for k, v in t.items() if isinstance(t, dict) else enumerate(t):
        if str(k)[0] != '_':
            if len(t) > 0:
                u.append(o(v, n))
            else:
                u.append(f"{o(k, n)}: {o(v, n)}")
    return "{" + ", ".join(u) + "}"

def anyCustom(t):
    return random.choice(t)

def many(t, n=None):
    n = n or len(t)
    u = []
    for _ in range(n):
        u.append(any(t))
    return u

def keysort(t, fun):
    u = [{'x': x, 'y': fun(x)} for x in t]
    u.sort(key=lambda a: a['y'])
    v = [xy['x'] for xy in u]
    return v