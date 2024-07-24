"""
mylo: recursive bi-clustering via random projections (lo is less. less is more. go lo)
(c) 2023, Tim Menzies, BSD-2

USAGE:
  lua mylo.lua [OPTIONS]

OPTIONS:
  -b --bins   max number of bins              = 16
  -B --Beam   max number of ranges            = 10
  -c --cohen  small effect size               = .35
  -C --Cut    ignore ranges less than C*max   = .1
  -d --d      frist cut                       = 32
  -D --D      second cut                      = 4
  -f --file   csv data file name              = ../data/diabetes.csv
  -F --Far    how far to search for faraway?  = .95
  -h --help   show help                       = false
  -H --Half   #items to use in clustering     = 256
  -p --p      weights for distance            = 2
  -s --seed   random number seed              = 31210
  -S --Support coeffecient on best            = 2
  -t --todo   start up action                 = help]]
  """

from helper import *

def settings(s):
    inp = {}
    s_inp = {}
    options = re.findall(r'-(\w+)\s+--(\w+)\s+.*=\s*(\S+)', s)
    for option in options:
        short_form, full_form, default_value = option
        inp[full_form] = coerce(default_value)
        s_inp[short_form] = full_form
    options_dict = {}
    options = sys.argv[1:]

    if("--help" in options or "-h" in options):
        inp["help"]=True
        return inp,s_inp

    for i in range(0, len(options), 2):
        options_dict[options[i]] = options[i+1]

    for opt,val in options_dict.items():
        if opt.startswith('--'):
            inp[opt[2:]] = coerce(val)
        elif opt.startswith('-'):
            inp[s_inp[opt[1:]]] = coerce(val)

    return inp,s_inp

help_str = __doc__

the,_ = settings(help_str)
