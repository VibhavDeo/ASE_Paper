"""
gate: guess, assess, try, expand
(c) 2023, Tim Menzies, BSD-2
Learn a little, guess a lot, try the strangest guess, repeat

USAGE:
  python3 gate.lua [OPTIONS] 

OPTIONS:
  -c --cohen  small effect size               = .35
  -f --file   csv data file name              = w2/data/auto93.csv
  -h --help   show help                       = False
  -k --k      low class frequency kludge      = 1
  -m --m      low attribute frequency kludge  = 2
  -s --seed   random number seed              = 31210
  -t --test   run test cases                  = None
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
