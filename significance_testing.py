import pdb
from analysis import TrainingAnalysis
import numpy as np
import analyze
from RandGen import rand_range
from DataManipulation import normalize

reload(analyze)

TRIALS = 20000

rdata = np.random.random(TRIALS)

p_up = .35
p_down = .33

pos_change = .05
neg_change = -.05
confidence = .60
max_error = .01

data = []

# pdb.set_trace()

for r in rdata:
    if 0 <= r < p_down:
        data.append(neg_change-.05)
    elif p_down <= r < p_down + p_up:
        data.append(pos_change+.05)
    else:
        data.append(0.)

data = np.array(data)

print data

print analyze.significance(data, pos_change, neg_change, confidence, max_error)
