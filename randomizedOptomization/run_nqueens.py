import numpy as np
from subprocess import call

exprange = np.unique(np.logspace(0,7,base=2).astype(int))

for i in exprange:
    call(['java', '-cp', 'ABAGAIL.jar', 'opt.test.NQueensTest', str(i)])
