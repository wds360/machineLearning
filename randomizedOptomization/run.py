import numpy as np
from subprocess import call

exprange = np.unique(np.logspace(0,15,base=2).astype(int))

for i in exprange:
    call(['java', '-cp', 'ABAGAIL.jar', 'opt.test.OptimizeNeualNet', str(i)])
