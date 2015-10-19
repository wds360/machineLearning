import numpy as np
from subprocess import call

exprange = np.unique(np.logspace(0,10,base=2).astype(int))

for i in exprange:
    call(['java', '-cp', 'ABAGAIL.jar', 'opt.test.FourPeaksTest', '1000', str(i)])
