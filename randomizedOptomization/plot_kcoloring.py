import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
matplotlib.style.use('ggplot')

results = pd.DataFrame.from_csv(sys.argv[1],header=0, index_col=False)
results.columns = ['algorithm','optimum','time_ms']

rhc = results[results['algorithm'] == 'RHC']
sa = results[results['algorithm'] == 'SA']
ga = results[results['algorithm'] == 'GA']
mimic = results[results['algorithm']=='MIMIC']

rhc = rhc.reset_index()
sa = sa.reset_index()
ga = ga.reset_index()
mimic = mimic.reset_index()

time_grouped = pd.concat([rhc['time_ms'], sa['time_ms'], ga['time_ms'], mimic['time_ms']], axis=1)
time_grouped.columns = ['rhc', 'sa', 'ga', 'mimic']

time_grouped.plot(title='Wall-clock time over K-colors')

opt_grouped = pd.concat([rhc['optimum'], sa['optimum'], ga['optimum'], mimic['optimum']], axis=1)
opt_grouped.columns = ['rhc', 'sa', 'ga', 'mimic']

opt_grouped.plot(kind='bar', title="Number of iterations over K-colors")

plt.show()
