import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
matplotlib.style.use('ggplot')

results = pd.DataFrame.from_csv(sys.argv[1],header=0, index_col=False)
results.columns = ['algorithm','iterations','train_error','test_error','train_time','test_time']

rhc = results[results['algorithm'] == 'RHC']
sa = results[results['algorithm'] == 'SA']
ga = results[results['algorithm'] == 'GA']

rhc.plot(x='iterations', y=['train_error', 'test_error'], logx=True, title='Randomized Hill Climbing')
sa.plot(x='iterations', y=['train_error', 'test_error'], logx=True,  title='Simulated Annealing')
ga.plot(x='iterations', y=['train_error', 'test_error'], logx=True,  title='Genetic Algorithm')

rhc = rhc.reset_index()
sa = sa.reset_index()
ga = ga.reset_index()

grouped = pd.concat([rhc['iterations'], rhc['train_time'], sa['train_time'], ga['train_time']], axis=1)
grouped.columns = ['iterations', 'rhc', 'sa', 'ga']

grouped.dropna().plot(x='iterations', y=['rhc','sa','ga'], logx=True, title='Wall-clock Training Duration in Seconds')

err_grouped = pd.concat([rhc['test_error'],sa['test_error'],ga['test_error']], axis=1)
err_grouped.columns = ['rhc', 'sa', 'ga']

plt.figure();
err_grouped.min().plot(kind='bar', title='Optimal Neural Network Test MSE by Training Algorithm')

plt.show()
