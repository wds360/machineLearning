import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

results = pd.DataFrame.from_csv('r.csv',header=0, index_col=False)
results.columns = ['algorithm','iterations','train_error','test_error','train_time','test_time']

rhc = results[results['algorithm'] == 'RHC']
sa = results[results['algorithm'] == 'SA']
ga = results[results['algorithm'] == 'GA']

rhc.plot(x='iterations', y=['train_error', 'test_error'], title='Randomized Hill Climbing')
sa.plot(x='iterations', y=['train_error', 'test_error'], title='Simulated Annealing')
ga.plot(x='iterations', y=['train_error', 'test_error'], title='Genetic Algorithm')

plt.show()
