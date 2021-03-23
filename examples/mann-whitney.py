import numpy as np
from scipy.stats import mannwhitneyu

sample1=[32, 34, 29, 39, 38, 37, 38, 36, 30, 26]
sample2=[40, 34, 30, 39, 38, 37, 38, 36, 50, 49]

stat, pvalue=mannwhitneyu(sample1, sample2)
print('statistics=%.3f, p=%.5f'%(stat,pvalue))
alpha=0.05

if pvalue> alpha:
  print('Two Groups are from the Same distribution(fail to reject H0) under alpha=0.05')
else:
  print('Two Groups are from Different distributions(reject H0) under alpha=0.05')
