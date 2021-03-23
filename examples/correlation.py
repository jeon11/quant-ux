from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
from numpy import cov
from scipy.stats import pearsonr
from scipy.stats import spearmanr


# seed random number generator
seed(1)
# prepare data
data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)
# summarize
print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))

# calculate covariance matrix
covariance = cov(data1, data2)
print(covariance)


corr, _ = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)


# calculate spearman's correlation
corr, _ = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)

# plot
pyplot.scatter(data1, data2)
pyplot.show()
