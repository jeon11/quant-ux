import os
import random
import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.regression.mixed_linear_model import MixedLMResults
from scipy import stats
import seaborn as sns

# information on experimental design
group_list = ['control','patient1','patient2']
subs_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

# read data into dataframe
df_1way = pd.DataFrame(columns=["group", "my_value"])
my_row = 0
for ind_g, group in enumerate(group_list):
    for sub in subs_list:
        # generate random value here as example
        my_val = np.random.normal(ind_g, 1, 1)[0]
        df_1way.loc[my_row] = [group, my_val]
        my_row = my_row + 1

# inspect data
sns.catplot(x="group", y="my_value", data=df_1way, dodge=True, kind='violin', aspect=3)
plt.show()

# generate model for linear regression
my_model = smf.ols(formula='my_value ~ group', data=df_1way)

# fit model to data to obtain parameter estimates
my_model_fit = my_model.fit()

# print summary of linear regression
print(my_model_fit.summary())

# show anova table
anova_table = sm.stats.anova_lm(my_model_fit, typ=2)
print(anova_table)
F, p = stats.f_oneway(df_1way[df_1way['group'] == 'control'].my_value, df_1way[df_1way['group'] == 'patient1'].my_value, df_1way[df_1way['group'] == 'patient2'].my_value)
print("ANOVA P value:"+str(p))

# information on experimental design
group_list = ['control','patient1','patient2']
language_list = ['English', 'German', 'French']
subs_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

# read data into dataframe
df_2way = pd.DataFrame(columns=["group", "language", "my_value"])
my_row = 0
for ind_g, group in enumerate(group_list):
    for ind_l, lan in enumerate(language_list):
        for sub in subs_list:
                # generate random value here as example
                my_val = np.random.normal(ind_g + ind_l, 1, 1)[0]
                df_2way.loc[my_row] = [group, lan, my_val]
                my_row = my_row + 1

# plot data
sns.catplot(x="language", y="my_value", data=df_2way, dodge=True, hue='group', kind='violin', aspect=3)
plt.show()

# fit model to data to obtain parameter estimates
my_model_fit = smf.ols(formula='my_value ~ group * language', data=df_2way).fit()
# print summary of linear regression
print(my_model_fit.summary())
# show anova table
#From the above results, we can see that the main effects are both significant,
#but the interaction between them isn't.
print(sm.stats.anova_lm(my_model_fit, typ=2))

#p =sm.stats.anova_lm(my_model_fit, typ=2)
#print (p)
