import pandas as pd
import researchpy as rp
import scipy.stats as stats

df = pd.read_csv("interfaceResultsT-Test.csv")
df.info()

summary, results = rp.ttest(group1= df['time_after'][df['sex'] == 'Male'], group1_name= "Male",
                            group2= df['time_after'][df['sex'] == 'Female'], group2_name= "Female")
print(summary)
print(results)
