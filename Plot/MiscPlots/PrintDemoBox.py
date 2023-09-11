import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# sns.set(rc={'figure.figsize':(6,5)})
comb_demo = pd.read_csv('comb_demographics')
comb_demo = comb_demo.dropna()
print(f'total number of genuine users who gave the demo infor {comb_demo.shape[0]-9}')
comb_demo= pd.melt(comb_demo, id_vars =['UserType'], value_vars =['Age:yr','Height:cm','Weight:lb','Waist:cm'], var_name ='Characteristics', value_name ='Values')
# ax = sns.boxplot(x="demo", y="demo_values", hue="UserType",data=comb_demo, whis="range", palette="vlag")
# ax = sns.violinplot(x="Characteristics", y="Values", hue="UserType",data=comb_demo, size=5, order=['Height (cm)','Weight (lb)','Waist (cm)','Age (yr)'])
ax = sns.swarmplot(x="Characteristics", y="Values", hue="UserType",data=comb_demo, size=4, order=['Height:cm','Weight:lb','Waist:cm','Age:yr'])
plt.yticks(list(np.linspace(start=0,stop=280,num=8)))
plt.grid(True)
plt.subplots_adjust(left=0.12, bottom=0.12, right=.98, top=0.97, wspace=None, hspace=None)
plt.show()
plt.tight_layout()