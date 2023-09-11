import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
gen_demo = pd.read_csv('gen_demographics.csv')
gen_demo = gen_demo.dropna()
print('gen_demo', gen_demo.describe())
gen_demo['UserType'] = ['Genuine']*gen_demo.shape[0]
imp_demo = pd.read_csv('imp_demographics.csv')
imp_demo = imp_demo.dropna()
print('imp_demo', imp_demo.describe())
imp_demo['UserType'] = ['Impostor']*imp_demo.shape[0]

# https://stackoverflow.com/questions/13872533/plot-different-dataframes-in-the-same-figure
# styles=['bs-', 'bs-', 'ro-', 'y^-']

# ax = gen_demo.plot.kde(bw_method=0.5, colormap='tab10', ms=1)
# imp_demo.plot.kde(bw_method=0.5, ax=ax, style = '--',colormap='tab10', ms=1)
ax = gen_demo.boxplot()
imp_demo.boxplot(ax=ax)
plt.grid(True)
plt.show()
plt.tight_layout()