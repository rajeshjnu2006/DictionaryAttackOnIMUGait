import pandas as pd
import numpy as np
ZeroEffortFAR = pd.read_csv('test.csv', index_col=0)
print(ZeroEffortFAR.to_string())
import seaborn as sns
from matplotlib import pyplot as plt

user_list = []
for i in range(1, 56):
    user_list.append('User' + str(i))

sns.set(font_scale=0.9)
fig, ax = plt.subplots(3, 1, sharex='col', sharey='row',figsize=(10, 6))
cm = sns.cubehelix_palette(6)
p1 = sns.heatmap(ZeroEffortFAR, annot=True, cmap=cm, ax=ax[0], annot_kws={"size":8}, vmin=0, vmax=100, fmt='.0f',linewidths=0.1, cbar=True, cbar_kws = dict(use_gridspec=False,location="top"))
plt.xlabel("Users")
plt.ylabel("Classifiers")
ax[0].set_title('FAR under zero-effort attack for accelerometer')

p2 = sns.heatmap(ZeroEffortFAR, annot=True, cmap=cm, ax=ax[1], annot_kws={"size":8}, vmin=0, vmax=100, fmt='.0f',linewidths=0.1, cbar=False)
plt.xlabel("Users")
plt.ylabel("Classifiers")
ax[1].set_title('FRR under zero-effort attack for accelerometer')
p3 = sns.heatmap(ZeroEffortFAR, annot=True, cmap=cm, ax=ax[2], annot_kws={"size":8}, vmin=0, vmax=100, fmt='.0f',linewidths=0.1, cbar=False)
plt.xlabel("Users")
plt.ylabel("Classifiers")
plt.xticks(np.linspace(0.5,56.5,56, endpoint=False),user_list+['Average'])
ax[2].set_title('HTER under zero-effort attack for accelerometer')
plt.subplots_adjust(left=0.08, bottom=0.17, right=.98, top=0.80, wspace=0.20, hspace=0.20)

plt.show()