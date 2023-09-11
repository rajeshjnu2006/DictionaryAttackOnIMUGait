import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('percent_succ.txt', index_col=0)
print(df)

# df['a+g+m+r'] =  df['a+g+m+r']/100
# df['a'] =  df['a']/100

# Sorting for making more sense
df = df.sort_values(by='a',ascending=True)

df = df.set_index('User')
df.plot.bar(color=['rosybrown', 'crimson'], rot=90)
plt.legend(['a+g+m+r','a'], ncol=2, loc='upper left')
plt.grid(color='lightgray', linestyle='-', linewidth=0.1)
# plt.ylabel('% of 178 unique patterns which achieved dict_far >= zero_far')
plt.ylabel('% of patterns')
plt.show()
