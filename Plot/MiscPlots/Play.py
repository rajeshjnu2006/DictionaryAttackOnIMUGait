from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
tips = sns.load_dataset("tips")
print(tips)
ax = sns.boxplot(x=tips["total_bill"])
# ax = sns.boxplot(x="day", y="total_bill", data=tips)
ax = sns.boxplot(x="day", y="total_bill", hue="smoker",
                 data=tips, palette="Set3")

plt.show()