import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks")
exercise = sns.load_dataset("exercise")
print(exercise)
g = sns.catplot(x="time", y="pulse", hue="kind", data=exercise)
plt.show()