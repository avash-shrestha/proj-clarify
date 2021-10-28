import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme(style="ticks")
df = pd.read_csv("test_dataframe__responses_2021-10-25T_03-09-31Z.csv")
print(type(df))
g = sns.barplot(x="is_ambig", y="is_correct", data=df)
# h = sns.barplot(x="is_ambig", y="percent_certainty_of_correct", data=df)
plt.show()
