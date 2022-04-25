import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
sns.set_theme(style="ticks")
df = pd.read_csv("test_dataframe__responses_2021-10-25T_03-09-31Z.csv")
print(type(df))
g = sns.barplot(x="is_ambig", y="is_correct", data=df)
# h = sns.barplot(x="is_ambig", y="percent_certainty_of_correct", data=df)
plt.show()"""


x = torch.rand(64, 121, 800)
print(x.size())
batch_size = x.shape[0]
c_len = x.shape[1]

hidden_size = 200
proj = nn.Linear(hidden_size*4, hidden_size)
x = x.view(-1, x.shape[2])
print(x.size())
x = proj(x)
print(x.size())
x = x.view(batch_size, c_len, -1)
print(x.size())