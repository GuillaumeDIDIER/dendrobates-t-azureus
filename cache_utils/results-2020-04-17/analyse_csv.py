import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sys import exit

columns = ["Addr", "Hash"]
core_number = 8  # FIXME
for i in range(0, core_number):
    for stat in ["Min", "Med", "Max"]:
        for op in ["Hit", "Miss"]:
            columns.append(op + str(i) + stat)
columns.append("Hmm")
df = pd.read_csv("./results_lite.csv")
print(df.head())


g = sns.FacetGrid(df, col="core", row="hash", legend_out=True)


def custom_hist(x,y, **kwargs):
    sns.distplot(x, range(100,150), hist_kws={"weights": y, "histtype":"step"}, kde=False, **kwargs)

g.map(custom_hist, "time", "clflush_hit")
# g.map(sns.distplot, "time", hist_kws={"weights": df["clflush_hit"]}, kde=False)
plt.show()

# test = pd.DataFrame({"value" : [0, 5], "weight": [5, 1]})
# plt.figure()
# sns.distplot(test["value"], hist_kws={"weights": test["weight"]}, kde=False)

exit(0)

selected_columns = columns[:-1]
df = df[selected_columns]
print(df.head())

median_columns = list(filter(lambda s: s.endswith("Med"), columns))

median_hits_col = list(filter(lambda s: s.startswith("Hit"), median_columns))
median_miss_col = list(filter(lambda s: s.startswith("Miss"), median_columns))

print(list(median_columns))
print(list(median_hits_col), list(median_miss_col))

hashes = df["Hash"].drop_duplicates()
print(hashes)

# def distrib(x, y, **kwargs):
#    sns.distplot()

separate_core_df = df.melt(id_vars=["Addr", "Hash"], value_vars=median_hits_col)

g = sns.FacetGrid(separate_core_df, row="variable")
g.map(sns.distplot, "value")
plt.figure()

separate_core_df = df.melt(id_vars=["Addr", "Hash"], value_vars=median_miss_col)
g = sns.FacetGrid(separate_core_df, row="variable")
g.map(sns.distplot, "value", hist_kws={"range": (75, 115)})

plt.show()

# sns.distplot(df["values"], hist_kws={"weights": df["count"]})
