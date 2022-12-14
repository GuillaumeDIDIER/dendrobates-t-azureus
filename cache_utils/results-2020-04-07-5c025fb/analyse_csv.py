import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

columns = ["Addr", "Hash"]
core_number = 8 # FIXME
for i in range(0, core_number):
    for stat in ["Min", "Med", "Max"]:
        for op in ["Hit", "Miss"]:
            columns.append(op + str(i) + stat)
columns.append("Hmm")
df = pd.read_csv("citron-vert/combined.csv", header=0, names=columns)
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

#def distrib(x, y, **kwargs):
#    sns.distplot()

separate_core_df = df.melt(id_vars=["Addr", "Hash"], value_vars=median_hits_col)

g = sns.FacetGrid(separate_core_df, row="variable")
g.map(sns.distplot, "value")
plt.figure()

separate_core_df = df.melt(id_vars=["Addr", "Hash"], value_vars=median_miss_col)
g = sns.FacetGrid(separate_core_df, row="variable")
g.map(sns.distplot, "value", hist_kws={"range":(75,115)})

plt.show()

#sns.distplot(df["values"], hist_kws={"weights": df["count"]})


