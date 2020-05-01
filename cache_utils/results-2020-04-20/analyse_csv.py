import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sys import exit
import wquantiles as wq

df = pd.read_csv("./results_lite.csv")
print(df.head())


g = sns.FacetGrid(df, col="core", row="hash", legend_out=True)


def custom_hist(x, y1, y2, **kwargs):
    sns.distplot(x,range(200, 280), hist_kws={"weights": y1, "histtype":"step"}, kde=False, **kwargs)
    kwargs["color"] = "r"
    sns.distplot(x, range(200, 280), hist_kws={"weights": y2, "histtype":"step"}, kde=False, **kwargs)

g.map(custom_hist, "time", "clflush_hit", "clflush_miss")
# g.map(sns.distplot, "time", hist_kws={"weights": df["clflush_hit"]}, kde=False)

plt.figure()

def stat(x, key):
    return wq.median(x["time"], x[key])


miss = df.groupby(["core", "hash"]).apply(stat, "clflush_miss")
stats = miss.reset_index()
stats.columns = ["Core", "Hash", "Miss"]
hit = df.groupby(["core", "hash"]).apply(stat, "clflush_hit")
stats["Hit"] = hit.values


print(stats.to_string())

g = sns.FacetGrid(stats, row="Core")

g.map(sns.distplot, 'Miss', bins=range(200, 280), color="r")
g.map(sns.distplot, 'Hit', bins=range(200, 280))
plt.show()

#stats["clflush_miss_med"] = stats[[0]].apply(lambda x: x["miss_med"])
#stats["clflush_hit_med"] = stats[[0]].apply(lambda x: x["hit_med"])
#del df[[0]]
#print(hit.to_string(), miss.to_string())

# test = pd.DataFrame({"value" : [0, 5], "weight": [5, 1]})
# plt.figure()
# sns.distplot(test["value"], hist_kws={"weights": test["weight"]}, kde=False)

exit(0)
