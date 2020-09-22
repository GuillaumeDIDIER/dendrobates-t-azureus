import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sys import exit
import wquantiles as wq
import numpy as np

from functools import partial

import sys

df = pd.read_csv(sys.argv[1], header=1, names=["Core", "Addr", "Hash", "Time", "ClflushHit", "ClflushMiss"], dtype={"Core": int, "Time": int, "ClflushHit": int, "ClflushMiss": int},
        converters={'Addr': partial(int, base=16), 'Hash': partial(int, base=16)},
        usecols=["Core", "Addr", "Hash", "Time", "ClflushHit", "ClflushMiss"]
        )

print(df.columns)
#df["Hash"] = df["Addr"].apply(lambda x: (x >> 15)&0x3)

print(df.head())

print(df["Hash"].unique())

g = sns.FacetGrid(df, col="Core", row="Hash", legend_out=True)


def custom_hist(x, y1, y2, **kwargs):
    sns.distplot(x, range(100, 400), hist_kws={"weights": y1, "histtype":"step"}, kde=False, **kwargs)
    kwargs["color"] = "r"
    sns.distplot(x, range(100, 400), hist_kws={"weights": y2, "histtype":"step"}, kde=False, **kwargs)

g.map(custom_hist, "Time", "ClflushHit", "ClflushMiss")
# g.map(sns.distplot, "time", hist_kws={"weights": df["clflush_hit"]}, kde=False)

#plt.figure()

plt.show()
exit(0)

def stat(x, key):
    return wq.median(x["Time"], x[key])


miss = df.groupby(["Core", "Hash"]).apply(stat, "ClflushMiss")
stats = miss.reset_index()
stats.columns = ["Core", "Hash", "Miss"]
hit = df.groupby(["Core", "Hash"]).apply(stat, "ClflushHit")
stats["Hit"] = hit.values


print(stats.to_string())

g = sns.FacetGrid(stats, row="Core")

g.map(sns.distplot, 'Miss', bins=range(100, 480), color="r")
g.map(sns.distplot, 'Hit', bins=range(100, 480))
plt.show()

#stats["clflush_miss_med"] = stats[[0]].apply(lambda x: x["miss_med"])
#stats["clflush_hit_med"] = stats[[0]].apply(lambda x: x["hit_med"])
#del df[[0]]
#print(hit.to_string(), miss.to_string())

# test = pd.DataFrame({"value" : [0, 5], "weight": [5, 1]})
# plt.figure()
# sns.distplot(test["value"], hist_kws={"weights": test["weight"]}, kde=False)

exit(0)
