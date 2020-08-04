import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sys import exit
import numpy as np
from scipy import optimize
import sys

# TODO
# sys.argv[1] should be the root
# with root-result_lite.csv.bz2 the result
# and .stats.csv
# root.slices a slice mapping - done
# root.cores a core + socket mapping - done -> move to analyse csv ?
#
# Facet plot with actual dot cloud + plot the linear regression
# each row is a slice
# each row is an origin core
# each column a helper core if applicable


stats = pd.read_csv(sys.argv[1] + ".stats.csv",
                    dtype={
                        "main_core": np.int8,
                        "helper_core": np.int8,
                        # "address": int,
                        "hash": np.int8,
                        # "time": np.int16,
                        "clflush_remote_hit": np.float64,
                        "clflush_shared_hit": np.float64,
                        # "clflush_miss_f": np.int32,
                        # "clflush_local_hit_f": np.int32,
                        "clflush_miss_n": np.float64,
                        "clflush_local_hit_n": np.float64,
                        # "reload_miss": np.int32,
                        # "reload_remote_hit": np.int32,
                        # "reload_shared_hit": np.int32,
                        # "reload_local_hit": np.int32
                    }
                    )

slice_mapping = pd.read_csv(sys.argv[1] + ".slices.csv")
core_mapping = pd.read_csv(sys.argv[1] + ".cores.csv")

print(core_mapping.to_string())
print(slice_mapping.to_string())

print("core {} is mapped to '{}'".format(4, repr(core_mapping.iloc[4])))

min_time_miss = stats["clflush_miss_n"].min()
max_time_miss = stats["clflush_miss_n"].max()


def remap_core(key):
    def remap(core):
        remapped = core_mapping.iloc[core]
        return remapped[key]

    return remap


stats["main_socket"] = stats["main_core"].apply(remap_core("socket"))
stats["main_core_fixed"] = stats["main_core"].apply(remap_core("core"))
stats["main_ht"] = stats["main_core"].apply(remap_core("hthread"))
stats["helper_socket"] = stats["helper_core"].apply(remap_core("socket"))
stats["helper_core_fixed"] = stats["helper_core"].apply(remap_core("core"))
stats["helper_ht"] = stats["helper_core"].apply(remap_core("hthread"))

# slice_mapping = {3: 0, 1: 1, 2: 2, 0: 3}

stats["slice_group"] = stats["hash"].apply(lambda h: slice_mapping.iloc[h])

graph_lower_miss = int((min_time_miss // 10) * 10)
graph_upper_miss = int(((max_time_miss + 9) // 10) * 10)

print("Graphing from {} to {}".format(graph_lower_miss, graph_upper_miss))

g = sns.FacetGrid(stats, row="main_core_fixed")

g.map(sns.scatterplot, 'slice_group', 'clflush_miss_n', color="b")
g.map(sns.scatterplot, 'slice_group', 'clflush_local_hit_n', color="g")

g2 = sns.FacetGrid(stats, row="main_core_fixed", col="slice_group")
g2.map(sns.scatterplot, 'helper_core_fixed', 'clflush_remote_hit', color="r")

g3 = sns.FacetGrid(stats, row="main_core_fixed", col="slice_group")
g3.map(sns.scatterplot, 'helper_core_fixed', 'clflush_shared_hit', color="y")

print(stats.head())


def miss_topology(x, C, h):
    main_core = x["main_core_fixed"]
    slice_group = x["slice_group"]
    return C + h * abs(main_core - slice_group) + h * abs(slice_group + 1)


res = optimize.curve_fit(miss_topology, stats[["main_core_fixed", "slice_group"]], stats["clflush_miss_n"])
print(res)


def local_hit_topology(x, C, h):
    main_core = x["main_core_fixed"]
    slice_group = x["slice_group"]
    return C + h * abs(main_core - slice_group)



def remote_hit_topology_1(x, C, h):
    main_core = x["main_core_fixed"]
    slice_group = x["slice_group"]
    helper_core = x["helper_core_fixed"]
    return C + h * abs(main_core - slice_group) + h * abs(slice_group - helper_core)


def remote_hit_topology_2(x, C, h):
    main_core = x["main_core_fixed"]
    slice_group = x["slice_group"]
    helper_core = x["helper_core_fixed"]
    return C + h * abs(main_core - slice_group) + h * abs(slice_group - helper_core) + h * abs(helper_core - main_core)


def shared_hit_topology_1(x, C, h):
    main_core = x["main_core_fixed"]
    slice_group = x["slice_group"]
    helper_core = x["helper_core_fixed"]
    return C + h * abs(main_core - slice_group) + h * max(abs(slice_group - main_core), abs(slice_group - helper_core))


# more ideas needed

plt.show()
