import os
import sys
import argparse
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    prog=sys.argv[0],
)
parser.add_argument("path", help="Path to the experiment files")
args = parser.parse_args()


assert os.path.exists(args.path + ".stats.csv")
assert os.path.exists(args.path + ".cores.csv")

stats = pd.read_csv(
    args.path + ".stats.csv",
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
    },
)

core_mapping = pd.read_csv(args.path + ".cores.csv")

cores = list(stats["main_core"].unique())
slices = list(stats["hash"].unique())



def slice_reorder(df, fst_slice, params=None):
    if params is None:
        params = ["clflush_miss_n", "clflush_remote_hit"]

    keys = slices.copy()
    sliced_df = {
        i : df[(df["hash"] == i)] for i in keys
    }

    def distance(df1, df2):
        dist = 0
        for param in params:
            for core, helper in itertools.product(cores, cores):
                med1 = df1[(df1["main_core"] == core) & (df1["helper_core"] == helper)][param].median()
                med2 = df2[(df2["main_core"] == core) & (df2["helper_core"] == helper)][param].median()
                dist += (med1 - med2)**2

        return dist

    def find_nearest(slice):
        distances = { i : distance(sliced_df[slice], sliced_df[i]) for i in keys}
        nearest = sorted(keys, key=lambda x: distances[x])[0]
        return nearest, distances[nearest]

    new_reorder = [fst_slice]
    total_dist = 0
    keys.remove(fst_slice)
    for i in range(len(slices)-1):
        next, dist = find_nearest(new_reorder[-1])
        total_dist += dist
        new_reorder.append(next)
        keys.remove(next)

    print("slice_group")
    print("\n".join([
        str(new_reorder.index(i)) for i in range(len(slices))
    ]))

    return total_dist


def core_reorder(df, fst_core, params=None, position="both", lcores=None):
    """
    Find a core ordering that minimizes the distance from each to the adjacents
    - df : panda dataframe
    - fst_core : first core to use in the ordering
    - params : columns to use (clflush_miss_n, clflush_remote_hit, ...)
    - position : both, helper, or main
    - lcores : subset of cores to reorder (eg 1 socket only)
    """
    from_main = False
    from_helper = False
    if position == "both":
        from_main = True
        from_helper = True
    elif position == "helper":
        from_helper = True
    elif position == "main":
        from_main = True

    if params is None:
        params = ["clflush_miss_n", "clflush_remote_hit"]


    if lcores is None:
        lcores = cores.copy()

    lcores.sort()
    keys = lcores.copy()
    print(keys)
    main_sliced_df = {
        i : df[(df["main_core"] == i)] for i in keys
    }
    helper_sliced_df = {
        i : df[(df["helper_core"] == i)] for i in keys
    }

    def distance(df1, df2, is_main=True):
        dist = 0
        for param in params:
            for hash, core in itertools.product(slices, lcores):
                col = "main_core"
                if is_main:
                    col ="helper_core"

                med1 = df1[(df1["hash"] == hash) & (df1[col] == core)][param].median()
                med2 = df2[(df2["hash"] == hash) & (df2[col] == core)][param].median()
                dist += (med1 - med2)**2

        return dist

    def find_nearest(slice):
        distances = { i : 0 for i in keys}
        for i in distances:
            if from_main:
                distances[i] += distance(main_sliced_df[fst_core], main_sliced_df[i], is_main=True)
            if from_helper:
                distances[i] += distance(helper_sliced_df[fst_core], helper_sliced_df[i], is_main=False)

        nearest = sorted(keys, key=lambda x: distances[x])[0]
        return nearest, distances[nearest]

    new_reorder = [fst_core]
    total_dist = 0
    keys.remove(fst_core)
    for i in range(len(lcores)-1):
        next, dist = find_nearest(new_reorder[-1])
        total_dist += dist
        new_reorder.append(next)
        keys.remove(next)

    print("core")
    print("\n".join([
        str(lcores[new_reorder.index(i)]) for i in lcores
    ]))
    return total_dist


for hash in slices:
    res = slice_reorder(stats, hash)
    print(f"hash: {hash}, total dist: {res}")

half = len(cores)/2
for core in cores:
    res = core_reorder(stats, core, lcores=[k for k in cores if (k//half == core//half)])
    print(f"core: {core}, total dist: {res}")
