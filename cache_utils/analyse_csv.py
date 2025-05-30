# SPDX-FileCopyrightText: 2021 Guillaume DIDIER
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import argparse
import warnings
import time
import json
import sys
import os

import matplotlib.style as mplstyle
import matplotlib.pyplot as plt
import wquantiles as wq
import seaborn as sns
import pandas as pd
import numpy as np
#import tikzplotlib


t = time.time()
def print_timed(*args, **kwargs):
    print(f"[{round(time.time()-t, 1):>8}]", *args, **kwargs)


def dict_to_json(d):
    if isinstance(d, dict):
        return json.dumps(d)
    return d

# For cyber cobay sanity check :
# from gmpy2 import popcount
def popcount(x):
    return x.bit_count()

functions_i9_9900 = [
             0b1111111111010101110101010001000000,
             0b0110111110111010110001001000000000,
             0b1111111000011111110010110000000000]


def complex_hash(addr):
    r = 0
    for f in reversed(functions_i9_9900):
        r <<= 1
        r |= (popcount(f & addr) & 1)
    return r


def convert64(x):
    return np.int64(int(x, base=16))

def convert8(x):
    return np.array(int(x, base=16)).astype(np.int64)
    # return np.int8(int(x, base=16))


parser = argparse.ArgumentParser(
    prog=sys.argv[0],
)

parser.add_argument("path", help="Path to the experiment files")

parser.add_argument(
    "--no-plot",
    dest="no_plot",
    action="store_true",
    default=False,
    help="No visible plot (save figures to files)"
)

parser.add_argument(
    "--stats",
    dest="stats",
    action="store_true",
    default=False,
    help="Don't compute figures, just create .stats.csv file"
)

parser.add_argument(
    "--no-slice-remap",
    dest="slice_remap",
    action="store_false",
    default=True,
    help="Don't remap the slices"
)

args = parser.parse_args()

warnings.filterwarnings('ignore')
print_timed("warnings are filtered, enable them back if you are having some trouble")

img_dir = os.path.dirname(args.path)+"/figs/"
os.makedirs(img_dir, exist_ok=True)

if args.slice_remap:
    assert os.path.exists(args.path + ".slices.csv")
assert os.path.exists(args.path + ".cores.csv")
assert os.path.exists(args.path + "-results_lite.csv.bz2")

df = pd.read_csv(args.path + "-results_lite.csv.bz2",
        dtype={
            "main_core": np.int8,
            "helper_core": np.int8,
            # "address": int,
            # "hash": np.int8,
            "time": np.int16,
            "clflush_remote_hit": np.int32,
            "clflush_shared_hit": np.int32,
            "clflush_miss_f": np.int32,
            "clflush_local_hit_f": np.int32,
            "clflush_miss_n": np.int32,
            "clflush_local_hit_n": np.int32,
            "reload_miss": np.int32,
            "reload_remote_hit": np.int32,
            "reload_shared_hit": np.int32,
            "reload_local_hit": np.int32},
        converters={'address': convert64, 'hash': convert8},
        )

print_timed(f"Loaded columns : {list(df.keys())}")

sample_columns = [
    "clflush_remote_hit",
    "clflush_shared_hit",
    "clflush_miss_f",
    "clflush_local_hit_f",
    "clflush_miss_n",
    "clflush_local_hit_n",
    "reload_miss",
    "reload_remote_hit",
    "reload_shared_hit",
    "reload_local_hit",
]

sample_flush_columns = [
    "clflush_remote_hit",
    "clflush_shared_hit",
    "clflush_miss_f",
    "clflush_local_hit_f",
    "clflush_miss_n",
    "clflush_local_hit_n",
]

if args.slice_remap:
    slice_mapping = pd.read_csv(args.path + ".slices.csv")
core_mapping = pd.read_csv(args.path + ".cores.csv")

def remap_core(key):
    column = core_mapping.columns.get_loc(key)
    def remap(core):
        return core_mapping.iat[core, column]

    return remap


columns = [
    ("main_socket", "main_core", "socket"),
    ("main_core_fixed", "main_core", "core"),
    ("main_ht", "main_core", "hthread"),
    ("helper_socket", "helper_core", "socket"),
    ("helper_core_fixed", "helper_core", "core"),
    ("helper_ht", "helper_core", "hthread"),
]
if not args.stats:
    for (col, icol, key) in columns:
        df[col] = df[icol].apply(remap_core(key))
        print_timed(f"Column {col} added")


if args.slice_remap:
    slice_remap = lambda h: slice_mapping["slice_group"].iloc[h]
    df["slice_group"] = df["hash"].apply(slice_remap)
    print_timed("Column slice_group added")
else:
    df["slice_group"] = df["hash"]


def get_graphing_bounds():
    q10s = [wq.quantile(df["time"], df[col], 0.1) for col in sample_flush_columns if col in df]
    q90s = [wq.quantile(df["time"], df[col], 0.9) for col in sample_flush_columns if col in df]

    return int(((min(q10s) - 10) // 10) * 10), int(((max(q90s) + 19) // 10) * 10)


mplstyle.use("fast")

graph_lower, graph_upper = get_graphing_bounds()
print_timed(f"graphing between {graph_lower}, {graph_upper}")

def plot(filename, g=None):
    if args.no_plot:
        if g is not None:
            g.savefig(img_dir+filename)
        else:
            plt.savefig(img_dir+filename)
        print_timed(f"Saved {filename}")
        plt.close()
    plt.show()

def custom_hist(x_axis, *values, **kwargs):
    if "title" in kwargs:
        plt.title(kwargs["title"])
        del kwargs["title"]

    plt.xlim([graph_lower, graph_upper])

    for (i, yi) in enumerate(values):
        color = ["b", "r", "g", "y"][i%4]
        kwargs["color"] = color

        sns.histplot(
            x=x_axis,
            weights=yi,
            binwidth=5,
            bins=range(graph_lower, graph_upper),
            element="step",
            edgecolor=color,
            alpha=0.2,
            kde=False,
            **kwargs
        )

def show_specific_position(attacker, victim, slice):
    df_ax_vx_sx = df[(df["hash"] == slice) & (df["main_core"] == attacker) & (df["helper_core"] == victim)]

    custom_hist(df_ax_vx_sx["time"], df_ax_vx_sx["clflush_miss_n"], df_ax_vx_sx["clflush_remote_hit"], title=f"A{attacker} V{victim} S{slice}")
    #tikzplotlib.save("fig-hist-good-A{}V{}S{}.tex".format(attacker,victim,slice))#, axis_width=r'0.175\textwidth', axis_height=r'0.25\textwidth')
    plot("specific-a{}v{}s{}.png".format(attacker, victim, slice))


def show_grid(df, col, row, shown=["clflush_miss_n", "clflush_remote_hit", "clflush_local_hit_n", "clflush_shared_hit"]):
    # Color convention here :
    # Blue = miss
    # Red = Remote Hit
    # Green = Local Hit
    # Yellow = Shared Hit
    g = sns.FacetGrid(df, col=col, row=row, legend_out=True)
    g.map(custom_hist, "time", *shown)
    return g

def export_stats_csv():
    def compute_stat(x, key):
        """
        Compute the statistic for 1 helper core/main core/slice/column
        - median : default, not influenced by errors
        - average : better precision when observing floor steps in the results
        """
        # return wq.median(x["time"], x[key])
        return np.average(x[key], weights=x["time"])

    df_grouped = df.groupby(["main_core", "helper_core", "hash"])

    miss = df_grouped.apply(lambda x: compute_stat(x, "clflush_miss_n"))
    hit_remote = df_grouped.apply(lambda x: compute_stat(x, "clflush_remote_hit"))
    hit_local = df_grouped.apply(lambda x: compute_stat(x, "clflush_local_hit_n"))
    hit_shared = df_grouped.apply(lambda x: compute_stat(x, "clflush_shared_hit"))

    stats = pd.DataFrame({
        "main_core": miss.index.get_level_values(0),
        "helper_core": miss.index.get_level_values(1),
        "hash": miss.index.get_level_values(2),
        "clflush_miss_n": miss.values,
        "clflush_remote_hit": hit_remote.values,
        "clflush_local_hit_n": hit_local.values,
        "clflush_shared_hit": hit_shared.values
    })

    stats.to_csv(args.path + ".stats.csv", index=False)


df.loc[:, ("hash",)] = df["hash"].apply(dict_to_json)

if not args.stats:
    custom_hist(df["time"], df["clflush_miss_n"], df["clflush_remote_hit"], title="miss v. hit")
    plot("miss_v_hit.png")

    custom_hist(df["time"], df["clflush_miss_n"], df["clflush_remote_hit"], df["clflush_local_hit_n"], df["clflush_shared_hit"], title="miss v. hit")
    plot("miss_vall_hits.png")

    show_specific_position(0, 2, 0)

    df_main_core_0 = df[df["main_core"] == 0]
    df_main_core_0.loc[:, ("hash",)] = df["hash"].apply(dict_to_json)

    g = show_grid(df_main_core_0, "helper_core", "hash", shown=["clflush_miss_n", "clflush_remote_hit"])
    plot("grid_helper_dual.png", g=g)

    g = show_grid(df, "main_core", "hash", shown=["clflush_miss_n", "clflush_remote_hit"])
    plot("grid_main_dual.png", g=g)

    g = show_grid(df, "main_core", "helper_core", shown=["clflush_miss_n", "clflush_remote_hit"])
    plot("grid_main_helper_dual.png", g=g)


if not os.path.exists(args.path + ".stats.csv") or args.stats:
    export_stats_csv()
else:
    print_timed("Skipping .stats.csv export")
