# SPDX-FileCopyrightText: 2021 Guillaume DIDIER
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import tikzplotlib
import wquantiles as wq
import numpy as np
import argparse

import sys
import os

import json
import warnings

warnings.filterwarnings('ignore')
print("warnings are filtered, enable them back if you are having some trouble")

sns.set_theme()

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

args = parser.parse_args()

img_dir = os.path.dirname(args.path)+"/figs/"
os.makedirs(img_dir, exist_ok=True)

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

print(f"Loaded columns : {list(df.keys())}")

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


slice_mapping = pd.read_csv(args.path + ".slices.csv")
core_mapping = pd.read_csv(args.path + ".cores.csv")

def remap_core(key):
    def remap(core):
        remapped = core_mapping.iloc[core]
        return remapped[key]

    return remap


df["main_socket"] = df["main_core"].apply(remap_core("socket"))
df["main_core_fixed"] = df["main_core"].apply(remap_core("core"))
df["main_ht"] = df["main_core"].apply(remap_core("hthread"))
df["helper_socket"] = df["helper_core"].apply(remap_core("socket"))
df["helper_core_fixed"] = df["helper_core"].apply(remap_core("core"))
df["helper_ht"] = df["helper_core"].apply(remap_core("hthread"))


slice_remap = lambda h: slice_mapping["slice_group"].iloc[h]
df["slice_group"] = df["hash"].apply(slice_remap)


def get_graphing_bounds():
    q10s = [wq.quantile(df["time"], df[col], 0.1) for col in sample_flush_columns if col in df]
    q90s = [wq.quantile(df["time"], df[col], 0.9) for col in sample_flush_columns if col in df]

    return int(((min(q10s) - 10) // 10) * 10), int(((max(q90s) + 19) // 10) * 10)


graph_lower, graph_upper = get_graphing_bounds()
print("graphing between {}, {}".format(graph_lower, graph_upper))


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
    if args.no_plot:
        plt.savefig(img_dir+"specific-a{}v{}s{}.png".format(attacker, victim, slice))
        plt.close()
    else:
        plt.show()

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
    def stat(x, key):
        return wq.median(x["time"], x[key])
    df_grouped = df.groupby(["main_core", "helper_core", "hash"])

    miss = df_grouped.apply(stat, "clflush_miss_n")
    hit_remote = df_grouped.apply(stat, "clflush_remote_hit")
    hit_local = df_grouped.apply(stat, "clflush_local_hit_n")
    hit_shared = df_grouped.apply(stat, "clflush_shared_hit")

    stats = miss.reset_index()
    stats.columns = ["main_core", "helper_core", "hash", "clflush_miss_n"]
    stats["clflush_remote_hit"] = hit_remote.values
    stats["clflush_local_hit_n"] = hit_local.values
    stats["clflush_shared_hit"] = hit_shared.values

    stats.to_csv(args.path + ".stats.csv", index=False)


df.loc[:, ("hash",)] = df["hash"].apply(dict_to_json)

if not args.stats:
    custom_hist(df["time"], df["clflush_miss_n"], df["clflush_remote_hit"], title="miss v. hit")
    if args.no_plot:
        plt.savefig(img_dir+"miss_v_hit.png")
        plt.close()
    else:
        plt.show()


    show_specific_position(0, 2, 0)

    df_main_core_0 = df[df["main_core"] == 0]
    df_main_core_0.loc[:, ("hash",)] = df["hash"].apply(dict_to_json)

    g = show_grid(df_main_core_0, "helper_core", "hash")

    if args.no_plot:
        g.savefig(img_dir+"helper_grid.png")
        plt.close()
    else:
        plt.show()

    g = show_grid(df, "main_core", "hash")

    if args.no_plot:
        g.savefig(img_dir+"main_grid.png")
        plt.close()
    else:
        plt.show()


if not os.path.exists(args.path + ".stats.csv") or args.stats:
    export_stats_csv()
else:
    print("Skipping .stats.csv export")
