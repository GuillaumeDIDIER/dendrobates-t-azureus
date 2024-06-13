# SPDX-FileCopyrightText: 2021 Guillaume DIDIER
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
print("warnings are filtered, enable them back if you are having some trouble")

# TODO
# args.path should be the root
# with root-result_lite.csv.bz2 the result
# and .stats.csv
# root.slices a slice mapping - done
# root.cores a core + socket mapping - done -> move to analyse csv ?
#
# Facet plot with actual dot cloud + plot the linear regression
# each row is a slice
# each row is an origin core
# each column a helper core if applicable

parser = argparse.ArgumentParser(
    prog=sys.argv[0],
)

parser.add_argument("path", help="Path to the experiment files")

parser.add_argument(
    "--no-plot",
    dest="no_plot",
    action="store_true",
    default=False,
    help="No visible plot (save figures to files)",
)

parser.add_argument(
    "--rslice",
    dest="rslice",
    action="store_true",
    default=False,
    help="Create slice{} directories with segmented grid",
)

args = parser.parse_args()

img_dir = os.path.dirname(args.path) + "/figs/"
os.makedirs(img_dir, exist_ok=True)

assert os.path.exists(args.path + ".stats.csv")
assert os.path.exists(args.path + ".slices.csv")
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

slice_mapping = pd.read_csv(args.path + ".slices.csv")
core_mapping = pd.read_csv(args.path + ".cores.csv")

# print("core mapping:\n", core_mapping.to_string())
# print("slice mapping:\n", slice_mapping.to_string())

# print("core {} is mapped to '{}'".format(4, repr(core_mapping.iloc[4])))

min_time_miss = stats["clflush_miss_n"].min()
max_time_miss = stats["clflush_miss_n"].max()


def remap_core(key):
    def remap(core):
        remapped = core_mapping.iloc[core]
        return remapped[key]

    return remap


def plot(filename, g=None):
    if args.no_plot:
        if g is not None:
            g.savefig(img_dir + filename)
        else:
            plt.savefig(img_dir + filename)
        # tikzplotlib.save(
        #   img_dir+filename+".tex",
        #   axis_width=r'0.175\textwidth',
        #   axis_height=r'0.25\textwidth'
        # )
        print(filename, "saved")
        plt.close()
    plt.show()


stats["main_socket"] = stats["main_core"].apply(remap_core("socket"))
stats["main_core_fixed"] = stats["main_core"].apply(remap_core("core"))
stats["main_ht"] = stats["main_core"].apply(remap_core("hthread"))
stats["helper_socket"] = stats["helper_core"].apply(remap_core("socket"))
stats["helper_core_fixed"] = stats["helper_core"].apply(remap_core("core"))
stats["helper_ht"] = stats["helper_core"].apply(remap_core("hthread"))

# slice_mapping = {3: 0, 1: 1, 2: 2, 0: 3}

stats["slice_group"] = stats["hash"].apply(
    lambda h: slice_mapping["slice_group"].iloc[h]
)

graph_lower_miss = int((min_time_miss // 10) * 10)
graph_upper_miss = int(((max_time_miss + 9) // 10) * 10)

# print("Graphing from {} to {}".format(graph_lower_miss, graph_upper_miss))


# also explains remote
# shared needs some thinking as there is something weird happening there.

#
# M 0 1 2 3 4 5 6 7
#


# print(stats.head())

num_core = len(stats["main_core_fixed"].unique())
# print("Found {}".format(num_core))


def miss_topology(main_core_fixed, slice_group, C, h):
    return C + h * abs(main_core_fixed - slice_group) + h * abs(slice_group + 1)


def miss_topology_df(x, C, h):
    func = lambda x, C, h: miss_topology(x["main_core_fixed"], x["slice_group"], C, h)
    return x.apply(func, args=(C, h), axis=1)


memory = -1
gpu_if_any = num_core


def exclusive_hit_topology_gpu(main_core, slice_group, helper_core, C, h1, h2):
    round_trip = gpu_if_any - memory

    if slice_group <= num_core / 2:
        # send message towards higher cores first
        if helper_core < slice_group:
            r = (
                C
                + h1 * abs(main_core - slice_group)
                + h2 * abs(round_trip - (helper_core - memory))
            )
        else:
            r = (
                C
                + h1 * abs(main_core - slice_group)
                + h2 * abs(helper_core - slice_group)
            )
    else:
        # send message toward lower cores first
        if helper_core > slice_group:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(helper_core - memory)
        else:
            r = (
                C
                + h1 * abs(main_core - slice_group)
                + h2 * abs(helper_core - slice_group)
            )
    return r


def exclusive_hit_topology_gpu_df(x, C, h1, h2):
    def func(x, C, h1, h2):
        return exclusive_hit_topology_gpu(
            x["main_core_fixed"], x["slice_group"], x["helper_core_fixed"], C, h1, h2
        )

    return x.apply(func, args=(C, h1, h2), axis=1)


def exclusive_hit_topology_gpu2(main_core, slice_group, helper_core, C, h1, h2):
    round_trip = gpu_if_any + 1 - memory

    if slice_group <= num_core / 2:
        # send message towards higher cores first
        if helper_core < slice_group:
            r = (
                C
                + h1 * abs(main_core - slice_group)
                + h2 * abs(round_trip - (helper_core - memory))
            )
        else:
            r = (
                C
                + h1 * abs(main_core - slice_group)
                + h2 * abs(helper_core - slice_group)
            )
    else:
        # send message toward lower cores first
        if helper_core > slice_group:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(helper_core - memory)
        else:
            r = (
                C
                + h1 * abs(main_core - slice_group)
                + h2 * abs(helper_core - slice_group)
            )
    return r


def exclusive_hit_topology_gpu2_df(x, C, h1, h2):
    def func(x, C, h1, h2):
        return exclusive_hit_topology_gpu2(
            x["main_core_fixed"], x["slice_group"], x["helper_core_fixed"], C, h1, h2
        )

    return x.apply(func, args=(C, h1, h2), axis=1)


# unlikely
def exclusive_hit_topology_nogpu(main_core, slice_group, helper_core, C, h1, h2):
    round_trip = (num_core - 1) - memory

    if slice_group <= num_core / 2:
        # send message towards higher cores first
        if helper_core < slice_group:
            r = (
                C
                + h1 * abs(main_core - slice_group)
                + h2 * abs(round_trip - (helper_core - memory))
            )
        else:
            r = (
                C
                + h1 * abs(main_core - slice_group)
                + h2 * abs(helper_core - slice_group)
            )
    else:
        # send message toward lower cores first
        if helper_core > slice_group:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(helper_core - memory)
        else:
            r = (
                C
                + h1 * abs(main_core - slice_group)
                + h2 * abs(helper_core - slice_group)
            )
    return r


def exclusive_hit_topology_nogpu_df(x, C, h1, h2):
    def func(x, C, h1, h2):
        return exclusive_hit_topology_nogpu(
            x["main_core_fixed"], x["slice_group"], x["helper_core_fixed"], C, h1, h2
        )

    return x.apply(func, args=(C, h1, h2), axis=1)


def remote_hit_topology_2(x, C, h):
    main_core = x["main_core_fixed"]
    slice_group = x["slice_group"]
    helper_core = x["helper_core_fixed"]
    return (
        C
        + h * abs(main_core - slice_group)
        + h * abs(slice_group - helper_core)
        + h * abs(helper_core - main_core)
    )


def shared_hit_topology_1(x, C, h):
    main_core = x["main_core_fixed"]
    slice_group = x["slice_group"]
    helper_core = x["helper_core_fixed"]
    return (
        C
        + h * abs(main_core - slice_group)
        + h * max(abs(slice_group - main_core), abs(slice_group - helper_core))
    )


def do_predictions(df):
    res_miss = optimize.curve_fit(
        miss_topology_df, df[["main_core_fixed", "slice_group"]], df["clflush_miss_n"]
    )
    # print("Miss topology:")
    # print(res_miss)

    res_gpu = optimize.curve_fit(
        exclusive_hit_topology_gpu_df,
        df[["main_core_fixed", "slice_group", "helper_core_fixed"]],
        df["clflush_remote_hit"],
    )
    # print("Exclusive hit topology (GPU):")
    # print(res_gpu)

    # res_gpu2 = optimize.curve_fit(
    #     exclusive_hit_topology_gpu2_df,
    #     df[["main_core_fixed", "slice_group", "helper_core_fixed"]],
    #     df["clflush_remote_hit"]
    # )
    # print("Exclusive hit topology (GPU2):")
    # print(res_gpu2)

    # res_no_gpu = optimize.curve_fit(
    #     exclusive_hit_topology_nogpu_df,
    #     df[["main_core_fixed", "slice_group", "helper_core_fixed"]],
    #     df["clflush_remote_hit"]
    # )
    # print("Exclusive hit topology (No GPU):")
    # print(res_no_gpu)

    df["predicted_miss"] = miss_topology_df(df, *(res_miss[0]))

    # df["predicted_remote_hit_no_gpu"] = exclusive_hit_topology_nogpu_df(df, *(res_no_gpu[0]))
    df["predicted_remote_hit_gpu"] = exclusive_hit_topology_gpu_df(df, *(res_gpu[0]))
    # df["predicted_remote_hit_gpu2"] = exclusive_hit_topology_gpu_df(df, *(res_gpu2[0]))

    df_A0 = df[df["main_core_fixed"] == 0]
    figure_A0 = sns.FacetGrid(df_A0, col="slice_group")
    figure_A0.map(sns.scatterplot, "helper_core_fixed", "clflush_remote_hit", color="r")
    figure_A0.map(
        sns.lineplot, "helper_core_fixed", "predicted_remote_hit_gpu", color="r"
    )
    figure_A0.set_titles(col_template="$S$ = {col_name}")
    plot("medians_remote_hit.png")

    g2 = sns.FacetGrid(df, row="main_core_fixed", col="slice_group")
    g2.map(sns.scatterplot, "helper_core_fixed", "clflush_remote_hit", color="r")
    g2.map(sns.lineplot, "helper_core_fixed", "predicted_remote_hit_gpu", color="r")
    # g2.map(sns.lineplot, 'helper_core_fixed', 'predicted_remote_hit_gpu2', color="g")
    # g2.map(sns.lineplot, 'helper_core_fixed', 'predicted_remote_hit_no_gpu', color="g")
    plot("medians_remote_hit_grid.png", g=g2)


def rslice():
    for core in stats["main_core_fixed"].unique():
        os.makedirs(img_dir + f"slices{core}", exist_ok=True)
        for slice_ in stats["slice_group"].unique():
            df = stats[
                (stats["slice_group"] == slice_) & (stats["main_core_fixed"] == core)
            ]
            fig = sns.scatterplot(
                df, x="helper_core_fixed", y="clflush_remote_hit", color="r"
            )
            fig.set(title=f"main_core={core} slice={slice_}")
            plt.savefig(img_dir + f"slices{core}/" + str(slice_) + ".png")
            plt.close()


def facet_grid(
        df, row, col, third,
        draw_fn=sns.scatterplot,
        shown=[
            "clflush_shared_hit",
            "clflush_remote_hit",
            "clflush_local_hit_n",
            "clflush_miss_n",
        ],
        colors=["y", "r", "g", "b"],
        title=None,
    ):
    """
    Creates a facet grid showing all points
    """
    grid = sns.FacetGrid(df, row=row, col=col)

    for i, el in enumerate(shown):
        grid.map(draw_fn, third, el, color=colors[i % len(colors)])

    if title is not None:
        plot(title, g=grid)
    return grid


def all_facets(df, id_, *args, **kwargs):
    """
    df : panda dataframe
    id_: the str to append to filenames
    """

    facet_grid(
        df, "main_core_fixed", "helper_core_fixed", "slice_group",
        title=f"medians_facet_{id_}s.png", *args, **kwargs
    )
    facet_grid(
        df, "helper_core_fixed", "slice_group", "main_core_fixed",
        title=f"medians_facet_{id}c.png", *args, **kwargs
    )
    facet_grid(
        df, "slice_group", "main_core_fixed", "helper_core_fixed",
        title=f"medians_facet_{id}h.png", *args, **kwargs
    )


if args.rslice:
    rslice()

do_predictions(stats)
all_facets(stats, "")

for main in (0, 1):
    for helper in (0, 1):
        print(f"Doing all facets {main}x{helper}")
        filtered_df = stats[
            (stats["main_core_fixed"] // (num_core / 2) == main)
            & (stats["helper_core_fixed"] // (num_core / 2) == helper)
        ]
        all_facets(filtered_df, f"m{main}h{helper}_")
