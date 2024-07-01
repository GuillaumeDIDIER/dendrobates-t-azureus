# SPDX-FileCopyrightText: 2021 Guillaume DIDIER
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT
import os
import sys
import argparse
import warnings
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

mplstyle.use("fast")

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

parser.add_argument(
    "--no-slice-remap",
    dest="slice_remap",
    action="store_false",
    default=True,
    help="Don't remap the slices"
)

args = parser.parse_args()

img_dir = os.path.dirname(args.path) + "/figs/"
os.makedirs(img_dir, exist_ok=True)

assert os.path.exists(args.path + ".stats.csv")
assert os.path.exists(args.path + ".cores.csv")
if args.slice_remap:
    assert os.path.exists(args.path + ".slices.csv")

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

if args.slice_remap:
    slice_mapping = pd.read_csv(args.path + ".slices.csv")
core_mapping = pd.read_csv(args.path + ".cores.csv")

# print("core mapping:\n", core_mapping.to_string())
# print("slice mapping:\n", slice_mapping.to_string())

# print("core {} is mapped to '{}'".format(4, repr(core_mapping.iloc[4])))

min_time_miss = stats["clflush_miss_n"].min()
max_time_miss = stats["clflush_miss_n"].max()


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
for (col, icol, key) in columns:
    stats[col] = stats[icol].apply(remap_core(key))

#! Remove points where helper_core == main_core but main_ht != helper_ht
stats = stats[
    (stats["main_ht"] == stats["helper_ht"])
    | (stats["main_core_fixed"] != stats["helper_core_fixed"])
]
# slice_mapping = {3: 0, 1: 1, 2: 2, 0: 3}

if args.slice_remap:
    stats["slice_group"] = stats["hash"].apply(
        lambda h: slice_mapping["slice_group"].iloc[h]
    )
else:
    stats["slice_group"] = stats["hash"]

graph_lower_miss = int((min_time_miss // 10) * 10)
graph_upper_miss = int(((max_time_miss + 9) // 10) * 10)

# print("Graphing from {} to {}".format(graph_lower_miss, graph_upper_miss))


# also explains remote
# shared needs some thinking as there is something weird happening there.

#
# M 0 1 2 3 4 5 6 7
#


# print(stats.head())

num_core = len(stats["main_core_fixed"].unique())/2
# print("Found {}".format(num_core))

def ring_distance(x0, x1):
    """
    return (a, b) where `a` is the core distance and `b` the larger "ring step"
    """
    dist = abs(x0-x1)
    if x0 // (num_core/2) != x1 // (num_core/2):
        return min(num_core-1-dist, dist-1), 1
    else:
        return dist, 0

def slice_msg_distance(x1, x0):
    """
    Si l'expéditeur est à l'extrémité d'une des lignes, il envoie toujours dans le même sens
    (vers toute sa ligne d'abord), sinon, il prend le chemin le plus court
    """
    dist = abs(x0-x1)
    if x0 == 3:
        dist = (x0-x1+8)%8
    elif x0 == 4:
        dist = (x1-x0+8)%8

    if x0 in [0, 3, 4, 7]:
        if dist > 3:
            return dist, 1
        return dist, 0

    return ring_distance(x0, x1)


def miss_topology(main_core, slice_group, C, h, H):
    core, ring = slice_msg_distance(main_core, slice_group)
    return C + h * core + H*ring

def miss_topology_df(x, C, h, H):
    func = lambda x, C, h, H: miss_topology(x["main_core_fixed"], x["slice_group"], C, h, H)
    return x.apply(func, args=(C, h, H), axis=1)


def remote_hit_topology(main_core, helper_core, slice_group, C, h, H):
    core0, ring0 = slice_msg_distance(main_core, slice_group)
    core1, ring1 = slice_msg_distance(helper_core, slice_group)
    return C + h*(core0+core1) + H*(ring0+ring1)

def remote_hit_topology_df(x, C, h, H):
    func = lambda x, C, h, H: remote_hit_topology(x["main_core_fixed"], x["helper_core_fixed"], x["slice_group"], C, h, H)
    return x.apply(func, args=(C, h, H), axis=1)


def do_predictions(df):
    def plot_predicted_topo(col, row, x_ax, target, pred):
        title_letter = {
            "main_core_fixed": "A",
            "helper_core_fixed": "V",
            "slice_group": "S"
        }.get(col, col[0])

        figure_A0 = sns.FacetGrid(df, col=col, row=row)
        figure_A0.map(sns.scatterplot, x_ax, pred, color="r")
        figure_A0.map(sns.scatterplot, x_ax, target, color="g", marker="+")
        figure_A0.set_titles(col_template="$"+title_letter+"$ = {col_name}")
        plot(f"medians_{pred}_{col}.png")


    
    df = df[(df["main_socket"] == 0) & (df["helper_socket"] == 0)]
    res_miss = optimize.curve_fit(
        miss_topology_df, df[["main_core_fixed", "slice_group"]], df["clflush_miss_n"]
    )
    print("Miss topology:")
    print(res_miss)


    res_remote_hit = optimize.curve_fit(
        remote_hit_topology_df, df[["main_core_fixed", "helper_core_fixed", "slice_group"]], df["clflush_remote_hit"]
    )
    print("Remote hit topology:")
    print(res_remote_hit)


    df["predicted_miss"] = miss_topology_df(df, *(res_miss[0]))
    plot_predicted_topo("slice_group", None, "main_core_fixed", "clflush_miss_n", "predicted_miss")
    plot_predicted_topo("main_core_fixed", None, "slice_group", "clflush_miss_n", "predicted_miss")

    df["predicted_remote_hit"] = remote_hit_topology_df(df, *(res_remote_hit[0]))
    plot_predicted_topo("slice_group", "helper_core_fixed", "main_core_fixed", "clflush_remote_hit", "predicted_remote_hit")
    plot_predicted_topo("main_core_fixed", "helper_core_fixed", "slice_group", "clflush_remote_hit", "predicted_remote_hit")
    
    


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
        separate_hthreads=False,
        title=None,
    ):
    """
    Creates a facet grid showing all points
    """
    if separate_hthreads:
        colors=["y", "r", "g", "b"]
        for el in shown:
            for helper, main in itertools.product((0, 1), (0, 1)):
                df[el+f"_m{main}h{helper}"] = df[(df["main_ht"] == main) & (df["helper_ht"] == helper)][el]

    grid = sns.FacetGrid(df, row=row, col=col)

    for i, el in enumerate(shown):
        if separate_hthreads:
            for helper, main in itertools.product((0, 1), (0, 1)):
                kwargs = {"marker": ['x', '+'][helper]} if draw_fn == sns.scatterplot else {}
                grid.map(
                    draw_fn,
                    third,
                    el+f"_m{main}h{helper}",
                    color=colors[(helper+2*main) % len(colors)],
                    **kwargs
                )
        else:
            grid.map(draw_fn, third, el, color=colors[i % len(colors)])

    if title is not None:
        plot(title, g=grid)
    return grid


def all_facets(df, pre="", post="", *args, **kwargs):
    """
    df : panda dataframe
    pre, post: strings to add before and after the filename
    """

    facet_grid(
        df, "helper_core_fixed", "main_core_fixed", "slice_group",
        title=f"{pre}facet_slice{post}.png", *args, **kwargs
    )
    facet_grid(
        df, "helper_core_fixed", "slice_group", "main_core_fixed",
        title=f"{pre}facet_main{post}.png", *args, **kwargs
    )
    facet_grid(
        df, "main_core_fixed", "slice_group", "helper_core_fixed",
        title=f"{pre}facet_helper{post}.png", *args, **kwargs
    )


def do_facet(main: int, helper: int, line: bool, metrics: str):
    """
    - metrics: hit, miss or all
    """
    df = stats.copy(deep=True)

    print(f"Doing all facets {main}x{helper} {metrics}")
    filtered_df = stats[
        (stats["main_socket"] == main)
        & (stats["helper_socket"] == helper)
    ]
    method = "line" if line else "pt"
    shown = []
    colors = []
    if metrics == "hit" or metrics == "all":
        shown.append("clflush_remote_hit")
        colors.append("r")
    if metrics == "miss" or metrics == "all":
        shown.append("clflush_miss_n")
        colors.append("b")

    all_facets(
        filtered_df,
        pre=f"{metrics}_{method}_",
        post=f"_m{main}h{helper}",
        shown=shown,
        colors=colors,
        draw_fn=sns.lineplot if line else sns.scatterplot
    )


if args.rslice:
    rslice()

# do_predictions(stats)
# all_facets(stats, shown=["clflush_remote_hit"], colors=["r"])



with Pool(8) as pool:
    pool.starmap(
        do_facet,
        itertools.product(
            stats["main_socket"].unique(),
            stats["helper_socket"].unique(),
            (True, False),
            ("hit", "miss")
        )
    )

