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
    it is possible that going from 0->7 costs one more than 3->4
    """
    dist = abs(x0-x1)
    if x0 // (num_core/2) != x1 // (num_core/2):
        # côté du coeur différent
        return min((num_core-1-dist, 2), (dist-1, 1))
    else:
        return dist, 0

def slice_msg_distance(source, dest):
    """
    Si l'expéditeur est à l'extrémité d'une des lignes, il envoie toujours dans le même sens
    (vers toute sa ligne d'abord), sinon, il prend le chemin le plus court
    le bonus correspond au fait que 0->7 puisse coûter 1 de plus que 3->4
    """
    dist = abs(source-dest)
    if source // (num_core/2) == dest // (num_core/2):
        return (dist, 0)

    # Pour aller de l'autre côté
    up, down = (num_core-1-dist, 2), (dist-1, 1)
    if source in [0, 7]:
        return down
    if source in [3, 4] or source in [2, 5]:
        return up
    if source in [1, 6]:
        return min(up, down)

    raise IndexError

def ha_dist(core, is_QPI):
    """
    distance to Home Agent
    """
    if is_QPI:
        if core < 4:
            return core, 0
        return 7-core, 1 # +1 for PCI

    if core < 4:
        return 3-core, 0
    return core-4, 0

def cclockwise_dist(source, dest):
    base = (dest+8-source)%8
    side_jump = 0
    if source < 4 and dest >= 4:
        side_jump = 1
    elif source >= 4 and dest < 4:
        side_jump = 2
    return base, side_jump

def cclockwise_ha_dist(core, is_QPI):
    """
    counter-clockwise distance to Home Agent
    """
    if is_QPI:
        return cclockwise_dist(core, 7)
    return cclockwise_dist(core, 3)

def miss_topology(main_core, slice_group, h, down_jump, top_jump, ini, ha_h):
    core, ring = slice_msg_distance(slice_group, main_core%8)

    side_jump = 0
    side_jump += top_jump if ring == 2 else 0
    side_jump += down_jump if ring == 1 else 0
    return (cclockwise_ha_dist(slice_group, False))*ha_h+h*core + side_jump + ini

def miss_topology_df(x, h, down_jump, top_jump, ini, ha_h):
    func = lambda x, h, down_jump, top_jump, ini, ha_h: miss_topology(x["main_core_fixed"], x["slice_group"], h, down_jump, top_jump, ini, ha_h)
    return x.apply(func, args=(h, down_jump, top_jump, ini, ha_h), axis=1)


def remote_hit_topology(main_core, helper_core, slice_group, const, core_jump, HA_jump):
    """
    main_core
    -> local_slice
    -> remote_slice
    -> helper_core
    -> remote_slice
    -> local_slice
    -> main_core
    """
    if main_core // 8 == helper_core // 8:
        print("Can only do hit predictions for different socket", file=sys.stderr)
        raise NotImplementedError

    helper, main = helper_core%8, main_core%8
    main_slice_local = slice_msg_distance(slice_group, main)
    slice_QPI = cclockwise_dist(0, slice_group) # clockwise
    QPI_slice_r = cclockwise_dist(0, slice_group)
    slice_r_helper = slice_msg_distance(slice_group, helper)

    costs = (main_slice_local[0]+slice_QPI[0]+QPI_slice_r[0]+slice_r_helper[0], main_slice_local[1]+slice_QPI[1]+QPI_slice_r[1]+slice_r_helper[1])
    return const+costs[0]*core_jump+costs[1]*HA_jump # may need some adjustments

def remote_hit_topology_df(x, const, core_jump, HA_jump):
    func = lambda x, const, core_jump, HA_jump: remote_hit_topology(x["main_core_fixed"], x["helper_core_fixed"], x["slice_group"], const, core_jump, HA_jump)
    return x.apply(func, args=(const, core_jump, HA_jump), axis=1)


def do_predictions(df):
    def plot_predicted_topo(col, row, x_ax, target, pred, df=df):
        titles = {
            "main_core_fixed": "A",
            "helper_core_fixed": "V",
            "slice_group": "S",
            None: "None"
        }

        figure_A0 = sns.FacetGrid(df, col=col, row=row)
        figure_A0.map(sns.scatterplot, x_ax, target, color="g")
        figure_A0.map(sns.scatterplot, x_ax, pred, color="r", marker="+")
        figure_A0.set_titles(
            col_template="$"+titles.get(col, col[0])+"$ = {col_name}",
            row_template="$"+titles.get(row, row[0])+"$ = {row_name}"
        )
        plot(f"medians_{pred}_{col}.png")


    main_socket, helper_socket = 0, 0
    dfc = df[(df["main_socket"] == main_socket) & (df["helper_socket"] == helper_socket)]
    res_miss = optimize.curve_fit(
        miss_topology_df, dfc[["main_core_fixed", "slice_group"]], dfc["clflush_miss_n"]
    )
    print("Miss topology:")
    print(res_miss)

    dfc["predicted_miss"] = miss_topology_df(dfc, *(res_miss[0]))
    plot_predicted_topo("slice_group", None, "main_core_fixed", "clflush_miss_n", "predicted_miss", df=dfc)
    plot_predicted_topo("main_core_fixed", None, "slice_group", "clflush_miss_n", "predicted_miss", df=dfc)

    main_socket, helper_socket = 0, 1
    dfc = df[(df["main_socket"] == main_socket) & (df["helper_socket"] == helper_socket)]
    res_remote_hit = optimize.curve_fit(
        remote_hit_topology_df, dfc[["main_core_fixed", "helper_core_fixed", "slice_group"]], dfc["clflush_remote_hit"]
    )
    print("Remote hit topology:")
    print(res_remote_hit)


    df["diff_miss"] = df["clflush_miss_n"] - df["predicted_miss"]
    facet_grid(
        df, None, "main_core_fixed", "slice_group",
        title=f"predicted_miss_diff_facet_slice.png",
        shown=["diff_miss"],
        separate_hthreads=True
    )
    facet_grid(
        df, None, "slice_group", "main_core_fixed",
        title=f"predicted_miss_diff_facet_main.png",
        shown=["diff_miss"],
        separate_hthreads=True
    )
    dfc["predicted_remote_hit"] = remote_hit_topology_df(dfc, *(res_remote_hit[0]))
    plot_predicted_topo("slice_group", "helper_core_fixed", "main_core_fixed", "clflush_remote_hit", "predicted_remote_hit", df=dfc)
    plot_predicted_topo("main_core_fixed", "slice_group", "helper_core_fixed", "clflush_remote_hit", "predicted_remote_hit", df=dfc)
    plot_predicted_topo("helper_core_fixed", "main_core_fixed", "slice_group", "clflush_remote_hit", "predicted_remote_hit", df=dfc)

    for col in ["slice_group", "helper_core_fixed", "main_core_fixed"]:
        for val in sorted(list(dfc[col].unique())):
            df_temp = dfc[(dfc[col] == val)]
            res_remote_hit = optimize.curve_fit(
            remote_hit_topology_df, df_temp[["main_core_fixed", "helper_core_fixed", "slice_group"]], df_temp["clflush_remote_hit"]
            )
            df_temp[f"predicted_remote_hit_{col}={val}"] = remote_hit_topology_df(df_temp, *(res_remote_hit[0]))
            plot_predicted_topo("slice_group", "helper_core_fixed", "main_core_fixed", "clflush_remote_hit", f"predicted_remote_hit_{col}={val}", df=df_temp)
            plot_predicted_topo("main_core_fixed", "helper_core_fixed", "slice_group", "clflush_remote_hit", f"predicted_remote_hit_{col}={val}", df=df_temp)




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
        letters=None
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
            grid.map(draw_fn, third, el, color=colors[i % len(colors)], marker='+')

    if letters is not None:
        grid.set_titles(col_template="$"+letters[0]+"$ = {row_name}", row_template="$"+letters[1]+"$ = {col_name}")
    

    if title is not None:
        plot(title, g=grid)
    return grid


def all_facets(df, pre="", post="", no_helper=False, *args, **kwargs):
    """
    df : panda dataframe
    pre, post: strings to add before and after the filename
    """

    helper = None if no_helper else "helper_core_fixed"
    facet_grid(
        df, helper, "main_core_fixed", "slice_group",
        title=f"{pre}facet_slice{post}.png", *args, **kwargs,
        separate_hthreads=False
    )
    facet_grid(
        df, helper, "slice_group", "main_core_fixed",
        title=f"{pre}facet_main{post}.png", *args, **kwargs,
        separate_hthreads=False
    )
    facet_grid(
        df, "main_core_fixed", "slice_group", "helper_core_fixed",
        title=f"{pre}facet_helper{post}.png", *args, **kwargs,
        separate_hthreads=False
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
        no_helper=True,
        draw_fn=sns.lineplot if line else sns.scatterplot
    )


if args.rslice:
    rslice()

do_predictions(stats)
all_facets(stats, shown=["clflush_remote_hit"], colors=["r"], pre="hit_")
all_facets(stats, shown=["clflush_miss_n"], colors=["b"], pre="miss_")

def compare_facing():
    df=stats
    for m, h, s in itertools.product((0, 1), (0, 1), df["slice_group"].unique()):
        dfc = df[(df["main_socket"] == m) & (df["main_core_fixed"]%8 == s) & (df["helper_socket"] == h)]

        grid = sns.FacetGrid(dfc, row=None, col=None)
        grid.map(sns.scatterplot, "slice_group", "clflush_miss_n", marker="+")

        plot(f"miss_m{m}h{h}m{s}", g=grid)


def isolate_sockets():
    with Pool(8) as pool:
        pool.starmap(
            do_facet,
            itertools.product(
                stats["main_socket"].unique(),
                stats["helper_socket"].unique(),
                (False, ),
                ("hit", "miss")
            )
        )

def superpose_sockets():
    for main, same_socket in itertools.product(sorted(stats["main_core_fixed"].unique()), (True, False)):
        df = stats[
            (stats["slice_group"] == (main%8))
            & (stats["main_core_fixed"] == main)
            & ((stats["main_socket"] == stats["helper_socket"]) == same_socket)
        ]
        ax = sns.scatterplot(df, x="helper_core_fixed", y="clflush_remote_hit", marker="+", color="r")
        ax.set_title(f"$S = {main%8}, V = {main}$")
        plot(f"hit_{same_socket}_main{main:02d}.png")

    df = stats[
        (stats["slice_group"] == (stats["main_core_fixed"]%8))
        & ((stats["main_core_fixed"]%8) == (stats["helper_core_fixed"]%8))
        & (stats["main_socket"] != stats["helper_socket"])
    ]
    ax = sns.scatterplot(df, x="slice_group", y="clflush_remote_hit", marker="+", color="r")
    plot(f"hit_same_slice.png")

    stats["main_core_nosock"] = stats["main_core_fixed"]%8
    stats["helper_core_nosock"] = stats["helper_core_fixed"]%8

    facet_grid(
        stats[(stats["main_socket"] != stats["helper_socket"])], "helper_core_nosock", "main_core_nosock", "slice_group",
        title=f"hit_facet_slice_diff_socket.png",
        separate_hthreads=True,
        shown=["clflush_remote_hit"],
        letters="VA"
    )

    facet_grid(
        stats[(stats["main_socket"] == stats["helper_socket"])], "helper_core_nosock", "main_core_nosock", "slice_group",
        title=f"hit_facet_slice_same_socket.png",
        separate_hthreads=True,
        letters="VA",
        shown=["clflush_remote_hit"]
    )