"""
Try some models and see what they look like
Following this die could help https://en.wikichip.org/w/images/4/48/E5_v4_LCC.png
Using the following naming convention:

    ------
    0    7
    1    6
    2    5
    3    4
    ------
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import os

nb_cores = 8
nb_slices = 8
num_core = nb_cores

cores = list(range(nb_cores))
slices = list(range(nb_slices))

img_dir = os.getenv("PWD")+"/"
def plot(filename, g=None):
    if g is not None:
        g.savefig(img_dir + filename)
    else:
        plt.savefig(img_dir + filename)
    # tikzplotlib.save(
    #   img_dir+filename+".tex",
    #   axis_width=r'0.175\textwidth',
    #   axis_height=r'0.25\textwidth'
    # )
    print(img_dir + filename, "saved")
    plt.close()

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

def no_QPI_dist(source, dest):
    """
    Path not using QPI hop
    """
    return abs(source-dest), 1 if source // 4 != dest //4 else 0


def miss():
    """
    - ini : initial cost
    - core_step : cost to go from one core to the following (eg 0 to 1)
    - ring_step : cost to go from one line to the other (eg 0 to 7)

    Issue: on a same socket, we observe always the first 4 or last 4 patterns, but not mixed
    """
    def miss_topo(main_core, slice, i, c, l, k):
        x, y, z = slice_msg_distance(main_core, slice)
        return i+c*x+l*y+k*z

    ini, core_step, ring_step, other_ring_step = 4, 1, 5, 2
    fig, axs = plt.subplots(nrows=1, ncols=8, figsize=(15, 5))

    for i, slice in enumerate(range(8)):
        axs[i].plot(cores, [miss_topo(main_core, slice, ini, core_step, ring_step, other_ring_step) for main_core in cores], "ro")
        axs[i].set_title(f"slice_group = {slice}")
        axs[i].set_ylabel("clflush_miss_n")
        axs[i].set_xlabel("main_core_fixed")
        axs[i].set_ylim([0, 25])

    plt.tight_layout()
    plt.show()


def hit():
    """
    - ini : initial cost
    - core_step : cost to go from one core to the following (eg 0 to 1)
    - ring_step : cost to go from one line to the other (eg 0 to 7)

    Issue: on a same socket, we observe always the first 4 or last 4 patterns, but not mixed
    """
    def hit_topo(main, helper, slice_g, i, c, l):
        helper = helper%8

        main_slice_local = slice_msg_distance(slice_g, main)
        slice_QPI = cclockwise_dist(0, slice_g) # clockwise
        QPI_slice_r = cclockwise_dist(0, slice_g)
        slice_r_helper = slice_msg_distance(slice_g, helper)

        costs = (main_slice_local[0]+slice_QPI[0]+QPI_slice_r[0]+slice_r_helper[0], main_slice_local[1]+slice_QPI[1]+QPI_slice_r[1]+slice_r_helper[1])
        return ini+costs[0]*c+costs[1]*l


    ini, core_step, ring_step = 12, 1, 1.5
    fig, axs = plt.subplots(nrows=1, ncols=8, figsize=(15, 5))

    # Define the ranges for x, y, z
    main = range(8)
    helper = range(8, 16)
    slice_g = range(8)

    # Create a DataFrame with all combinations of x, y, and z
    data = pd.DataFrame([
            (x, y, z) for z in slice_g
            for x, y in itertools.product(main, helper)
        ],
        columns=['main', 'helper', 'slice_group']
    )

    # Define the function
    def my_function(x):
        return hit_topo(x["main"], x["helper"], x["slice_group"], ini, core_step, ring_step)

    # Apply the function to create a new column
    data['predicted_hit'] = data.apply(my_function, axis=1)

    fig = sns.FacetGrid(data, col="main", row="helper")
    fig.map(sns.scatterplot, "slice_group", "predicted_hit", color="r", marker="+")
    fig.map(sns.scatterplot, "slice_group", "predicted_hit", color="r", marker="x")
    fig.set_titles(col_template="$A$ = {col_name}", row_template="$V$ = {row_name}")

    plot("model_hit.png", g=fig)



hit()
