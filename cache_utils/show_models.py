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
import numpy as np

nb_cores = 8
nb_slices = 8

cores = list(range(nb_cores))
slices = list(range(nb_slices))



def hit_jump_ring():
    """
    - ini : initial cost
    - core_step : cost to go from one core to the following (eg 0 to 1)
    - ring_step : cost to go from one line to the other (eg 0 to 7)

    Issue: on a same socket, we observe always the first 4 or last 4 patterns, but not mixed
    """
    def hit(helper, slice, i, c, l):
        if helper // (nb_cores/2) != slice // (nb_cores/2):
            mini, maxi = min(slice, helper), max(slice, helper)
            return i+l+c*(min(mini+nb_cores-1-maxi, maxi-mini-1))
        else:
            return i+c*abs(slice-helper)

    ini, core_step, ring_step = 4, 1, 5
    fig, axs = plt.subplots(nrows=1, ncols=8, figsize=(15, 5))

    for i, helper in enumerate(range(8)):
        axs[i].plot(cores, [hit(helper, slice, ini, core_step, ring_step) for slice in slices], "ro")
        axs[i].set_title(f"helper = {helper}")
        axs[i].set_ylabel("clflush_hit_time")
        axs[i].set_xlabel("slice_group")
        axs[i].set_ylim([0, 20])

    plt.tight_layout()
    plt.show()



hit_jump_ring()
