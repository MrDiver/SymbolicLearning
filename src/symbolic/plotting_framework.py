"""
####################################################################

                    PLOTTING SECTION

####################################################################
"""


from typing import List

import numpy as np
from matplotlib import pyplot as plt

from symbolic.agent import Playroom, PlayroomAgent
from symbolic.option_framework import Option, partition_options
from symbolic.tools import calculate_size, combine_images, image_merge_params


def plot_planning_graph(g):

    dotstring = "strict digraph {\n"
    dotstring += """
    layout = sfdp;
	//K=1;
	start=0
	overlap = prism1000;
    splines = curved;
    repulsiveforce = 2.0;
    sep = 5;
    smoothing = "spring";
    """
    dotstring += "{\n"
    dotstring += '\tnode[margin=0 color="#222222" penwidth=2.0 shape=circle style=filled fontname=serif]\n'
    for n in g.nodes:
        propstring = ""
        for prop, propmap in g.node_properties.items():
            propstring += '{}="{}" '.format(prop, str(propmap[n]))
        dotstring += "\t{} [{}];\n".format(str(n.id), propstring)
    dotstring += "}\n"

    dotstring += '\t edge[style="solid" penwidth=2.0 colorscheme=spectral11]'
    for e in g.edges:
        propstring = ""
        for prop, propmap in g.edge_properties.items():
            propstring += '{}="{}" '.format(prop, str(propmap[e]))
        dotstring += "\t{} -> {} [{}];\n".format(str(e.a.id), str(e.b.id), propstring)

    dotstring += "}\n"
    file = open("data/graph.dot", "w")
    file.write(dotstring)
    file.close()
    import os

    os.system("sfdp -Tpng data/graph.dot > data/graph.png")


def plot_playroom(pr: Playroom, agent: PlayroomAgent):
    print("- Plot Playroom -")
    # Plot Options
    init_color = (0, 255, 0)
    effect_color = (255, 255, 0)

    def plot_option(o):
        pr.overlay_background()
        pr.overlay_states(np.unique(o.initiation(), axis=0), init_color, ui_offset=-100)
        pr.overlay_states(np.unique(o.effect(), axis=0), effect_color)
        for t in o.transitions:
            pr.overlay_transition(t)
        pr.overlay_text(o.name, (1, 1), size=20)
        pr.overlay_text(
            "Initiation Set",
            (1, pr.game.HEIGHT - 25),
            color=init_color,
            size=20,
        )
        pr.overlay_text(
            "Effect Set",
            (200, pr.game.HEIGHT - 25),
            color=effect_color,
            size=20,
        )
        pr.update_screen()
        # pr.save_image(
        #     "images/{}_{}.png".format(
        #         datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S"), o.name
        #     )
        # )
        pr.save_image("images/{}.png".format(o.name))

    for i, o in enumerate(agent.options):
        if o.empty():
            print("Skipping option {}", o.name)
            continue
        plot_option(o)
    # Plot Subgoal Options
    subgoals = partition_options(agent.options)

    for i, o in enumerate(subgoals):
        if o.empty():
            print("Skipping option {}", o.name)
            continue
        plot_option(o)

    params = image_merge_params()
    newpaths = []
    optionpaths = []
    for name, main, paths in params:
        for p in paths:
            newpaths.append(p)
        paths.append(main)
        optionpaths.append(main)
        combine_images(name, paths)
    combine_images("images/combinedALL.png", newpaths)
    combine_images("images/combinedOptions.png", optionpaths)


def plot_options(options: List[Option]):

    subgoals = partition_options(options)

    def plot_samples():
        print("- Plot Samples -")
        cols, rows = calculate_size(len(options))
        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(8, 8))
        axs = axs.flatten()
        for i, o in enumerate(options):
            if o.empty():
                print("Skipping option {}", o.name)
                continue
            init = o.initiation()
            eff = o.effect()
            axs[i].scatter(init[:, 0], init[:, 1], label="init", alpha=0.1)
            axs[i].scatter(eff[:, 0], eff[:, 1], label="eff")
            axs[i].set_title(o.name)

        fig.tight_layout()

    def plot_clusters():
        print("- Plot Clusters -")
        cols, rows = calculate_size(len(options))
        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(12, 12))
        axs = axs.flatten()
        for so in subgoals:
            if so.empty():
                print("Skipping option {}", so.name)
                continue
            init = so.initiation()
            eff = so.effect()
            # find index
            for i, o in enumerate(options):
                if o == so.option:
                    so.i = i
                    break
            axs[so.i].scatter(
                init[:, 0],
                init[:, 1],
                label=so.name,
                c="C" + str(so.index),
                marker="x",
            )
            axs[so.i].scatter(
                eff[:, 0], eff[:, 1], label=so.name, c="C" + str(so.index)
            )
        for i, o in enumerate(options):
            axs[i].set_title(o.name)

        fig.tight_layout()

    def plot_classifiers():
        print("- Plot Classifiers -")
        cols, rows = calculate_size(len(subgoals) * 2)
        fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(12, 12))
        axs = axs.flatten()

        resolution = 100
        for ax in axs:
            ax.set_axis_off()
        for i, o in enumerate(subgoals * 2):
            if o.empty():
                print("Skipping option {}", o.name)
                continue
            # field = np.zeros((240, 240))
            state_length = len(o.initiation()[0])
            test_values = np.transpose(
                np.meshgrid(
                    np.linspace(-10, 310, resolution),
                    np.linspace(-10, 310, resolution),
                    np.linspace(0, 0, 1) if i < len(subgoals) else np.linspace(1, 1, 1),
                )
            ).reshape((-1, state_length))

            p = o.init_probability(test_values)
            field = p.reshape((resolution, resolution)).T
            # field[y, x] = p
            ax: plt.Axes = axs[i]
            ax.imshow(field)
            ax.set_title("Key:{} {}".format(i < len(subgoals), o.name))
        fig.suptitle("Initiation Sets of Partitioned Subgoals")
        fig.tight_layout()

    # plot_samples()
    # plot_clusters()
    plot_classifiers()
    plt.show()
