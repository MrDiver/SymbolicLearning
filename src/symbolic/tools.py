"""
General Tools providing functions for grid size calculation or image merging
"""
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame as pg


def calculate_size(length: int) -> Tuple[int, int]:
    root2 = int(np.ceil(np.sqrt(length)))

    IDX_X, IDX_Y = (1, 1)
    for i in range(1, root2 + 1):
        if length % i == 0:
            IDX_Y = i
            IDX_X = length // i
        elif (length + 1) % i == 0:
            IDX_Y = i
            IDX_X = (length + 1) // i

    return IDX_X, IDX_Y


def image_merge_params(prefix="images/"):
    import os

    postfix = ".png"
    paths = [p[: -len(postfix)] for p in os.listdir(prefix)]
    mains = [p for p in paths if p.find("[") == -1]
    subs = {s: [] for s in mains}
    for s in paths:
        index = s.find("[")
        if index == -1:
            continue
        group = s[:index]
        subs[group].append(s)

    res = []
    for group, paths in subs.items():
        if group.find("combined") != -1:
            continue
        res.append(
            (
                "{}{}{}{}".format(prefix, "combined", group, postfix),
                "{}{}{}".format(prefix, group, postfix),
                ["{}{}{}".format(prefix, path, postfix) for path in paths],
            )
        )

    return res


def combine_images(name, paths):

    length = len(paths)

    IDX_X, IDX_Y = calculate_size(length)

    images = [pg.image.load(path) for path in paths]
    IMG_WIDTH, IMG_HEIGHT = np.max([im.get_size() for im in images], axis=0)

    coords = [
        (x * IMG_WIDTH, y * IMG_HEIGHT) for y in range(IDX_Y) for x in range(IDX_X)
    ]

    out_img: pg.Surface = pg.Surface((IDX_X * IMG_WIDTH, IDX_Y * IMG_HEIGHT), 0, 32)
    out_img.blits(zip(images, coords))
    pg.image.save_extended(out_img, name)
    print("Writing", name, "-", IDX_X * IMG_WIDTH, IDX_Y * IMG_HEIGHT)
    # out_img.blits()
    # for im in images:
    # print(im.get_size())


def get_color(idx: int, name: str = "Set2"):
    cmap = plt.cm.get_cmap(name)
    color_rgba = cmap(idx)
    color_hex = matplotlib.colors.to_hex(color_rgba)
    return color_hex


def power_set(x, with_zero=False):
    power_set_size = pow(2, len(x))
    print(power_set_size)
    set_size = len(x)
    sets = []
    for counter in range(0 if with_zero else 1, power_set_size):
        subset = []
        for i in range(set_size):
            if counter & (1 << i) > 0:
                subset.append(x[i])
        sets.append(subset)
    return sets


class Projection:
    def __init__(self) -> None:
        pass
