"""
General Tools providing functions for grid size calculation or image merging
"""
from audioop import minmax
from typing import Any, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as nptype
import pygame as pg
from typing_extensions import Self


def calculate_size(length: int) -> Tuple[int, int]:
    """Calculates the width and height for a balanced grid with the given number of images

    Args:
        length (int): the number of images

    Returns:
        Tuple[int, int]: width, height
    """
    root2 = int(np.ceil(np.sqrt(length)))

    idx_x, idx_y = (1, 1)
    for i in range(1, root2 + 1):
        if length % i == 0:
            idx_y = i
            idx_x = length // i
        elif (length + 1) % i == 0:
            idx_y = i
            idx_x = (length + 1) // i

    return idx_x, idx_y


def image_merge_params(prefix="images/") -> List[Tuple[str, str, List[str]]]:
    """Reads the currently available images in the prefix folder and creates a list of tuples

    Args:
        prefix (str, optional): folder to be searched in. Defaults to "images/".

    Returns:
        List[Tuple[str, str, str]]: [(Groupname,MainImagePath,[SubImagePath])]
    """
    import os

    postfix = ".png"
    paths = [p[: -len(postfix)] for p in os.listdir(prefix)]
    mains = [p for p in paths if p.find("[") == -1]
    subs = {s: [] for s in mains}
    for path in paths:
        index = path.find("[")
        if index == -1:
            continue
        group = path[:index]
        subs[group].append(path)

    res: List[Tuple[str, str, List[str]]] = []
    for group, paths in subs.items():
        if group.find("combined") != -1:
            continue
        res.append(
            (
                f"{prefix}combined{group}{postfix}",
                f"{prefix}{group}{postfix}",
                [f"{prefix}{path}{postfix}" for path in paths],
            )
        )

    return res


def combine_images(name: str, paths: List[str]) -> None:
    """Combines the given images to one images and saves it

    Args:
        name (str): new image path
        paths (List[str]): list of paths to images to be combined
    """

    length = len(paths)

    idx_x, idx_y = calculate_size(length)

    images = [pg.image.load(path) for path in paths]
    img_width, img_height = np.max([im.get_size() for im in images], axis=0)

    coords = [
        (x * img_width, y * img_height) for y in range(idx_y) for x in range(idx_x)
    ]

    out_img: pg.Surface = pg.Surface((idx_x * img_width, idx_y * img_height), 0, 32)
    out_img.blits(zip(images, coords))
    pg.image.save_extended(out_img, name)
    print("Writing", name, "-", idx_x * img_width, idx_y * img_height)
    # out_img.blits()
    # for im in images:
    # print(im.get_size())


def get_color(idx: int, name: str = "Set2") -> str:
    """Gets a color from the colormap and returns it as hex string

    Args:
        idx (int): index of the color in cmap
        name (str, optional): name of the color map. Defaults to "Set2".

    Returns:
        str: the color in hex format
    """
    cmap = plt.cm.get_cmap(name)
    color_rgba = cmap(idx)
    color_hex = matplotlib.colors.to_hex(color_rgba)
    return color_hex


def power_set(original_set: List[Any], with_empty: bool = False) -> List[List[Any]]:
    """Calculates the power set

    Args:
        original_set (List[Any]): the set to be calculated from
        with_empty (bool, optional): if the empty set should be in result. Defaults to False.

    Returns:
         List[List[Any]]: power set of the given set
    """
    power_set_size = pow(2, len(original_set))
    print(power_set_size)
    set_size = len(original_set)
    sets = []
    for counter in range(0 if with_empty else 1, power_set_size):
        subset = []
        for i in range(set_size):
            if counter & (1 << i) > 0:
                subset.append(original_set[i])
        sets.append(subset)
    return sets


class StateBounds:
    def __init__(self, bounds: List[Tuple[int, int]]) -> None:
        self.bounds = np.array(bounds)

    def fill_dim(self, idx: int, resolution: int = 100) -> nptype.NDArray:
        return np.linspace(self.bounds[idx, 0], self.bounds[idx, 1], resolution)

    def intersect(self, other: Self) -> Self:
        new_bounds = []
        for self_bound, other_bound in zip(self.bounds, other.bounds):
            min_max = (
                max(self_bound[0], other_bound[0]),
                min(self_bound[1], other_bound[1]),
            )
            new_bounds.append(min_max)
        return StateBounds(new_bounds)

    def union(self, other: Self) -> Self:
        new_bounds = []
        for self_bound, other_bound in zip(self.bounds, other.bounds):
            new_bounds.append(
                (min(self_bound[0], other_bound[0], max(self_bound[1], other_bound[1])))
            )
        return StateBounds(new_bounds)


class Projection:
    def __init__(
        self,
        states: nptype.NDArray,
        ids: nptype.NDArray[np.integer],
        state_bounds: StateBounds,
    ) -> None:
        self.states = np.array(states)
        self.shape = self.states.shape
        self.state_bounds = state_bounds
        self.ids = np.array(ids)
        max_id = self.shape[1]
        for idx in ids:
            if idx >= max_id or idx < 0:
                raise IndexError(
                    f"Cannot project id {idx} out of state with length {max_id}"
                )

        # Remove unnecessary states
        for idx in self.ids:
            self.states[:, idx] = 0

        self.states = np.unique(self.states, axis=0)
        # print(self.states)

    def intersect(self, other: Self) -> Self:
        other = Projection.as_projection(other)
        if self.shape[1] != other.shape[1]:
            raise ArithmeticError(
                f"Cannot intersect projection with shapes {self.shape} and {other.shape}"
            )
        equal_states = []
        still_projected_ids = []
        for self_state in self.states:
            for other_state in other.states:
                equal = True
                tmp_state = np.zeros_like(self_state)
                for i, other_i in enumerate(other_state):
                    # print(f"CMP: {self_state}, {other_state}")
                    if i in self.ids and i in other.ids:
                        still_projected_ids.append(i)
                        # print(f"Added still projected {i}")
                    if i in self.ids:
                        tmp_state[i] = other_i
                        # print(f"{i} is in self.ids state is now {tmp_state}")
                    elif i in other.ids:
                        tmp_state[i] = self_state[i]
                        # print(f"{i} is in other.ids state is now {tmp_state}")
                    elif self_state[i] != other_i:
                        # print(f"states are not equal at {i}")
                        equal = False
                        break
                    else:
                        tmp_state[i] = self_state[i]
                if equal:
                    equal_states.append(tmp_state)
        return Projection(
            equal_states,
            still_projected_ids,
            self.state_bounds.intersect(other.state_bounds),
        )

    @staticmethod
    def as_projection(states: nptype.NDArray | Self):
        if isinstance(states, Projection):
            return states
        min_bounds = np.min(states, axis=0)
        max_bounds = np.max(states, axis=0)
        return Projection(states, [], StateBounds(list(zip(min_bounds, max_bounds))))

    def to_array(self, resolution=10) -> nptype.NDArray:
        output = np.array(self.states)
        # Projecting out individual indices
        for idx in self.ids:
            tmp_output = []
            # For every state in the current output append projected states
            for state in output:
                extended_state = np.repeat([state], resolution, axis=0)
                filler = self.state_bounds.fill_dim(idx, resolution)
                extended_state[:, idx] = filler
                # Append states individually to avoid boxing
                for ex_state in extended_state:
                    tmp_output.append(ex_state)
            output = np.asarray(tmp_output)
        return np.unique(output, axis=0)


# if __name__ == "__main__":
#     bounds = StateBounds([(-1, 1), (-1, 1)])
#     states1 = np.asarray([[-0.2, -1], [0, -1], [0.2, -1]], dtype=np.float64)
#     # states2 = np.asarray([[-1, -0.2], [-1, 0], [-1, 0.2]], dtype=np.float64)
#     states2 = np.asarray([[-0.3, -1], [-0.1, -1], [0, -1]], dtype=np.float64)
#     projection1 = Projection(states1, [1], bounds)
#     projection2 = Projection(states2, [1], bounds)
#     new_states1 = projection1.to_array()
#     new_states2 = projection2.to_array()

#     intersect_states = projection1.intersect(projection2).to_array()
#     # print(new_states)

#     plt.figure()
#     plt.scatter(new_states1[:, 0], new_states1[:, 1], alpha=0.2)
#     plt.scatter(new_states2[:, 0], new_states2[:, 1], alpha=0.2)
#     plt.scatter(intersect_states[:, 0], intersect_states[:, 1])
#     plt.scatter(states1[:, 0], states1[:, 1])
#     plt.scatter(states2[:, 0], states2[:, 1])
#     plt.suptitle("After")
#     plt.show()
