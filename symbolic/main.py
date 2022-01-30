"""
Main Module
"""
import random as rng
import time
from collections import namedtuple
from typing import Any, Callable, List, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from Game import Direction, MiniGame, Player, combine_images

# from hdbscan import HDBSCAN
from nptyping import Bool, Float, NDArray
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LowLevelState: TypeAlias = NDArray[Float]
LowLevelStates: TypeAlias = NDArray[Float]
SymbolicState: TypeAlias = NDArray[Bool]
StateTransform: TypeAlias = Callable[[LowLevelState], LowLevelState]
StateTest: TypeAlias = Callable[[LowLevelState], float]
StateConsumer: TypeAlias = Callable[[LowLevelState], Any]


"""[summary]
    state - refers to the new state after the option is executed
    executed - true iff the option could be executed
    i.e. state was in initiation set
"""
OptionExecuteReturn = namedtuple("OptionExecuteReturn", ["state", "executed", "time"])
OptionTransition = namedtuple("OptionTransition", ["start", "dest", "states"])


class Option:
    def __init__(
        self,
        name: str,
        policy: StateTransform,
        init_test: StateTest,
        termination: StateConsumer,
    ) -> None:
        self.name = name
        self.init_test = init_test
        self.policy = policy
        self.termination = termination

        self.initiation_set: List[LowLevelState] = []
        self.effect_set: List[LowLevelState] = []
        self.transitions: List[OptionTransition] = []

    def _add_transition(
        self, start: LowLevelState, dest: LowLevelState, states: LowLevelStates
    ):
        self.initiation_set.append(start)
        self.effect_set.append(dest)
        self.transitions.append(OptionTransition(start, dest, states))

    def effect(self) -> LowLevelStates:
        return np.array(self.effect_set)

    def initiation(self) -> LowLevelStates:
        return np.array(self.initiation_set)

    def execute(
        self, state: LowLevelState, update_callback: Callable[[LowLevelState], None]
    ) -> OptionExecuteReturn:
        if self.init_test(state) > 0:
            return OptionExecuteReturn(state, False, 0)
        terminated = False
        start_state = state
        start_time = time.time()
        states = [state]
        while not terminated:
            state = self.policy(state)
            terminated = self.termination(state) > 0
            update_callback(state)
            states.append(state)

        self._add_transition(start_state, state, states)

        return OptionExecuteReturn(state, True, time.time() - start_time)


class SubgoalOption(Option):
    def __init__(
        self,
        index: int,
        option: Option,
        init_test: StateTest,
        initiation_set: LowLevelStates,
        effect_set: LowLevelStates,
        transitions: List[OptionTransition],
    ):
        super().__init__(
            "{}[{}]".format(option.name, index),
            option.policy,
            init_test,
            option.termination,
        )
        self.index = index
        self.option = option
        for i, (start, dest) in enumerate(zip(initiation_set, effect_set)):
            self._add_transition(start, dest, transitions[i].states)


def partition_options(options: List[Option]) -> List[SubgoalOption]:
    """
    Partition options with DBSCAN into subgoals
    Train SVM for init classifier
    """
    subgoals: List[SubgoalOption] = []
    for i, option in enumerate(options):
        db = DBSCAN(eps=0.1, min_samples=1).fit(
            StandardScaler().fit_transform(option.effect())
        )
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        unique_labels = set(labels)

        print("Partitioned", option.name, "into", n_clusters_, "clusters")
        for k in unique_labels:
            class_member_mask = labels == k

            eff_sub = option.effect()[class_member_mask & core_samples_mask]
            init_sub = option.initiation()[class_member_mask & core_samples_mask]
            trans_sub = np.array(option.transitions, dtype=object)[
                class_member_mask & core_samples_mask
            ]
            trans_sub = [
                x
                for i, x in enumerate(option.transitions)
                if (class_member_mask & core_samples_mask)[i]
            ]
            svm = SVC(probability=True)

            def test(s: LowLevelState) -> bool:
                return 1.0

            subgoals.append(
                SubgoalOption(k, option, test, init_sub, eff_sub, trans_sub)
            )
            # svm.fit()
            subgoals[-1].svm = svm

    return subgoals


class PropositionSymbol:
    def __init__(self, name: str, test: StateTest) -> None:
        self.name = name
        self.test = test
        self.low_level_states: List[LowLevelState] = []

    def p_test(self, state: LowLevelState):
        res = self.test(state)
        if res:
            self.low_level_states.append("Hallo")
        return res

    def p_and(self, other):
        return __make_symbol(
            "(" + self.name + "&" + other.name + ")",
            (lambda x: self.test(x) and other.test(x)),
        )

    def p_or(self, other):
        return __make_symbol(
            "(" + self.name + "|" + other.name + ")",
            (lambda x: self.test(x) or other.test(x)),
        )

    def p_not(self):
        return __make_symbol("not_" + self.name, (lambda x: not self.test(x)))

    def p_null(self) -> bool:
        return len(self.low_level_states) == 0


def __make_symbol(name: str, test: StateTest) -> PropositionSymbol:
    return PropositionSymbol(name, test)


class Plan:
    def __init__(self, seq: List[Option]) -> None:
        self.seq = seq

    # Plan feasability definition 3


class Playroom:
    def __init__(self) -> None:
        self.game = MiniGame()

    def get_state(self) -> LowLevelState:
        return np.array([self.game.player.pos.x, self.game.player.pos.y])

    def set_state(self, s: LowLevelState):
        self.game.player.set_state(s[0], s[1])

    def execute_no_change(
        self, s: LowLevelState, f: Callable[[], Any]
    ) -> LowLevelState:
        old_state = self.get_state()
        self.set_state(s)
        f()
        new_state = self.get_state()
        self.set_state(old_state)
        return new_state

    def execute_no_change_return(
        self, s: LowLevelState, f: Callable[[], Any]
    ) -> LowLevelState:
        old_state = self.get_state()
        self.set_state(s)
        res = f()
        self.set_state(old_state)
        return res

    def draw(self) -> None:
        self.game.draw()

    def draw_state(self, s: LowLevelState):
        old_state = self.get_state()
        self.set_state(s)
        self.game.draw()
        self.set_state(old_state)

    def overlay_background(self):
        self.game.overlay_background()

    def overlay_state(self, s: LowLevelState, color=(255, 0, 0), alpha=10):
        old_state = self.get_state()
        self.set_state(s)
        self.game.overlay(color, alpha)
        self.set_state(old_state)

    def overlay_transition(
        self,
        transition: OptionTransition,
        start_color=(255, 0, 0),
        end_color=(0, 255, 0),
        alpha=25,
    ):
        self.game.overlay_transition(
            (transition.start[0], transition.start[1]),
            (transition.dest[0], transition.dest[1]),
            start_color,
            end_color,
            alpha,
        )

    def overlay_states(self, states: LowLevelStates, color):
        for state in states:
            self.overlay_state(state, color, 220.0 / len(states) + 30)

    def overlay_text(self, text: str, pos, size=24, color=(255, 255, 255)):
        self.game.overlay_text(text, pos, size, color)

    def update_screen(self):
        self.game.update_screen()

    def save_image(self, name: str):
        self.game.screenshot(name)


class Agent:
    def __init__(self, options: List[Option]):
        self.options: List[Option] = options


class PlayroomAgent(Agent):
    def __init__(self, playroom: Playroom, player: Player):
        super().__init__(
            [
                Option(
                    "MoveLeft",
                    self._move(Direction.LEFT),
                    self._move_test(Direction.LEFT),
                    self._move_terminate(Direction.LEFT),
                ),
                Option(
                    "MoveRight",
                    self._move(Direction.RIGHT),
                    self._move_test(Direction.RIGHT),
                    self._move_terminate(Direction.RIGHT),
                ),
                Option(
                    "MoveUp",
                    self._move(Direction.UP),
                    self._move_test(Direction.UP),
                    self._move_terminate(Direction.UP),
                ),
                Option(
                    "MoveDown",
                    self._move(Direction.DOWN),
                    self._move_test(Direction.DOWN),
                    self._move_terminate(Direction.DOWN),
                ),
            ]
        )
        self.player = player
        self.playroom = playroom

    def _move(self, direction: Direction) -> StateTransform:
        def _move_dir(s: LowLevelState) -> LowLevelState:
            return self.playroom.execute_no_change(
                s, (lambda: self.player.move(direction))
            )

        return _move_dir

    def _move_test(self, direction: Direction) -> StateTest:
        def _move_test_dir(s: LowLevelState) -> float:
            return self.playroom.execute_no_change_return(
                s, (lambda: 1 if self.player.collide(direction) else 0)
            )

        return _move_test_dir

    def _move_terminate(self, direction: Direction) -> StateConsumer:
        def _move_terminate_dir(s: LowLevelState) -> float:
            return (
                1
                if self.playroom.execute_no_change_return(
                    s, (lambda: self.player.collide(direction))
                )
                else 0
            )

        return _move_terminate_dir


def main():
    pr = Playroom()
    game = pr.game
    agent: Agent = PlayroomAgent(pr, game.player)
    state = pr.get_state()

    def recurse_options(state: LowLevelState, options: List[Option], depth: int):
        if depth <= 0:
            return
        for o in options:
            res = o.execute(state, (lambda s: None))
            if res.executed:
                recurse_options(res.state, options, depth - 1)
            # pr.draw_state(res.state)

    def random_options(state: LowLevelState, options: List[Option], depth: int):
        i = 0
        while i < depth:
            res = options[rng.randint(0, len(options) - 1)].execute(
                state, (lambda s: None)
            )
            # pr.draw_state(res.state)
            state = res.state
            i += 1

    random_options(state, agent.options, 1000)
    plot_playroom(pr, agent)

    params = image_merge_params()
    for name, paths in params:
        combine_images(name, paths)
    pr.game.destroy()
    plot_options(agent.options)


def plot_playroom(pr: Playroom, agent: PlayroomAgent):
    # Plot Options
    def plot_option(o):
        pr.overlay_background()
        pr.overlay_states(np.unique(o.initiation(), axis=0), (0, 255, 0))
        pr.overlay_states(np.unique(o.effect(), axis=0), (0, 0, 255))
        for t in o.transitions:
            pr.overlay_transition(t)
        pr.overlay_text(o.name, (0, 0))
        pr.overlay_text("- Initiation Set", (0, 25), color=(50, 255, 50))
        pr.overlay_text("- Effect Set", (0, 50), color=(50, 50, 255))
        pr.update_screen()
        # pr.save_image(
        #     "images/{}_{}.png".format(
        #         datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S"), o.name
        #     )
        # )
        pr.save_image("images/{}.png".format(o.name))

    for i, o in enumerate(agent.options):
        plot_option(o)
    # Plot Subgoal Options
    subgoals = partition_options(agent.options)

    for i, o in enumerate(subgoals):
        plot_option(o)


def plot_options(options: List[Option]):
    def plot_samples():
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
        axs = axs.flatten()
        for i, o in enumerate(options):
            init = o.initiation()
            eff = o.effect()
            axs[i].scatter(init[:, 0], init[:, 1], label="init")
            axs[i].scatter(eff[:, 0], eff[:, 1], label="eff")
            axs[i].set_title(o.name)

    def plot_clusters():
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
        axs = axs.flatten()
        subgoals = partition_options(options)
        for so in subgoals:
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

    def plot_classifiers():
        subgoals = partition_options(options)
        length = int(np.ceil(np.sqrt(len(subgoals))))
        fig, axs = plt.subplots(ncols=length, nrows=length, figsize=(12, 12))
        axs = axs.flatten()

        # for i, o in enumerate(options):
        #     field = np.zeros((240, 240))
        #     test_values = np.transpose(
        #         np.meshgrid(np.arange(0, 240, 1), np.arange(0, 240, 1))
        #     ).reshape((240 * 240, 2))
        #     # print(test_values)
        #     for so in subgoals:
        #         init = so.initiation()
        #         eff = so.effect()

        #         ax: plt.Axes = axs[i]
        #         # est.fit_predict()
        #         field += labels
        #         # tmp_field = est.score_samples(test_values).reshape((240, 240))

        #         # print(tmp_field)
        #         # for y in range(240):
        #         #     for x in range(240):
        #         #         res = svm.predict([[x, y]])
        #         #         field[y, x] = res[0]
        #     axs[i].imshow(field)
        #     axs[i].set_title(o.name)

    plot_clusters()
    plot_classifiers()
    plt.show()


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
        paths.append(group)
        res.append(
            (
                "{}{}{}{}".format(prefix, "combined", group, ".png"),
                ["{}{}{}".format(prefix, path, ".png") for path in paths],
            )
        )

    return res


if __name__ == "__main__":
    main()
