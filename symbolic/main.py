"""
Main Module
"""
import random as rng
import time
from collections import namedtuple
from typing import Any, Callable, List

import graph_tool.all as gt

# from sklearnex import patch_sklearn
# patch_sklearn()
import matplotlib.pyplot as plt
import numpy as np
from Game import Direction, MiniGame, Player, combine_images
from nptyping import Bool, Float, NDArray
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LowLevelState = NDArray[Float]
LowLevelStates = NDArray[Float]
SymbolicState = NDArray[Bool]
StateTransform = Callable[[LowLevelState], LowLevelState]
StateTest = Callable[[NDArray[LowLevelState]], NDArray[Bool]]
StateProbability = Callable[[NDArray[LowLevelState]], NDArray[Float]]
StateConsumer = Callable[[LowLevelState], Any]


"""[summary]
    state - refers to the new state after the option is executed
    executed - true iff the option could be executed
    i.e. state was in initiation set
"""
OptionExecuteReturn = namedtuple("OptionExecuteReturn", ["state", "executed", "time"])
OptionTransition = namedtuple("OptionTransition", ["start", "dest", "states"])


class Option:
    count: int = 0

    def __init__(
        self,
        name: str,
        policy: StateTransform,
        init_test: StateTest,
        termination: StateConsumer,
    ) -> None:
        self.index = Option.count
        Option.count += 1
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

    def init_test():
        pass

    def effect(self) -> LowLevelStates:
        return np.array(self.effect_set)

    def initiation(self) -> LowLevelStates:
        return np.array(self.initiation_set)

    def execute(
        self, state: LowLevelState, update_callback: Callable[[LowLevelState], None]
    ) -> OptionExecuteReturn:
        # print(state)
        if not self.init_test([state])[0] > 0:
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


# TODO: Add StateTestProb and init_probability as well as effect_probabilty functions
# TODO: To generate graph we need to call init_probabilty with all states in our initiation set
class SubgoalOption(Option):
    def __init__(
        self,
        index: int,
        option: Option,
        init_probability: StateProbability,
        eff_probability: StateProbability,
        initiation_set: LowLevelStates,
        effect_set: LowLevelStates,
        transitions: List[OptionTransition],
    ):
        super().__init__(
            "{}[{}]".format(option.name, index),
            option.policy,
            option.init_test,
            option.termination,
        )
        self.index = index
        self.option = option
        self.init_probability = init_probability
        self.eff_probability = eff_probability
        for i, (start, dest) in enumerate(zip(initiation_set, effect_set)):
            self._add_transition(start, dest, transitions[i].states)

    def init_probability(states):
        pass

    def eff_probability(states):
        pass


def partition_options(options: List[Option]) -> List[SubgoalOption]:
    """
    Partition options with DBSCAN into subgoals
    Train SVM for init classifier
    """
    scaler = StandardScaler()
    for o in options:
        scaler.partial_fit(o.initiation())
        scaler.partial_fit(o.effect())
    print("Data Info -", scaler.scale_)
    subgoals: List[SubgoalOption] = []
    estimators = []
    for i, option in enumerate(options):
        db = DBSCAN(eps=0.1, min_samples=1).fit(scaler.transform(option.effect()))
        # optics = OPTICS(min_samples=0.3)
        # optics.fit(StandardScaler().fit_transform(option.effect()))
        # print(optics.labels_)
        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True

        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        unique_labels = set(labels)
        # print(unique_labels)
        # print(labels)
        print("Partitioned", option.name, "into", n_clusters_, "clusters")
        for k in unique_labels:
            if k == -1:
                continue
            # print("Check", k)
            class_member_mask = labels == k
            # print(class_member_mask)
            # print(core_samples_mask)

            eff_sub = option.effect()[class_member_mask]
            init_sub = option.initiation()[class_member_mask]
            trans_sub = np.array(option.transitions, dtype=object)[class_member_mask]
            trans_sub = [
                x for i, x in enumerate(option.transitions) if (class_member_mask)[i]
            ]

            def create_test():
                svm = SVC(class_weight="balanced", probability=True, C=10.0)

                def test(states: NDArray[LowLevelState]) -> NDArray[Float]:
                    p = svm.predict_proba(states)
                    return (1 - p[:, 1]) * p[:, 0]

                return svm, test

            estimator, test = create_test()

            subgoals.append(
                SubgoalOption(
                    k,
                    option,
                    test,
                    (lambda s: [1] * len(s)),
                    init_sub,
                    eff_sub,
                    trans_sub,
                )
            )
            estimators.append(estimator)

    for i, (so, estimator) in enumerate(zip(subgoals, estimators)):
        initiation_set = so.initiation()
        initiation_set_complement = np.array([])
        # Gather negative examples from other states
        for j, so2 in enumerate(subgoals):
            if i == j:
                continue
            initiation_set_complement = np.append(
                initiation_set_complement, so2.initiation()
            )
        initiation_set_complement = initiation_set_complement.reshape(
            -1, initiation_set.shape[1]
        )
        # Generate labels
        labels = np.append(
            -np.ones(len(initiation_set)), np.ones(len(initiation_set_complement))
        )
        data = np.append(initiation_set, initiation_set_complement, axis=0)
        print(
            so.name,
            "Positive:",
            len(initiation_set),
            "Negative:",
            len(initiation_set_complement),
        )
        estimator.fit(data, labels)

    return subgoals


class OptionNode:
    def __init__(self, name) -> None:
        pass


def generate_planning_graph(
    start_states: NDArray[LowLevelState], subgoals: List[SubgoalOption]
) -> gt.Graph:
    # Generate Graph and add properties
    g = gt.Graph()
    node_names = g.new_vertex_property("string")
    edge_names = g.new_edge_property("string")
    node_option = g.new_vertex_property("int")
    edge_option = g.new_edge_property("int")
    g.vp.label = node_names
    g.vp.option = node_option
    g.ep.label = edge_names
    g.ep.option = edge_option

    # Add names to nodes for the effect sets of the subgoals
    g.add_vertex(len(subgoals))
    subgoal_nodes = g.get_vertices()
    max_option = 0
    for i, v in enumerate(subgoal_nodes):
        if subgoals[i].option.name == "Goal":
            node_names[v] = "Goal"
        else:
            node_names[v] = str(subgoals[i].index)
        node_option[v] = subgoals[i].option.index
        max_option = max(max_option, subgoals[i].option.index)

    # Generate Start
    start_node = g.add_vertex()
    node_names[start_node] = "Start"
    node_option[start_node] = max_option
    for j, v2 in enumerate(subgoal_nodes[:]):
        s2 = subgoals[j]
        probs = s2.init_probability(start_states)
        checked = np.all(probs > 0.01)
        if checked:
            e = g.add_edge(start_node, v2)
            edge_names[e] = "Start"
            edge_option[e] = max_option

    # Generate All Edges
    for i, v1 in enumerate(subgoal_nodes[:]):
        for j, v2 in enumerate(subgoal_nodes[:]):
            if i == j:
                continue

            s1 = subgoals[i]
            s2 = subgoals[j]
            probs = s2.init_probability(s1.effect())
            checked = np.all(probs > 0.01)
            if checked:
                e = g.add_edge(v1, v2)
                edge_names[e] = subgoals[i].name
                edge_option[e] = s1.option.index

    return g


def plot_planning_graph(g):
    import cairo

    # Plotting FUN

    colors = g.new_vertex_property("float")
    for v in g.get_vertices():
        idx = g.vp.option[v]
        colors[v] = idx / 4

    # pos = gt.random_layout(g)
    pos = gt.sfdp_layout(g, theta=0.2, multilevel=True, C=15.0, p=2)
    gt.graph_draw(
        g,
        pos,
        groups=g.vp.option,
        edge_pen_width=2,
        vertex_aspect=1,
        vertex_text_position=-1,
        # vertex_text_offset=[-0.12, 0.0],
        vertex_text_color="black",
        vertex_font_family="sans",
        vertex_font_weight=cairo.FONT_WEIGHT_NORMAL,
        vertex_font_slant=cairo.FONT_SLANT_NORMAL,
        vertex_font_size=10,
        vertex_fill_color=g.vp.option,
        vertex_color=None,
        vertex_shape=g.vp.option,
        vertex_size=20,
        output_size=(500, 500),
        vertex_text=g.vp.label,
        edge_color=g.ep.option,
        edge_marker_size=15,
        ink_scale=1,
    )


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
                    "Left",
                    self._move(Direction.LEFT),
                    self._move_test(Direction.LEFT),
                    self._move_terminate(Direction.LEFT),
                ),
                Option(
                    "Right",
                    self._move(Direction.RIGHT),
                    self._move_test(Direction.RIGHT),
                    self._move_terminate(Direction.RIGHT),
                ),
                Option(
                    "Up",
                    self._move(Direction.UP),
                    self._move_test(Direction.UP),
                    self._move_terminate(Direction.UP),
                ),
                Option(
                    "Down",
                    self._move(Direction.DOWN),
                    self._move_test(Direction.DOWN),
                    self._move_terminate(Direction.DOWN),
                ),
                Option(
                    "Goal",
                    (lambda s: s),
                    (
                        lambda states: np.array(
                            [
                                (1 if (s[0] > 290) and (s[1] > 290) else 0)
                                for s in states
                            ]
                        )
                    ),
                    (lambda s: 1),
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
        def _move_test_dir(states: NDArray[LowLevelState]) -> NDArray[Bool]:
            return np.array(
                [
                    self.playroom.execute_no_change_return(
                        s, (lambda: 0 if self.player.collide(direction) else 1)
                    )
                    for s in states
                ]
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


"""
####################################################################

                        MAIN SECTION

####################################################################
"""


def main():
    pr = Playroom()
    game = pr.game
    agent: Agent = PlayroomAgent(pr, game.player)
    start_state = pr.get_state()
    state = start_state

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

    pr.game.destroy()
    # plot_options(agent.options)
    graph = generate_planning_graph([start_state], partition_options(agent.options))
    plot_planning_graph(graph)


"""
####################################################################

                    PLOTTING SECTION

####################################################################
"""


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

    params = image_merge_params()
    newpaths = []
    for name, main, paths in params:
        for p in paths:
            newpaths.append(p)
        paths.append(main)
        combine_images(name, paths)
    combine_images("images/combinedALL.png", newpaths)


def plot_options(options: List[Option]):

    subgoals = partition_options(options)

    def plot_samples():
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
        axs = axs.flatten()
        for i, o in enumerate(options):
            init = o.initiation()
            eff = o.effect()
            axs[i].scatter(init[:, 0], init[:, 1], label="init", alpha=0.1)
            axs[i].scatter(eff[:, 0], eff[:, 1], label="eff")
            axs[i].set_title(o.name)

        fig.tight_layout()

    def plot_clusters():
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
        axs = axs.flatten()
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

        fig.tight_layout()

    def plot_classifiers():
        length = int(np.ceil(np.sqrt(len(subgoals))))
        fig, axs = plt.subplots(ncols=length, nrows=length, figsize=(12, 12))
        axs = axs.flatten()

        resolution = 100
        for ax in axs:
            ax.set_axis_off()
        for i, o in enumerate(subgoals):
            # field = np.zeros((240, 240))
            test_values = np.transpose(
                np.meshgrid(
                    np.linspace(-10, 310, resolution), np.linspace(-10, 310, resolution)
                )
            ).reshape((-1, 2))

            p = o.init_probability(test_values)
            field = p.reshape((resolution, resolution)).T
            # field[y, x] = p
            ax: plt.Axes = axs[i]
            ax.imshow(field)
            ax.set_title(o.name)
        fig.suptitle("Initiation Sets of Partitioned Subgoals")
        fig.tight_layout()

    # plot_samples()
    # plot_clusters()
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
        res.append(
            (
                "{}{}{}{}".format(prefix, "combined", group, postfix),
                "{}{}{}".format(prefix, group, postfix),
                ["{}{}{}".format(prefix, path, postfix) for path in paths],
            )
        )

    return res


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(__file__))
    os.chdir("../")  # import os

    os.chdir(os.path.dirname(__file__))
    os.chdir("../")
    main()
