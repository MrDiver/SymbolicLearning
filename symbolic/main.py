"""
Main Module
"""
import time
from collections import namedtuple
from typing import Callable, List, TypeAlias, Any
from Game import MiniGame, Direction, Player
import numpy as np
from nptyping import NDArray, Bool, Float
import random as rng

OptionTransition = namedtuple('OptionTransition', ['start', 'dest'])
"""[summary]
    state - refers to the new state after the option is executed
    executed - true iff the option could be executed
    i.e. state was in initiation set
"""
OptionExecuteReturn = namedtuple(
    'OptionExecuteReturn', ['state', 'executed', 'time'])


LowLevelState: TypeAlias = NDArray[Float]
LowLevelStates: TypeAlias = NDArray[Float]
SymbolicState: TypeAlias = NDArray[Bool]
StateTransform: TypeAlias = Callable[[LowLevelState], LowLevelState]
StateTest: TypeAlias = Callable[[LowLevelState], bool]
StateConsumer: TypeAlias = Callable[[LowLevelState], Any]


class Option:
    def __init__(self, name: str,
                 policy: StateTransform,
                 init_test: StateTest,
                 termination: StateConsumer) -> None:
        self.name = name
        self.init_test = init_test
        self.policy = policy
        self.termination = termination

        self.initiation_set: List[LowLevelState] = []
        self.effect_set: List[LowLevelState] = []
        self.transitions: List[OptionTransition] = []

    def _add_transition(self, start: LowLevelState, dest: LowLevelState):
        self.initiation_set.append(start)
        self.effect_set.append(dest)
        self.transitions.append(OptionTransition(start, dest))

    def effect(self) -> LowLevelStates:
        return np.array(self.effect_set)

    def initiation(self) -> LowLevelStates:
        return np.array(self.initiation_set)

    def execute(self, state: LowLevelState,
                update_callback: Callable[[LowLevelState], None]) -> OptionExecuteReturn:
        if self.init_test(state):
            return OptionExecuteReturn(state, False, 0)
        terminated = False
        start_state = state
        start_time = time.time()
        while not terminated:
            state = self.policy(state)
            terminated = self.termination(state) > 0
            update_callback(state)

        self._add_transition(start_state, state)

        return OptionExecuteReturn(state, True, time.time()-start_time)


class SubgoalOption(Option):
    def __init__(self, index, option, initiation_set, effect_set):
        super().__init__(str(index)+"("+option.name+")", option.policy,
                         option.init_test, option.termination)
        self.index = index
        self.option = option
        for start, dest in zip(initiation_set, effect_set):
            self._add_transition(start, dest)


def partition_options(option: Option) -> List[SubgoalOption]:
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    subgoals: List[SubgoalOption] = []

    db = DBSCAN(eps=0.1, min_samples=1).fit(
        StandardScaler().fit_transform(option.effect_set))
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    print("Partitioned", option.name, "into", n_clusters_, "clusters")
    for k in unique_labels:

        class_member_mask = labels == k

        eff_sub = option.effect()[class_member_mask & core_samples_mask]
        init_sub = option.initiation()[class_member_mask & core_samples_mask]
        subgoals.append(SubgoalOption(k, option, init_sub, eff_sub))

    return subgoals


class PropositionSymbol:
    def __init__(self, name: str,
                 test: StateTest) -> None:
        self.name = name
        self.test = test
        self.low_level_states: List[LowLevelState] = []

    def p_test(self, state: LowLevelState):
        res = self.test(state)
        if res:
            self.low_level_states.append("Hallo")
        return res

    def p_and(self, other):
        return __make_symbol("("+self.name+"&"+other.name+")",
                             (lambda x: self.test(x) and other.test(x)))

    def p_or(self, other):
        return __make_symbol("("+self.name+"|"+other.name+")",
                             (lambda x: self.test(x) or other.test(x)))

    def p_not(self):
        return __make_symbol("not_"+self.name, (lambda x: not self.test(x)))

    def p_null(self) -> bool:
        return len(self.low_level_states) == 0


def __make_symbol(name: str,
                  test: StateTest) -> PropositionSymbol:
    return PropositionSymbol(name, test)


class Plan:
    def __init__(self, seq: List[Option]) -> None:
        self.seq = seq

    # Plan feasability definition 3


class Playroom:
    def __init__(self) -> None:
        self.game = MiniGame()

    def get_state(self) -> LowLevelState:
        return np.array([self.game.player.pos.x,
                         self.game.player.pos.y])

    def set_state(self, s: LowLevelState):
        self.game.player.set_state(s[0], s[1])

    def execute_no_change(self, s: LowLevelState, f: Callable[[], Any]) -> LowLevelState:
        old_state = self.get_state()
        self.set_state(s)
        f()
        new_state = self.get_state()
        self.set_state(old_state)
        return new_state

    def execute_no_change_return(self, s: LowLevelState, f: Callable[[], Any]) -> LowLevelState:
        old_state = self.get_state()
        self.set_state(s)
        res = f()
        self.set_state(old_state)
        return res

    def update(self) -> None:
        self.game.draw()

    def draw_state(self, s: LowLevelState):
        old_state = self.get_state()
        self.set_state(s)
        self.game.draw()
        self.set_state(old_state)

    def draw_background(self):
        self.game.draw_background()

    def overlay_state(self, s: LowLevelState, color=(255, 0, 0), alpha=10):
        old_state = self.get_state()
        self.set_state(s)
        self.game.overlay(color, alpha)
        self.set_state(old_state)

    def overlay_transition(self, start: LowLevelState, end: LowLevelState, start_color=(255, 0, 0), end_color=(0, 255, 0), alpha=10):
        self.game.overlay_transition(
            (start[0], start[1]), (end[0], end[1]), start_color, end_color, alpha)


class Agent:
    def __init__(self, options: List[Option]):
        self.options: List[Option] = options


class PlayroomAgent(Agent):
    def __init__(self, playroom: Playroom, player: Player):
        super().__init__([Option("MoveLeft",
                                 self._move(Direction.LEFT),
                                 self._move_test(Direction.LEFT),
                                 self._move_terminate(Direction.LEFT)
                                 ),
                          Option("MoveRight",
                                 self._move(Direction.RIGHT),
                                 self._move_test(Direction.RIGHT),
                                 self._move_terminate(Direction.RIGHT)
                                 ),
                          Option("MoveUp",
                                 self._move(Direction.UP),
                                 self._move_test(Direction.UP),
                                 self._move_terminate(Direction.UP)
                                 ),
                          Option("MoveDown",
                                 self._move(Direction.DOWN),
                                 self._move_test(Direction.DOWN),
                                 self._move_terminate(Direction.DOWN)
                                 )
                          ]
                         )
        self.player = player
        self.playroom = playroom

    def _move(self, direction: Direction) -> StateTransform:
        def _move_dir(s: LowLevelState) -> LowLevelState:
            return self.playroom.execute_no_change(s, (lambda: self.player.move(direction)))
        return _move_dir

    def _move_test(self, direction: Direction) -> StateTest:
        def _move_test_dir(s: LowLevelState) -> bool:
            return self.playroom.execute_no_change_return(s, (lambda: self.player.collide(direction)))
        return _move_test_dir

    def _move_terminate(self, direction: Direction) -> StateConsumer:
        def _move_terminate_dir(s: LowLevelState) -> int:
            return 1 if self.playroom.execute_no_change_return(s, (lambda: self.player.collide(direction))) else 0
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
                recurse_options(res.state, options, depth-1)
            # pr.draw_state(res.state)

    def random_options(state: LowLevelState, options: List[Option], depth: int):
        i = 0
        while i < depth:
            res = options[rng.randint(0, len(options)-1)
                          ].execute(state, (lambda s: None))
            # pr.draw_state(res.state)
            state = res.state
            i += 1

    random_options(state, agent.options, 1000)
    # pr.draw_background()
    # for i in range(4):
    #     print(agent.options[i].name)
    #     init = agent.options[i].initiation()
    #     eff = agent.options[i].effect()
    #     for s in init:
    #         pr.overlay_state(s, (0, 255, 0), 255/len(init)+5)
    #     for s in eff:
    #         pr.overlay_state(s, (255, 0, 0), 255/len(eff)+5)
    #     for e in agent.options[i].transitions:
    #         pr.overlay_transition(
    #             e.start, e.dest, (255, 0, 0), (0, 255, 0), 255/len(init)+5)
    pr.game.destroy()
    plot_options(agent.options)


def plot_options(options: List[Option]):
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import KernelDensity
    from sklearn.preprocessing import StandardScaler

    def plot_samples():
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
        axs = axs.flatten()
        for i, o in enumerate(options):
            init = o.initiation()
            eff = o.effect()
            axs[i].scatter(init[:, 0], init[:, 1], label="init")
            axs[i].scatter(eff[:, 0], eff[:, 1], label="eff")
            axs[i].set_title(o.name)
        plot.show()

    def plot_clusters():
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
        axs = axs.flatten()
        for i, o in enumerate(options):
            subgoals = partition_options(o)
            for so in subgoals:
                init = so.initiation()
                eff = so.effect()
                axs[i].scatter(init[:, 0], init[:, 1],
                               label=so.name, c='C'+str(so.index), marker='x')
                axs[i].scatter(eff[:, 0], eff[:, 1],
                               label=so.name, c='C'+str(so.index))
            axs[i].set_title(o.name)
        plt.show()

    def plot_classifiers():
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
        axs = axs.flatten()
        for i, o in enumerate(options):
            subgoals = partition_options(o)
            field = np.zeros((240, 240))
            test_values = np.transpose(np.meshgrid(
                np.arange(0, 240, 1), np.arange(0, 240, 1))).reshape((240*240, 2))
            print(test_values)
            for so in subgoals:
                init = so.initiation()
                eff = so.effect()
                est = KernelDensity(kernel='gaussian')
                est.fit(eff)
                tmp_field = est.score_samples(
                    test_values).reshape((240, 240))

                print(tmp_field)
                # for y in range(240):
                #     for x in range(240):
                #         res = svm.predict([[x, y]])
                #         field[y, x] = res[0]
            axs[i].imshow(field)
            axs[i].set_title(o.name)
        plt.show()

    plot_classifiers()


if __name__ == "__main__":
    main()
