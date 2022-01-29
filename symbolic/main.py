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
        return np.unique(self.effect_set, axis=0)

    def initiation(self) -> LowLevelStates:
        return np.unique(self.initiation_set, axis=0)

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

    def overlay_state(self, s: LowLevelState, color=(255, 0, 0)):
        old_state = self.get_state()
        self.set_state(s)
        self.game.overlay(color, 128)
        self.set_state(old_state)

    def overlay_transition(self, start: LowLevelState, end: LowLevelState, start_color=(255, 0, 0), end_color=(0, 255, 0)):
        self.game.overlay_transition(
            (start[0], start[1]), (end[0], end[1]), start_color, end_color, 50)


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
    for i in range(500):
        res: OptionExecuteReturn = agent.options[
            rng.randint(0, 3)].execute(
            state, (lambda s: None))
        # print(pr.get_state(), res.state, res.time)
        state = res.state
        pr.draw_state(state)
        # state = res.state
        # print(pr.get_state())
        # time.sleep(0.0001)
    pr.draw_background()
    for i in range(4):
        print(agent.options[i].name)
        init = agent.options[i].initiation()
        eff = agent.options[i].effect()
        for s in init:
            pr.overlay_state(s, (0, 255, 0))
        for s in eff:
            pr.overlay_state(s, (255, 0, 0))
        for e in agent.options[i].transitions:
            pr.overlay_transition(e.start, e.dest, (255, 0, 0), (0, 255, 0))
    input()


if __name__ == "__main__":
    main()
