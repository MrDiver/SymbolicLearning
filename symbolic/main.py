
import time
from Game import MiniGame, Direction
import numpy as np
from collections import namedtuple
from typing import Callable, List


OptionTransition = namedtuple('OptionTransition', ['start', 'dest'])
"""[summary]
    state - refers to the new state after the option is executed
    executed - true iff the option could be executed i.e. state was in initiation set
"""
OptionExecuteReturn = namedtuple(
    'OptionExecuteReturn', ['state', 'executed'])


class Option:
    def __init__(self, name: str, init_test: Callable[[np.ndarray], bool], policy: Callable[[np.ndarray], np.ndarray], termination: Callable[[np.ndarray], int]) -> None:
        self.name = name
        self.init_test = init_test
        self.policy = policy
        self.termination = termination

        self.initiation_set = np.array([])
        self.transitions: List[OptionTransition] = []

    def execute(self, state: np.ndarray, update_callback: Callable[[], None]) -> OptionExecuteReturn:
        if self.init_test(state):
            return OptionExecuteReturn(state, False)

        terminated = False
        while not terminated:
            state = self.policy(state)
            terminated = self.termination(state) > 0

        return OptionExecuteReturn(state, True)


class PropositionSymbol:
    def __init__(self, name: str, test: Callable[[np.ndarray], bool]) -> None:
        self.name = name
        self.test = test

    # Add operators or, and, not, null


class Plan:
    def __init__(self, seq: List[Option]) -> None:
        self.seq = seq

    # Plan feasability definition 3


class Playroom:
    def __init__(self) -> None:
        self.game = MiniGame()

    def get_state(self) -> np.ndarray:
        return np.array([self.game.player.pos.x,
                         self.game.player.pos.y])

    def update(self) -> None:
        self.game.draw()


def main():
    pr = Playroom()
    game = pr.game
    for i in range(1000):
        game.draw()
        if not game.player.move(Direction.RIGHT):
            break
        print(pr.get_state())
        time.sleep(0.01)


if __name__ == "__main__":
    main()
