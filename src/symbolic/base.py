import time
from abc import abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, TypeVar

import numpy as np
from typing_extensions import TypeAlias

from symbolic.tools import StateBounds
from symbolic.typing import LowLevelState, LowLevelStates


@dataclass
class ProbOptionTransition:
    start: LowLevelState
    dest: LowLevelState
    states: LowLevelStates
    prob: float


@dataclass
class OptionExecuteReturn:
    state: LowLevelState
    executed: bool
    time: float


class BasePhysicalOption:
    def __init__(self, name: str) -> None:
        self.name = name
        self.transitions: List[ProbOptionTransition] = []
        self.initation_data: List[Tuple[LowLevelState, bool]] = []

    def _add_transition(
        self,
        start: LowLevelState,
        dest: LowLevelState,
        states: LowLevelStates,
        prob: float,
    ) -> None:
        transition: ProbOptionTransition = ProbOptionTransition(
            start, dest, states, prob
        )
        self.transitions.append(transition)

    def _add_init_sample(self, state: LowLevelState, in_init: bool) -> None:
        self.initation_data.append((state, in_init))

    @abstractmethod
    def init_test(self, state: LowLevelState) -> bool:
        raise NotImplementedError

    @abstractmethod
    def termination_test(self, state: LowLevelState) -> float:
        raise NotImplementedError

    @abstractmethod
    def execute_policy(self, state: LowLevelState) -> None:
        raise NotImplementedError

    T = TypeVar("T", bound="BaseEnvironment")

    def execute(
        self,
        env: T,
        update_callback: Optional[Callable[[], None]] = None,
    ) -> OptionExecuteReturn:
        # print(state)
        state: LowLevelState = env.get_state()
        if not self.init_test(state):
            self._add_init_sample(state, False)
            return OptionExecuteReturn(state, False, 0)
        terminated: bool = False
        start_state: LowLevelState = state
        start_time: float = time.time()
        intermediate_states = []
        while not terminated:
            intermediate_states.append(state)
            self.execute_policy(state)
            state = env.get_state()
            terminated = self.termination_test(state) > 0
            if update_callback is not None:
                update_callback()

        self._add_transition(start_state, state, np.asarray(intermediate_states), 0)
        self._add_init_sample(start_state, True)
        self._add_init_sample(state, self.init_test(state))

        return OptionExecuteReturn(state, True, time.time() - start_time)


RawData: TypeAlias = Any


class BaseEnvironmentTransformer:
    def __init__(self) -> None:
        pass

    def transform_data(self, data: RawData) -> LowLevelState:
        raise NotImplementedError

    def get_state_bounds(self) -> StateBounds:
        raise NotImplementedError


class BaseEnvironment:
    """Abstract class for Environment abstraction layer"""

    def __init__(self, transformer: BaseEnvironmentTransformer) -> None:
        self.transformer = transformer

    def sense_state(self) -> RawData:
        raise NotImplementedError

    def get_state(self) -> LowLevelState:
        return self.transformer.transform_data(self.sense_state())

    def needs_reset(self) -> bool:
        raise NotImplementedError

    def execute_option(self, option: BasePhysicalOption) -> None:
        option.execute(self)
