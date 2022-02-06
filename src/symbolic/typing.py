from typing import Any, Callable

import numpy as np
import numpy.typing as nptype
from typing_extensions import TypeAlias

LowLevelState: TypeAlias = nptype.NDArray[np.floating]
LowLevelStates: TypeAlias = nptype.NDArray[np.floating]
SymbolicState: TypeAlias = nptype.NDArray[np.bool8]
StateTransform = Callable[[LowLevelState], LowLevelState]
StateTest = Callable[[LowLevelStates], nptype.NDArray[np.floating]]
StateProbability = Callable[[LowLevelStates], nptype.NDArray[np.floating]]
StateConsumer = Callable[[LowLevelState], Any]
