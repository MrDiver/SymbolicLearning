"""The agent module holds all classes and types necessary for an agent to be created

"""
from abc import abstractmethod
from typing import Any, Callable, List, Tuple

import numpy as np
import numpy.typing as nptype

# trunk-ignore(mypy/attr-defined)
from typing_extensions import Self

from symbolic.game import Direction, MiniGame, Player
from symbolic.option_framework import (
    LowLevelState,
    LowLevelStates,
    Option,
    OptionTransition,
    StateConsumer,
    StateTest,
    StateTransform,
)
from symbolic.tools import StateBounds


class Playroom:
    """The playroom class is a wrapper around MiniGame which holds the environment for testing"""

    def __init__(self: Self) -> None:
        self.game: MiniGame = MiniGame()
        self.start_state: LowLevelState = self.get_state()
        tmp = [(0, 300), (0, 300)]
        for _ in self.game.get_key_states():
            tmp.append((0, 1))
        self.key_end_index = len(tmp)

        self.state_bounds = StateBounds(tmp)

    def get_state(self: Self) -> LowLevelState:
        """Returns the current state of the game

        Args:
            self (Self): instance of object

        Returns:
            LowLevelState: the current low-level state of the internal MiniGame
        """
        current_state = []
        for val in self.game.player.get_position():
            current_state.append(val)
        for val in self.game.get_key_states():
            current_state.append(val)
        current_state = np.array(current_state).flatten()
        # print("Getting", current_state)
        return current_state

    def set_state(self: Self, state: LowLevelState) -> None:
        """Sets the low-level state

        Args:
            s (LowLevelState): the new low-level state
        """
        # print("Setting", state)
        self.game.player.set_position(state[0], state[1])
        self.game.set_key_states(state[2 : self.key_end_index])

    def reset(self: Self) -> LowLevelState:
        """Resets the internal MiniGame returns the new state

        Returns:
            LowLevelState: the new low-level state
        """
        self.game.reset()
        self.set_state(self.start_state)
        return self.start_state

    def execute_no_change(
        self: Self, state: LowLevelState, func: Callable[[], Any]
    ) -> LowLevelState:
        """Executes a function f in the environment without changing the internal state
            and returns the new state

        Args:
            state (LowLevelState): starting state that will be set before execution of f
            func (Callable[[], Any]): the function that will be executed in the environment

        Returns:
            LowLevelState: [description]
        """
        old_state = self.get_state()
        self.set_state(state)
        func()
        new_state = self.get_state()
        self.set_state(old_state)
        return new_state

    def execute_no_change_return(
        self: Self, state: LowLevelState, func: Callable[[], Any]
    ) -> Any:
        """Executes a function f in the environment without changing the internal state
            and returns the value that is returned by f

        Args:
            state (LowLevelState): starting state that will be set before execution of f
            func (Callable[[], Any]): the function that will be executed in the environment

        Returns:
            Any: the value returned by f after execution
        """
        old_state = self.get_state()
        self.set_state(state)
        res = func()
        self.set_state(old_state)
        return res

    def draw(self: Self) -> None:
        """Draw the current state of the internal game which also updates the screen"""
        self.game.draw()

    def draw_state(self: Self, state: LowLevelState) -> None:
        """Draw the given state without changing the internal state and update the screen

        Args:
            s (LowLevelState): state to be drawn
        """
        old_state = self.get_state()
        self.set_state(state)
        self.game.draw()
        self.set_state(old_state)

    def overlay_background(self: Self) -> None:
        """Overlaying the background image on to the temporary screen without updating"""
        self.game.overlay_background()

    def overlay_state(
        self: Self,
        state: LowLevelState,
        color: Tuple[int, int, int] = (255, 0, 0),
        alpha: int = 10,
        ui_offset: int = 0,
    ) -> None:
        """Overlaying a given stat on to the temporary screen without updating
            This will overlay all interactibles

        Args:
            s (LowLevelState): the state to be drawn
            color (Tuple[int, int, int], optional): A rgb color tuple which will be used
                                                    as background color for the interactibles.
                                                    Defaults to (255, 0, 0).
            alpha (int, optional): Alpha value for the colored background. Defaults to 10.
            ui_offset (int, optional): x_offset for the ui draw for the state values. Defaults to 0.
        """
        old_state = self.get_state()
        self.set_state(state)
        self.game.overlay(color, alpha, ui_offset)
        self.set_state(old_state)

    def overlay_transition(
        self: Self,
        transition: OptionTransition,
        start_color: Tuple[int, int, int] = (255, 0, 0),
        end_color: Tuple[int, int, int] = (0, 255, 0),
        alpha: int = 25,
    ) -> None:
        """Overlaying a given transistion on to the temporary screen without updating
            This will create a colored line between start and end states of the transistion

        Args:
            transition (OptionTransition): the transistion to be drawn
            start_color (Tuple[int, int, int], optional): Starting color for gradient.
                                                          Defaults to (255, 0, 0).
            end_color (Tuple[int, int, int], optional): End color for gradient.
                                                        Defaults to (0, 255, 0).
            alpha (int, optional): alpha value for the line drawn. Defaults to 25.
        """
        self.game.overlay_transition(
            (transition.start[0], transition.start[1]),
            (transition.dest[0], transition.dest[1]),
            start_color,
            end_color,
            alpha,
        )

    def overlay_states(
        self: Self,
        states: LowLevelStates,
        color: Tuple[int, int, int],
        ui_offset: int = 0,
    ) -> None:
        """Does the same as overlay state but accepts a list of states

        Args:
            states (LowLevelStates): a list or ndarray of states
            color (Tuple[int, int, int]): color used for interactible background
            ui_offset (int, optional): x_offset for the ui draw for the state values. Defaults to 0.
        """
        for state in states:
            self.overlay_state(state, color, int(220.0 / len(states)) + 30, ui_offset)

    def overlay_text(
        self: Self,
        text: str,
        pos: Tuple[int, int],
        size: int = 24,
        color: Tuple[int, int, int] = (255, 255, 255),
        alpha: int = 255,
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        bg_alpha: int = 0,
    ) -> None:
        """Draws a text to the temporary screen without updating

        Args:
            text (str): Text to be drawn
            pos (Tuple[int, int]): the position on the screen where the text will be drawn
            size (int, optional): the fontsize that will be used. Defaults to 24.
            color (Tuple[int, int, int], optional): RGB fontcolor. Defaults to (255, 255, 255).
            alpha (int, optional): alpha value for the text. Defaults to 255.
            bg_color (Tuple[int, int, int], optional): RGB background_color. Defaults to (0, 0, 0).
            bg_alpha (int, optional): alpha value for the background. Defaults to 0.
        """
        self.game.overlay_text(text, pos, size, color, alpha, bg_color, bg_alpha)

    def update_screen(self: Self):
        """Draw the temporary screen to the actual screen and update"""
        self.game.update_screen()

    def save_image(self: Self, path: str):
        """Make a screenshot of the current screen and save it to the given path

        Args:
            path (str): path to save the screenshot
        """
        self.game.screenshot(path)


class Agent:
    """The abstract Agent class which provides the scaffolding to implement an Agent"""

    def __init__(self: Self, options: List[Option]):
        self.options: List[Option] = options
        self._done: bool = False

    def is_done(self: Self) -> bool:
        """Returns True if the agent has reached it's goal option
        Returns:
            bool: True if agent has reached goal
        """
        return self._done

    def done(self: Self) -> None:
        """Function to be called by the goal option"""
        self._done = True

    def reset(self: Self) -> LowLevelState:
        """Resets the Agent to begin a new episode
        Returns:
            LowLevelState: the new state of the agent after reset
        """
        self._done = False
        return self._reset()

    @abstractmethod
    def _reset(self: Self) -> None:
        """Internal reset function to be implemented by the subclass

        Raises:
            NotImplementedError: To be implemented by subclass agent
        """
        raise NotImplementedError


class PlayroomAgent(Agent):
    """Playroom agent which implements functionality for the Playroom environment"""

    def __init__(self: Self, playroom: Playroom):
        super().__init__(
            [
                Option(
                    "Left",
                    self._move(Direction.LEFT),
                    self._move_test(Direction.LEFT),
                    self._move_terminate(Direction.LEFT),
                    playroom.state_bounds,
                ),
                Option(
                    "Right",
                    self._move(Direction.RIGHT),
                    self._move_test(Direction.RIGHT),
                    self._move_terminate(Direction.RIGHT),
                    playroom.state_bounds,
                ),
                Option(
                    "Up",
                    self._move(Direction.UP),
                    self._move_test(Direction.UP),
                    self._move_terminate(Direction.UP),
                    playroom.state_bounds,
                ),
                Option(
                    "Down",
                    self._move(Direction.DOWN),
                    self._move_test(Direction.DOWN),
                    self._move_terminate(Direction.DOWN),
                    playroom.state_bounds,
                ),
                Option(
                    "Take",
                    self._take(),
                    self._take_test(),
                    (lambda s: 1),
                    playroom.state_bounds,
                ),
                Option(
                    "Goal",
                    (lambda s: [self.done(), s][1]),
                    (
                        lambda states: np.array(
                            [
                                (
                                    1
                                    if (s[0] > 290) and (s[1] > 290) and (s[2] > 0)
                                    else 0
                                )
                                for s in states
                            ]
                        )
                    ),
                    (lambda s: 1),
                    playroom.state_bounds,
                ),
            ]
        )
        self.playroom: Playroom = playroom
        self.player: Player = self.playroom.game.player

    def _reset(self: Self) -> LowLevelState:
        return self.playroom.reset()

    def _move(self: Self, direction: Direction) -> StateTransform:
        def _move_dir(state: LowLevelState) -> LowLevelState:
            return self.playroom.execute_no_change(
                state, (lambda: self.player.move(direction))
            )

        return _move_dir

    def _move_test(self: Self, direction: Direction) -> StateTest:
        def _move_test_dir(states: LowLevelStates) -> nptype.NDArray[np.bool8]:
            return np.array(
                [
                    self.playroom.execute_no_change_return(
                        s, (lambda: 0 if self.player.collide(direction) else 1)
                    )
                    for s in states
                ]
            )

        return _move_test_dir

    def _take(self: Self):
        def _take_(state: LowLevelState) -> LowLevelState:
            return self.playroom.execute_no_change(state, self.player.pick_key)

        return _take_

    def _take_test(self: Self) -> StateTest:
        def _take_test_(states: LowLevelStates) -> nptype.NDArray[np.bool8]:
            return np.array(
                [
                    self.playroom.execute_no_change_return(
                        s, (lambda: 1 if self.player.collide_key() else 0)
                    )
                    for s in states
                ]
            )

        return _take_test_

    def _move_terminate(self: Self, direction: Direction) -> StateConsumer:
        def _move_terminate_dir(state: LowLevelState) -> float:
            return (
                1
                if self.playroom.execute_no_change_return(
                    state, (lambda: self.player.collide(direction))
                )
                else 0
            )

        return _move_terminate_dir
