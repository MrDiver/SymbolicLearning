"""
Main Module
"""


import random as rng
from typing import List

from symbolic.agent import Agent, Playroom, PlayroomAgent
from symbolic.option_framework import (
    LowLevelState,
    Option,
    calculate_factors,
    generate_planning_graph,
    partition_options,
)
from symbolic.plotting_framework import plot_planning_graph


def check_subgoal_conditions(options: List[Option]) -> None:
    """
    Checking subgoal conditions for a list of options
    Args:
        options (List[Option]): a list of options
    """
    for opt in options:
        opt.weak_subgoal_condition(options)

    for opt in options:
        opt.strong_subgoal_condition(options)


def main():
    """
    Main function for testing
    """
    playroom = Playroom()
    agent: Agent = PlayroomAgent(playroom)
    start_state = playroom.get_state()
    state = start_state
    print("MAIN")

    # currently not working
    def recurse_options(state: LowLevelState, options: List[Option], depth: int):
        if depth <= 0:
            return
        if agent.is_done():
            agent.reset()
            return
        for opt in options:
            res = opt.execute(state, (lambda s: None))

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
            # pr.overlay_state(res.state)
            # pr.update_screen()
            # print(pr.get_state())
            state = res.state
            if agent.is_done():
                state = agent.reset()

            i += 1

    random_options(state, agent.options, 1000)
    # plot_playroom(pr, agent)

    playroom.game.destroy()

    subgoals = partition_options(agent.options)
    check_subgoal_conditions(agent.options)
    check_subgoal_conditions(subgoals)

    calculate_factors(agent.options)
    # plot_options(agent.options)
    # graph = generate_planning_graph([start_state], partition_options(agent.options))
    # plot_planning_graph(graph)


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(__file__))
    os.chdir("../")  # import os

    os.chdir(os.path.dirname(__file__))
    os.chdir("../")
    main()
