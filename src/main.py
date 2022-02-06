"""
Main Module
"""


import random as rng
from typing import List

from symbolic.agent import Agent, Playroom, PlayroomAgent
from symbolic.option_framework import (
    LowLevelState,
    Option,
    PropositionSymbol,
    calculate_factors,
    calculate_propositions,
    generate_planning_graph,
    get_factors_for_option,
    partition_options,
)
from symbolic.plotting_framework import plot_planning_graph, plot_playroom


def check_subgoal_conditions(options: List[Option]) -> None:
    """
    Checking subgoal conditions for a list of options
    Args:
        options (List[Option]): a list of options
    """
    weaks = []
    strongs = []
    for opt in options:
        if opt.strong_subgoal_condition(options):
            strongs.append(opt)
    for opt in options:
        if opt.weak_subgoal_condition(options):
            weaks.append(opt)

    print(f"Is Weak Subgoal {weaks}")
    print(f"Is Strong Subgoal {strongs}")


def main() -> None:
    """
    Main function for testing
    """
    playroom = Playroom()
    agent: Agent = PlayroomAgent(playroom)
    start_state = playroom.get_state()
    state = start_state
    print("MAIN")

    # currently not working
    def recurse_options(
        state: LowLevelState, options: List[Option], depth: int
    ) -> None:
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

    def random_options(state: LowLevelState, options: List[Option], depth: int) -> None:
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
    # plot_playroom(playroom, agent)

    playroom.game.destroy()

    subgoals = partition_options(agent.options)
    # check_subgoal_conditions(agent.options)
    # check_subgoal_conditions(subgoals)

    # factors = calculate_factors(agent.options)
    # for f in factors:
    #     print(f)

    # for o in agent.options:
    #     print(o.name)
    #     factors_o = get_factors_for_option(o, factors)
    #     for f in factors_o:
    #         independent = f.is_independent(o, factors_o)
    #         print("\t", f.indices, independent)

    propositions: List[PropositionSymbol] = calculate_propositions(agent.options)
    for prop in propositions:
        print(prop.name)

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
