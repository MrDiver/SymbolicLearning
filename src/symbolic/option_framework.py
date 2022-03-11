import time
from collections import namedtuple
from typing import Callable, List, Tuple

import numpy as np
import numpy.typing as nptype
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# trunk-ignore(mypy/attr-defined)
from typing_extensions import Self

from symbolic.base import ProbOptionTransition
from symbolic.tools import Projection, StateBounds, get_color, power_set
from symbolic.typing import (
    LowLevelState,
    LowLevelStates,
    StateConsumer,
    StateProbability,
    StateTest,
    StateTransform,
)

"""[summary]
    state - refers to the new state after the option is executed
    executed - true iff the option could be executed
    i.e. state was in initiation set
"""
OptionExecuteReturn = namedtuple("OptionExecuteReturn", ["state", "executed", "time"])
OptionTransition = namedtuple("OptionTransition", ["start", "dest", "states"])


class ProjectionOption:
    count = 0

    def __init__(
        self,
        name: str,
        policy: StateTransform,
        init_test: StateTest,
        termination: StateConsumer,
        state_bounds: StateBounds,
        pool,
    ) -> None:
        self.index = ProjectionOption.count
        ProjectionOption.count += 1
        self.name: str = name
        self._init_test: StateTest = init_test
        self._policy: StateTransform = policy
        self._termination: StateConsumer = termination
        self.state_bounds: StateBounds = state_bounds
        self.pool = pool

        self.transitions: List[ProbOptionTransition] = []
        self.initation_data: List[Tuple[LowLevelState, bool]] = []
        self.init_projection = Projection([], [], state_bounds)
        self.eff_projection = Projection([], [], state_bounds)

    def _add_transition(
        self,
        start: LowLevelState,
        dest: LowLevelState,
        states: LowLevelStates,
        prob: float,
    ):
        self.transitions.append(ProbOptionTransition(start, dest, states, prob))

    def _add_init_state(self, state: LowLevelState) -> None:
        self.init_projection.add([state])

    def _add_effect_state(self, state: LowLevelState) -> None:
        self.eff_projection.add([state])

    def _add_init_sample(self, state: LowLevelState, in_init: bool) -> None:
        self.initation_data.append(state, in_init)

    def effect(self) -> Projection:
        return self.eff_projection

    def initiation(self) -> Projection:
        return self.init_projection

    def remainder(self, states: LowLevelStates) -> Projection:
        return Projection(states, self.effect_mask(), self.state_bounds)

    # TODO
    def weak_subgoal_condition(self) -> bool:
        pass

    # TODO
    def strong_subgoal_condition(self) -> bool:
        pass

    def effect_mask_bool(self, eps=1e-3) -> nptype.NDArray[np.bool8]:
        modified = np.zeros(len(self.state_bounds.bounds)) != 0
        for t in self.transitions:
            modified |= np.abs(t.start - t.dest) > eps
        return modified

    def effect_mask(self, eps=1e-3) -> nptype.NDArray[np.int64]:
        modified = self.effect_mask_bool(eps)
        return np.arange(len(modified), dtype=np.int64)[modified]

    def empty(self) -> bool:
        return self.init_projection.is_empty() or self.eff_projection.is_empty()

    def init_test(self, states: LowLevelStates) -> nptype.NDArray[np.floating]:
        return self._init_test(states)

    def termination_test(self, state: LowLevelState) -> float:
        return self._termination(state)

    def execute_policy(self, state: LowLevelState) -> LowLevelState:
        return self._policy(state)

    def execute(
        self, state: LowLevelState, update_callback: Callable[[LowLevelState], None]
    ) -> OptionExecuteReturn:
        # print(state)
        if not self.init_test([state])[0] > 0:
            self._add_init_sample(state, False)
            return OptionExecuteReturn(state, False, 0)
        terminated: bool = False
        start_state: LowLevelState = state
        start_time: float = time.time()
        intermediate_states = []
        while not terminated:
            intermediate_states.append(state)
            state: LowLevelState = self.execute_policy(state)
            terminated: bool = self.termination_test(state) > 0
            update_callback(state)

        self._add_transition(start_state, state, intermediate_states, 0)

        self._add_init_sample(start_state, True)
        self._add_init_state(start_state)

        self._add_effect_state(state)
        self._add_init_sample(state, self.init_test([state])[0] > 0)

        return OptionExecuteReturn(state, True, time.time() - start_time)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class OptionPool:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.pool: List[ProjectionOption] = []

    def add_option(
        self,
        name: str,
        policy: StateTransform,
        init_test: StateTest,
        termination: StateConsumer,
        state_bounds: StateBounds,
    ) -> ProjectionOption:
        option: ProjectionOption = ProjectionOption(
            name, policy, init_test, termination, state_bounds, self
        )
        self.pool.append(option)
        return option

    def get_options(self) -> List[ProjectionOption]:
        return self.pool


class Option:
    count: int = 0

    def __init__(
        self: Self,
        name: str,
        policy: StateTransform,
        init_test: StateTest,
        termination: StateConsumer,
        state_bounds: StateBounds,
    ) -> None:
        self.index = Option.count
        Option.count += 1
        self.name = name
        self.init_test = init_test
        self.policy = policy
        self.termination = termination
        self.state_bounds = state_bounds

        self.initiation_set: List[LowLevelState] = []
        self.initiation_set_complement: List[LowLevelState] = []
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

    def remainder(self, states: LowLevelStates, resolution=10) -> LowLevelStates:
        return Projection.new(states, self.mask(), self.state_bounds).to_array(
            resolution
        )

    def remainder_projection(self, states: LowLevelStates) -> Projection:
        return Projection.new(states, self.mask(), self.state_bounds)

    # Not a complete mathematical check but good enough
    def weak_subgoal_condition(self, options: List[Self]) -> bool:
        # print(f"Checking weak subgoal for {self.name}")
        failed = []
        for option in options:
            if option == self:
                continue
            unique_effect = np.unique(self.effect(), axis=0)
            for eff_state in unique_effect:
                image_check = option.init_test([eff_state]).all()
                effect_check = option.init_test(unique_effect).all()
                if not image_check == effect_check:
                    failed.append(option)
                    break
        if len(failed) > 0:
            # print(f" - Condition does not hold with {failed}")
            return False

        # print(" - Condition Holds")
        return True

    def strong_subgoal_condition(self, options: List[Self]) -> bool:
        # print("Checking strong subgoal for {}".format(self.name))
        failed = []

        # calculating sub effect sets for every unique starting point
        groups = {np.array_repr(t.start): [] for t in self.transitions}
        for t in self.transitions:
            groups[np.array_repr(t.start)].append(t.dest)

        for o in options:
            if o == self:
                continue
            for sub_image in groups.values():
                arr0 = np.unique(sub_image, axis=0)
                arr1 = np.unique(self.effect(), axis=0)
                if not np.array_equal(arr0, arr1):
                    failed.append(o)
                    break
        if len(failed) > 0:
            # print(" - Condition does not hold with {}".format(failed))
            return False
        # print(" - Condition Holds")
        return True

    def mask_bool(self, eps=1e-3) -> nptype.NDArray[np.bool8]:
        modified = np.zeros(len(self.state_bounds.bounds)) != 0
        for t in self.transitions:
            modified |= np.abs(t.start - t.dest) > eps
        return modified

    def mask(self, eps=1e-3) -> nptype.NDArray[np.int64]:
        modified = self.mask_bool(eps)
        return np.arange(len(modified), dtype=np.int64)[modified]

    def empty(self) -> bool:
        return len(self.initiation_set) == 0 or len(self.effect_set) == 0

    def execute(
        self, state: LowLevelState, update_callback: Callable[[LowLevelState], None]
    ) -> OptionExecuteReturn:
        # print(state)
        if not self.init_test([state])[0] > 0:
            self.initiation_set_complement.append(state)
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

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


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
            f"{option.name}[{index}]",
            option.policy,
            option.init_test,
            option.termination,
            option.state_bounds,
        )
        self.index = index
        self.option = option
        self.init_probability = init_probability
        self.eff_probability = eff_probability
        for i, (start, dest) in enumerate(zip(initiation_set, effect_set)):
            self._add_transition(start, dest, transitions[i].states)

    # def init_probability(states):
    #     pass

    # def eff_probability(states):
    #     pass


class PropositionSymbol:
    def __init__(self, name: str, grounding_set: Projection or nptype.NDArray) -> None:
        self.name = name
        self.grounding_set: Projection = Projection.as_projection(grounding_set)

    def p_test(self, states: LowLevelStates) -> bool:
        return np.array_equal(
            np.unique(states, axis=0), self.grounding_set.intersect(states).to_array()
        )

    # def p_and(self, other: Self):
    #     return PropositionSymbol(
    #         f"({self.name} & {other.name})",
    #         self.grounding_set.intersect(other.grounding_set),
    #     )

    # def p_or(self, other: Self):
    #     return PropositionSymbol(
    #         f"({self.name} | {other.name})",
    #         (lambda x: self.p_test(x) or other.p_test(x)),
    #     )

    # def p_not(self):
    #     return PropositionSymbol(f"not_{self.name}", (lambda x: not self.p_test(x)))

    # def p_null(self) -> bool:
    #     return len(self.grounding_set.states) == 0


def partition_options(options: List[Option]) -> List[SubgoalOption]:
    """
    Partition options with DBSCAN into subgoals
    Train SVM for init classifier
    """
    scaler = StandardScaler()
    for o in options:
        if o.empty():
            print("Skipping option {}", o.name)
            continue
        scaler.partial_fit(o.initiation())
        scaler.partial_fit(o.effect())
    print("Data Info -", scaler.scale_)
    subgoals: List[SubgoalOption] = []
    estimators = []
    for i, option in enumerate(options):
        if option.empty():
            print("Skipping option {}", option.name)
            continue
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
                scaler = StandardScaler()
                svm = SVC(
                    # class_weight="balanced",
                    probability=True,
                    # C=10.0,
                    shrinking=False,
                    cache_size=500
                    # decision_function_shape="ovo",
                    # break_ties=True,
                )

                pipe = make_pipeline(scaler, svm)

                def test(states: LowLevelStates) -> nptype.NDArray[np.float64]:
                    probability = pipe.predict_proba(states)
                    predicted_class = pipe.predict(states)
                    return predicted_class * (1 - probability[:, 0]) * probability[:, 1]

                return pipe, test

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
        initiation_set_complement = np.append(initiation_set_complement, so.effect())
        initiation_set_complement = np.append(
            initiation_set_complement, so.initiation_set_complement
        )
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
            np.ones(len(initiation_set)), np.zeros(len(initiation_set_complement))
        )
        # Generate Data
        data = np.append(initiation_set, initiation_set_complement, axis=0)

        # generate weights
        ratio = len(initiation_set_complement) / len(initiation_set)
        weight = np.append(
            np.ones(len(initiation_set)) * ratio,
            np.ones(len(initiation_set_complement)),
        )
        print(
            so.name,
            "Positive:",
            len(initiation_set),
            "Negative:",
            len(initiation_set_complement),
            ratio,
        )
        estimator.fit(data, labels, svc__sample_weight=weight)

    return subgoals


class Factor:
    def __init__(self, indices: nptype.NDArray[np.int64], options: List[Option]):
        self.indices = indices
        self.options = options

    def __str__(self) -> str:
        return f"F:{self.indices} - {self.options}"

    def __repr__(self) -> str:
        return self.__str__()

    def is_independent(self, option: Option, factors: List[Self]) -> bool:
        states = option.effect()
        other_ids = [i for f in factors for i in f.indices if i not in self.indices]
        inside = Projection.new(states, self.indices, option.state_bounds)
        outside = Projection.new(states, other_ids, option.state_bounds)
        partitioned_states = inside.intersect(outside)
        # print("Inside ID", inside.ids, len(inside.states))
        # print("Outside ID", outside.ids, len(outside.states))
        # print("Part ID", partitioned_states.ids)

        import matplotlib.pyplot as plt
        import pandas as pd

        length_states = len(option.state_bounds.bounds)

        def plot_stuff():
            fig, axs = plt.subplots(length_states, length_states, figsize=(12, 12))
            df_inside = pd.DataFrame(inside.to_array())
            df_outside = pd.DataFrame(outside.to_array())
            df_intersect = pd.DataFrame(partitioned_states.to_array())
            df_eff = pd.DataFrame(states)
            for i, ax_y in enumerate(axs):
                for j, ax in enumerate(ax_y):
                    df_inside.plot.scatter(
                        x=i, y=j, ax=ax, c="orange", alpha=0.2, label="inside"
                    )
                    df_outside.plot.scatter(
                        x=i, y=j, ax=ax, c="blue", alpha=0.2, label="outside"
                    )
                    df_intersect.plot.scatter(
                        x=i, y=j, ax=ax, c="green", alpha=0.2, label="intersect"
                    )
                    df_eff.plot.scatter(
                        x=i, y=j, ax=ax, c="red", alpha=0.2, label="effect"
                    )
            plt.legend()
            plt.show()

        # plot_stuff()
        # print(
        #     len(np.unique(states, axis=0)),
        #     len(np.unique(partitioned_states.to_array(), axis=0)),
        # )
        return np.array_equal(
            np.unique(states, axis=0), np.unique(partitioned_states.to_array(), axis=0)
        )


# Very unclear algorithm TODO: fix this
def calculate_factors(options: List[Option]) -> List[Factor]:
    print("Computing Factors")
    masks = np.array([o.mask_bool() for o in options])
    state_modified_by = []

    for i in range(len(masks[0])):
        modifies = [o for o, p in zip(options, masks[:, i]) if p]
        state_modified_by.append(modifies)

    factors = []
    used = []
    for i, mod in enumerate(state_modified_by):
        if i in used:
            continue
        state_ids = [i]
        used.append(i)
        # print("-", i, mod)
        for _j, mod2 in enumerate(state_modified_by[i + 1 :]):
            j = _j + i + 1
            # print("\t", j, mod2)
            if j not in used and np.array_equal(mod, mod2):
                # print("\t True")
                state_ids.append(j)
                used.append(j)
        f = Factor(state_ids, mod)
        factors.append(f)
    return factors


def get_factors_for_option(option: Option, factors: List[Factor]) -> List[Factor]:
    tmp = []
    for f in factors:
        if option in f.options:
            tmp.append(f)

    return tmp


def calculate_propositions(options: List[Option]) -> List[PropositionSymbol]:
    print("Calculate Propositions")
    propositions = []
    factors = calculate_factors(options)
    for option in options:
        factors_oi = get_factors_for_option(option, factors)
        independent_factors = []
        independent_ids = []
        remaining_factors = []
        for factor in factors_oi:
            if factor.is_independent(option, factors_oi):
                independent_factors.append(factor)
                independent_ids = np.append(independent_ids, factor.indices)
            else:
                remaining_factors.append(factor)
        independent_ids = np.asarray(independent_ids, dtype=np.int64)
        print("Independent", independent_factors)
        print("Dependent", remaining_factors)

        effect_remaining = Projection.new(
            option.effect(), independent_ids, option.state_bounds
        )

        for factor in independent_factors:
            out_ids = []
            for other_factor in factors_oi:
                if other_factor != factor:
                    out_ids = np.append(out_ids, other_factor.indices)
            out_ids = np.asarray(out_ids, dtype=np.int64)
            print(
                f"Creating P(E({option.name}),{out_ids}) for independent Factor {factor}"
            )
            proposition = PropositionSymbol(
                f"P(E({option.name}),{out_ids})",
                Projection.new(option.effect(), out_ids, option.state_bounds),
            )
            propositions.append(proposition)

        for factor_set in power_set(remaining_factors):
            out_ids = []
            for factor in factor_set:
                out_ids = np.append(out_ids, factor.indices)
            out_ids = np.asarray(out_ids, dtype=np.int64)
            print(
                f"Creating P(E_r({option.name}),{out_ids}) for independent Factor {factor_set}"
            )
            proposition = PropositionSymbol(
                f"P(E_r({option.name}),{out_ids})", effect_remaining.project(out_ids)
            )
            propositions.append(proposition)
    return propositions


# TODO Calculate operator description
# TODO abstract subgoal partitioning


class Node:
    id_counter = 0

    def __init__(self, name: str):
        self.name = name
        self._id: int = Node.id_counter
        Node.id_counter += 1

    @property
    def id(self):
        return self._id

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id


class Edge:
    edge_counter = 0

    def __init__(self, a: Node, b: Node):
        self.a = a
        self.b = b
        self._id = Edge.edge_counter
        Edge.edge_counter += 1

    @property
    def id(self):
        return self._id

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id


class Graph:
    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.node_properties = {}
        self.edge_properties = {}

    def add_node_property(self, name: str) -> dict:
        self.node_properties[name] = {}
        return self.node_properties[name]

    def get_node_property(self, name: str) -> dict:
        return self.node_properties[name]

    def add_edge_property(self, name: str) -> dict:
        self.edge_properties[name] = {}
        return self.edge_properties[name]

    def get_edge_property(self, name: str) -> dict:
        return self.edge_properties[name]

    def add_node(self, name: str) -> Node:
        n = Node(name)
        self.nodes.append(n)
        return n

    def add_edge(self, a: Node, b: Node):
        e = Edge(a, b)
        self.edges.append(e)
        return e


def generate_planning_graph(
    start_states: LowLevelStates, subgoals: List[SubgoalOption]
) -> Graph:
    # Generate Graph and add properties
    g = Graph()
    node_names = g.add_node_property("label")
    # edge_names = g.add_edge_property("label")
    node_color = g.add_node_property("fillcolor")
    edge_color = g.add_edge_property("color")

    # Add names to nodes for the effect sets of the subgoals
    max_option = 0
    for i, so in enumerate(subgoals):
        v = g.add_node(so.name)
        if so.option.name == "Goal":
            node_names[v] = "Goal"
        else:
            node_names[v] = so.name
        node_color[v] = get_color(so.option.index + 1)
        max_option = max(max_option, so.option.index)

    # Generate Start
    start_node = g.add_node("Start")
    node_names[start_node] = "Start"
    node_color[start_node] = get_color(max_option + 2)
    for j, so in enumerate(subgoals):
        probs = so.init_probability(start_states)
        checked = np.all(probs > 0.01)
        if checked:
            e = g.add_edge(start_node, g.nodes[j])
            # edge_names[e] = "Start"
            edge_color[e] = get_color(max_option + 2)

    # Generate All Edges
    for i, s1 in enumerate(subgoals):
        for j, s2 in enumerate(subgoals):
            v1 = g.nodes[i]
            v2 = g.nodes[j]
            probs = s2.init_probability(s1.effect())
            checked = np.all(probs > 0.01)
            if checked:
                e = g.add_edge(v1, v2)
                # edge_names[e] = subgoals[i].name
                edge_color[e] = get_color(s1.option.index + 1)

    return g


class Plan:
    def __init__(self, seq: List[Option]) -> None:
        self.seq = seq

    # Plan feasability definition 3
