import time
from collections import namedtuple
from typing import Any, Callable, List

import numpy as np
import numpy.typing as nptype
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# trunk-ignore(mypy/attr-defined)
from typing_extensions import Self, TypeAlias

from symbolic.tools import get_color

LowLevelState: TypeAlias = nptype.NDArray[np.float64]
LowLevelStates: TypeAlias = nptype.NDArray[np.float64]
SymbolicState: TypeAlias = nptype.NDArray[np.bool8]
StateTransform = Callable[[LowLevelState], LowLevelState]
StateTest = Callable[[LowLevelStates], nptype.NDArray[np.bool8]]
StateProbability = Callable[[LowLevelStates], nptype.NDArray[np.float64]]
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
        self: Self,
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

    def remainder(self) -> LowLevelStates:
        raise NotImplementedError  # TODO: define remainder

    # Not a complete mathematical check but good enough
    def weak_subgoal_condition(self, options: List[Self]) -> bool:
        print("Checking weak subgoal for {}".format(self.name))
        failed = []
        for o in options:
            if o == self:
                continue
            for X in self.effect():
                image_check = o.init_test([X]).all()
                effect_check = o.init_test(self.effect()).all()
                if not image_check == effect_check:
                    failed.append(o)
                    break
        if len(failed) > 0:
            print(" - Condition does not hold with {}".format(failed))
            return False

        print(" - Condition Holds")
        return True

    def strong_subgoal_condition(self, options: List[Self]) -> bool:
        print("Checking strong subgoal for {}".format(self.name))
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
            print(" - Condition does not hold with {}".format(failed))
            return False
        print(" - Condition Holds")
        return True

    def mask_bool(self, eps=1e-3) -> nptype.NDArray[np.bool8]:
        if len(self.initiation_set) > 0:
            modified = np.zeros_like(self.initiation_set[0]) == 0
            for t in self.transitions:
                modified &= np.abs(t.start - t.dest) > eps
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
            "{}{}".format(option.name, index),
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

    # def init_probability(states):
    #     pass

    # def eff_probability(states):
    #     pass


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
        return "F:{} - {}".format(self.indices, self.options)

    def __repr__(self) -> str:
        return self.__str__()


# Very unclear algorithm TODO: fix this
def calculate_factors(options: List[Option]) -> List[Factor]:
    masks = np.array([o.mask_bool() for o in options])
    print("Computing Factors")
    state_modified_by = []
    for i in range(len(masks[0])):
        modifies = [o for o, p in zip(options, masks[:, i]) if p]
        # print(modifies)
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

    for f in factors:
        print(f)
    return factors


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


class PropositionSymbol:
    def __init__(self, name: str, test: StateTest) -> None:
        self.name = name
        self.test = test
        self.low_level_states: List[LowLevelState] = []

    def p_test(self, state: LowLevelState):
        res = self.test(state)
        if res:
            self.low_level_states.append(state)
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
