from itertools import combinations, product

from geoqa.data import Question
from geoqa.fca import FormalContext
from geoqa.twc import Context, Params, Solver, parse_letters


def solver(**kw) -> Solver:
    s = Solver.__new__(Solver)
    s.params = Params(**kw)
    return s


def question(qtype="multiple", options=("A", "B", "C"), **kw) -> Question:
    return Question(uid="q", dataset="t", qtype=qtype, stem="stem", options={o: o for o in options}, answer=["A"],
                    **kw)


def test_regions():
    ctx = Context(["m1", "m2"], "positive", {"A": ["+", "+"], "B": ["+", "-"], "C": ["+", "?"]})
    assert Solver.regions(ctx) == {"A": "POS", "B": "NEG", "C": "BND"}


def test_multiple_answer_is_the_positive_region():
    ctx = Context(["m1"], "positive", {"A": ["+"], "B": ["-"], "C": ["+"]})
    answer, stage, _ = solver().decide(question(), ctx)
    assert (answer, stage) == (["A", "C"], "context")


def test_negative_question_answers_the_negative_region():
    ctx = Context(["m1"], "negative", {"A": ["+"], "B": ["-"], "C": ["+"]})
    answer, stage, _ = solver().decide(question(qtype="single"), ctx)
    assert (answer, stage) == (["B"], "context")


def test_two_open_units_complete_a_multiple_answer():
    ctx = Context(["m1"], "positive", {"A": ["+"], "B": ["?"], "C": ["-"]})
    answer, stage, _ = solver().decide(question(), ctx)
    assert (answer, stage) == (["A", "B"], "context")


def test_statement_combination_agreement():
    q = question(qtype="single", options=("A", "B"), statements={"1": "s1", "2": "s2", "3": "s3"},
                 option_statements={"A": ["1", "2"], "B": ["2", "3"]})
    ctx = Context(["m1"], "positive", {"1": ["+"], "2": ["+"], "3": ["-"]})
    answer, stage, _ = solver().decide(q, ctx)
    assert (answer, stage) == (["A"], "context")


def test_explanation_is_the_three_way_concept_of_the_answer():
    ctx = Context(["m1", "m2"], "positive", {"A": ["+", "+"], "B": ["-", "-"], "C": ["+", "+"]})
    assert Solver.explain(ctx, ["A", "C"]) == {"extent": ["A", "C"], "jointly_satisfied": ["m1", "m2"],
                                               "jointly_violated": []}


def test_certain_and_possible_answers():
    """Proposition 1: POS ⊆ Ans ⊆ POS ∪ BND in every completion of the undetermined judgments."""
    judgments = {"A": ["+", "?"], "B": ["?", "?"], "C": ["+", "+"], "D": ["-", "?"]}
    reg = Solver.regions(Context(["m1", "m2"], "positive", judgments))
    pos = {u for u, r in reg.items() if r == "POS"}
    bnd = {u for u, r in reg.items() if r == "BND"}
    holes = [(u, j) for u, v in judgments.items() for j, x in enumerate(v) if x == "?"]
    seen = set()
    for fill in product("+-", repeat=len(holes)):
        full = {u: list(v) for u, v in judgments.items()}
        for (u, j), x in zip(holes, fill):
            full[u][j] = x
        ans = frozenset(u for u, v in full.items() if all(x == "+" for x in v))
        assert pos <= ans <= pos | bnd
        seen.add(ans)
    assert frozenset(pos) in seen and frozenset(pos | bnd) in seen


def test_composite_attributes_are_redundant():
    """Proposition 2: one attribute per option combination leaves the extents unchanged."""
    G, O = ["g1", "g2", "g3", "g4"], ["A", "B", "C"]
    J = [("g1", "A"), ("g1", "B"), ("g2", "B"), ("g3", "A"), ("g3", "B"), ("g3", "C"), ("g4", "C")]
    atomic = FormalContext(G, O, J)
    combos = [frozenset(c) for k in range(1, 4) for c in combinations(O, k)]
    has = {g: {o for h, o in J if h == g} for g in G}
    composite = FormalContext(G, combos, [(g, S) for g in G for S in combos if S <= has[g]])
    assert atomic.extents() == composite.extents()


def test_parse_letters():
    assert parse_letters("B  \nC", list("ABC")) == ["B", "C"]
    assert parse_letters("Answer: AC", list("ABC")) == ["A", "C"]
    assert parse_letters("The answer is B", list("ABC")) == ["B"]
