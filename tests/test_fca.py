import itertools
import random

import pytest

from geoqa.fca import FormalContext


def random_context(seed, n=6, m=5, p=0.4):
    rng = random.Random(seed)
    objs = [f"g{i}" for i in range(n)]
    attrs = [f"a{j}" for j in range(m)]
    inc = [(g, a) for g in objs for a in attrs if rng.random() < p]
    return FormalContext(objs, attrs, inc)


def all_masks(k):
    return range(1 << k)


def test_paper_example_operators():
    # A small conditional context of geographic concepts.
    rows = {
        "QTP": "1001", "IMP": "1001", "LP": "1001", "WR": "1100", "WF": "0110",
        "Plain": "0000", "Landform": "0000", "ER": "1100", "EF": "0110", "China": "0000", "GD": "0110",
    }
    attrs = ["C1", "C2", "C3", "C4"]
    inc = [(g, attrs[j]) for g, r in rows.items() for j, b in enumerate(r) if b == "1"]
    K = FormalContext(rows, attrs, inc)
    X = K.obj_mask(["QTP", "IMP", "LP"])
    assert K.attrs(K.up(X)) == {"C1", "C4"}
    assert K.attrs(K.neg_up(X)) == {"C2", "C3"}
    assert K.objs(K.down(K.attr_mask(["C1", "C4"]))) == {"QTP", "IMP", "LP"}


@pytest.mark.parametrize("seed", range(40))
def test_oe_ae_concepts_match_definition(seed):
    K = random_context(seed)
    oe_brute = set()
    for X in all_masks(K.n):
        A, B = K.up(X), K.neg_up(X)
        if K.down(A) & K.neg_down(B) == X:
            oe_brute.add(X)
    assert {X for X, _ in K.oe_concepts()} == oe_brute

    ae_brute = set()
    for A in all_masks(K.m):
        X, Y = K.down(A), K.neg_down(A)
        if K.up(X) & K.neg_up(Y) == A:
            ae_brute.add(A)
    assert {A for _, A in K.ae_concepts()} == ae_brute


@pytest.mark.parametrize("seed", range(60))
def test_merged_concept_closed_form(seed):
    K = random_context(seed, n=random.Random(seed).randint(1, 7), m=random.Random(seed + 1).randint(1, 6))
    assert K.merged_concepts_bruteforce() == K.merged_concepts_fast()


@pytest.mark.parametrize("seed", range(20))
def test_c1_order_is_classical_order(seed):
    K = random_context(seed)
    c1 = [q for q, k in K.merged_concepts_fast().items() if "C1" in k]
    for (X1, Y1, A1, B1), (X2, Y2, A2, B2) in itertools.product(c1, c1):
        ext_le = (X1 & ~X2) == 0 and (Y1 & ~Y2) == 0
        int_ge = (A2 & ~A1) == 0 and (B2 & ~B1) == 0
        classical = (X1 & ~X2) == 0
        assert ext_le == int_ge == classical


def random_incomplete(seed, n=5, m=4):
    rng = random.Random(seed)
    objs = [f"g{i}" for i in range(n)]
    attrs = [f"a{j}" for j in range(m)]
    pos, neg = [], []
    for g in objs:
        for a in attrs:
            r = rng.random()
            (pos if r < 0.4 else neg if r < 0.75 else []).append((g, a))
    return FormalContext(objs, attrs, pos, neg)


@pytest.mark.parametrize("seed", range(60))
def test_incomplete_context_closed_forms(seed):
    K = random_incomplete(seed, n=random.Random(seed).randint(1, 6), m=random.Random(seed + 3).randint(1, 5))
    oe_brute = {X for X in all_masks(K.n) if K.down(K.up(X)) & K.neg_down(K.neg_up(X)) == X}
    assert {X for X, _ in K.oe_concepts()} == oe_brute
    ae_brute = {A for A in all_masks(K.m) if K.up(K.down(A)) & K.neg_up(K.neg_down(A)) == A}
    assert {A for _, A in K.ae_concepts()} == ae_brute
    assert K.merged_concepts_bruteforce() == K.merged_concepts_fast()


def test_incomplete_rejects_contradictory_pairs():
    with pytest.raises(ValueError):
        FormalContext(["g"], ["a"], [("g", "a")], [("g", "a")])
