"""The power-deficit estimator, on synthetic strata with a known planted effect.

The estimator's whole claim is that pairing a player against themselves removes skill. So the
tests plant a large *skill* spread across players and a small *power* effect within each, and
assert that the recovered number is the power effect and not the skill spread — which is exactly
the failure mode that makes the naive cross-player contrast read ~-16pp instead of ~-8pp.

    PYTHONPATH=backend python -m pytest backend/tests/test_readiness_build.py
"""
from __future__ import annotations

import random

import pytest

from bsdraft.data.readiness_build import (
    DEFAULTS,
    _Arms,
    _monotone,
    _power_deficit,
    _ship_value,
    mh_combine,
)


def _arm(per_stratum):
    """{key: (mean, n)} -> the [sum, n] shape the accumulators use."""
    return {k: [mean * n, float(n)] for k, (mean, n) in per_stratum.items()}


# --------------------------------------------------------------------------- pairing


def test_recovers_planted_effect_through_a_large_skill_spread():
    rng = random.Random(11)
    effect = -0.08
    treated, control = {}, {}
    for i in range(2000):
        skill = rng.uniform(-0.30, 0.30)      # dwarfs the effect, and differs per player
        key = (f"P{i}", "Diamond")
        control[key] = [skill * 20, 20.0]
        treated[key] = [(skill + effect) * 20, 20.0]
    r = mh_combine(treated, control)
    assert r["n_strata"] == 2000
    assert r["rd"] == pytest.approx(effect, abs=1e-9)


def test_unpaired_strata_are_dropped_entirely():
    """A player with no control appearances carries no information about *their* power penalty."""
    control = _arm({("A", "Gold"): (0.0, 10)})
    treated = _arm({("A", "Gold"): (-0.05, 10), ("ghost", "Gold"): (-0.90, 10)})
    r = mh_combine(treated, control)
    assert r["n_strata"] == 1
    assert r["rd"] == pytest.approx(-0.05)


def test_same_player_different_brackets_are_separate_strata():
    control = _arm({("A", "Gold"): (0.0, 10), ("A", "Mythic"): (0.0, 10)})
    treated = _arm({("A", "Gold"): (-0.10, 10), ("A", "Mythic"): (-0.02, 10)})
    r = mh_combine(treated, control)
    assert r["n_strata"] == 2
    assert r["rd"] == pytest.approx(-0.06)          # equal weights -> plain mean


def test_thin_strata_are_downweighted_not_dropped():
    """One noisy 1-vs-1 stratum must not outvote a well-sampled one."""
    control = _arm({("solid", "Gold"): (0.0, 30), ("thin", "Gold"): (0.0, 1)})
    treated = _arm({("solid", "Gold"): (-0.05, 30), ("thin", "Gold"): (-1.0, 1)})
    r = mh_combine(treated, control)
    assert r["n_strata"] == 2
    assert -0.12 < r["rd"] < -0.05                  # pulled, but nowhere near the -1.0 outlier


def test_no_pairs_returns_nan_not_zero():
    """An empty result must not read as 'measured, and the effect is zero'."""
    r = mh_combine(_arm({("A", "Gold"): (-0.1, 5)}), {})
    assert r["n_strata"] == 0
    assert r["rd"] != r["rd"]                       # NaN


def test_null_effect_recovers_zero():
    rng = random.Random(5)
    treated, control = {}, {}
    for i in range(1500):
        skill = rng.uniform(-0.3, 0.3)
        key = (f"P{i}", "Diamond")
        control[key] = [skill * 20, 20.0]
        treated[key] = [skill * 20, 20.0]
    r = mh_combine(treated, control)
    assert r["rd"] == pytest.approx(0.0, abs=1e-9)
    assert abs(r["rd"]) <= DEFAULTS["placebo_tol"]


# --------------------------------------------------------------------------- per-player cap


def test_cap_bounds_one_players_contribution():
    cap = DEFAULTS["cap_per_arm"]
    pool = {}
    for _ in range(cap * 5):
        _Arms._add(pool, ("grinder", "Gold"), 1.0, cap)
    assert pool[("grinder", "Gold")][1] == cap


# --------------------------------------------------------------------------- shipping rule


def _ship(rd, se):
    return _ship_value(rd, se, DEFAULTS["se_haircut"], DEFAULTS["ship_grid"],
                       DEFAULTS["methodology_margin"])


def test_methodology_margin_dominates_when_the_se_is_small():
    """The real case: the corpus is large, so the SE is tiny next to the cross-specification
    spread. Keying the haircut to the SE alone would ship a number ~1pp too aggressive."""
    # measured P9: -9.44pp, se 0.45pp. 2*se = 0.9pp < the 1.5pp margin, so the margin governs.
    assert _ship(-0.0944, 0.0045) == pytest.approx(0.075)
    # measured P10: -5.87pp, se 0.36pp
    assert _ship(-0.0587, 0.0036) == pytest.approx(0.040)


def test_se_dominates_when_the_estimate_is_noisy():
    """A thin future rebuild must not get the same confidence as a well-sampled one."""
    # se 1.2pp -> 2*se = 2.4pp > the 1.5pp margin
    assert _ship(-0.0944, 0.012) == pytest.approx(0.070)


def test_ship_never_exceeds_the_measurement():
    for rd, se in ((-0.08, 0.004), (-0.04, 0.003), (-0.02, 0.001)):
        assert _ship(rd, se) <= abs(rd)


def test_ship_floors_at_zero_for_a_noisy_estimate():
    assert _ship(-0.01, 0.02) == 0.0


def test_ship_floors_at_zero_when_the_effect_is_under_the_margin():
    """An effect smaller than the methodological noise floor is not shippable at all."""
    assert _ship(-0.012, 0.0005) == 0.0


def test_ship_of_a_non_finite_estimate_is_zero():
    assert _ship(float("nan"), float("nan")) == 0.0


# --------------------------------------------------------------------------- gates


def test_monotone_requires_p9_at_least_as_bad_as_p10():
    assert _monotone({"9": {"rd": -0.08}, "10": {"rd": -0.04}})
    assert not _monotone({"9": {"rd": -0.02}, "10": {"rd": -0.06}})


def test_power_deficit_inherits_p9_below_the_floor_and_zeroes_p11():
    levels = {
        "9": {"rd": -0.08, "ship": 0.070, "shippable": True},
        "10": {"rd": -0.04, "ship": 0.030, "shippable": True},
    }
    d = _power_deficit(levels)
    assert d["11"] == 0.0
    assert d["9"] == pytest.approx(0.070)
    assert d["10"] == pytest.approx(0.030)
    assert all(d[str(p)] == d["9"] for p in range(1, 9))


def test_unshippable_level_contributes_no_penalty():
    """A level that misses the bar must score as 'no measured deficit', never as a guess."""
    levels = {
        "9": {"rd": -0.08, "ship": 0.070, "shippable": True},
        "10": {"rd": -0.04, "ship": 0.030, "shippable": False},
    }
    assert _power_deficit(levels)["10"] == 0.0
