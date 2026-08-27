"""
Regression tests for the scipy port.

The ``legacy_accounting=True`` cases pin the port against the numbers the
GAMSPy version produced and that the write-up published, so the port can be
verified independently of the accounting fixes.  The default-parameter cases
pin the corrected model.

Run with:  python -m pytest test_model.py -q
       or: python test_model.py
"""

import numpy as np
import pytest

from turkey_tank import Params, SolveError, VARIABLES, solve
from turkey_tank.experiments import find_balancing_rate, sweep

LEGACY = Params(legacy_accounting=True)


# ---------------------------------------------------------------------------
# The port reproduces the published GAMSPy results
# ---------------------------------------------------------------------------

def test_legacy_baseline_unemployment():
    assert solve(LEGACY, strict=False)["u"] == pytest.approx(0.210189, abs=1e-6)


def test_legacy_labour_tax_scenario():
    p = LEGACY.at(p_ben_level=0.40, tau_w=0.40, tau_c=0.18)
    assert solve(p, strict=False)["u"] == pytest.approx(0.419255, abs=1e-6)


def test_legacy_vat_scenario():
    base = solve(LEGACY.at(tau_c=0.18), strict=False)
    search = find_balancing_rate(LEGACY.at(p_ben_level=0.40, tau_c=0.18),
                                 "tau_c", target=base["lump_tax"],
                                 lo=0.02, hi=0.60)
    assert search.found
    assert search.rate == pytest.approx(0.205688, abs=1e-5)
    assert search.solution["u"] == pytest.approx(0.254005, abs=1e-6)


def test_legacy_violates_resource_constraint():
    """The bug the fixes address: savers were paid r*k twice."""
    sol = solve(LEGACY.at(tau_c=0.18), strict=False)
    expected = sol["r"] * sol["k"] - 0.5 * LEGACY.p_ben_level * sol["u"]
    assert sol.walras_gap == pytest.approx(expected, rel=1e-9)

    # The same allocation fails the check once legacy accounting is disowned.
    sol.params = sol.params.at(legacy_accounting=False)
    with pytest.raises(SolveError):
        sol.check()


# ---------------------------------------------------------------------------
# The corrected model
# ---------------------------------------------------------------------------

def test_resource_constraint_holds():
    for p in (Params(),
              Params(tau_c=0.18),
              Params(p_ben_level=0.40, tau_w=0.42),
              Params(tau_w=0.10, tau_c=0.25)):
        assert abs(solve(p).walras_gap) < 1e-9


def test_government_budget_identity():
    s = solve(Params(tau_c=0.18))
    assert s["gov_rev"] + s["lump_tax"] == pytest.approx(s["gov_exp"], abs=1e-10)


def test_fixed_vat_rate_is_higher_than_published():
    """Correcting the inflated VAT base raises the required rate."""
    base = solve(Params(tau_c=0.18))
    search = find_balancing_rate(Params(p_ben_level=0.40, tau_c=0.18),
                                 "tau_c", target=base["lump_tax"],
                                 lo=0.02, hi=0.60)
    assert search.found
    assert search.rate == pytest.approx(0.210731, abs=1e-5)
    assert search.rate > 0.205688


def test_vat_cannot_move_the_labour_market():
    """The labour block is recursive: tau_c enters none of its equations."""
    a = solve(Params(tau_c=0.18))
    b = solve(Params(tau_c=0.35))
    for key in ("u", "l_f", "w_f", "theta", "y_f", "l_i"):
        assert a[key] == pytest.approx(b[key], rel=1e-9), key


def test_labour_tax_does_move_the_labour_market():
    assert solve(Params(tau_w=0.40))["u"] > solve(Params(tau_w=0.20))["u"]


def test_benefits_raise_unemployment():
    assert solve(Params(p_ben_level=0.40))["u"] > solve(Params(p_ben_level=0.30))["u"]


def test_equilibrium_ceases_to_exist_at_high_tax():
    """Beyond a ceiling the job-creation and wage conditions share no solution."""
    with pytest.raises(SolveError):
        solve(Params(tau_w=0.60))


def test_sweep_records_infeasible_points_without_raising():
    points = sweep(Params(), "tau_w", np.linspace(0.05, 0.70, 25))
    assert any(pt.ok for pt in points)
    assert any(not pt.ok for pt in points)


def test_laffer_curve_has_an_interior_peak():
    points = sweep(Params(), "tau_w", np.linspace(0.05, 0.70, 25))
    rev = [(pt.value, pt.solution["gov_rev"]) for pt in points if pt.ok]
    peak = max(rev, key=lambda t: t[1])[0]
    assert rev[0][0] < peak < rev[-1][0]


def test_solution_is_an_actual_root():
    from turkey_tank.model import residuals
    s = solve(Params(tau_c=0.18))
    assert np.max(np.abs(residuals(s.x, s.params))) < 1e-10



# ---------------------------------------------------------------------------
# Non-existence is analytical, not a solver artefact
# ---------------------------------------------------------------------------

def _existence_ceiling(p: Params) -> float:
    """Highest tau_w admitting an interior equilibrium, in closed form.

    Eliminating the v1 system down to one quadratic in x = l_f / u gives

        Q*x**2 + G*P*x + C == 0,   Q, G, P > 0,
        C = (1 - eta) * [(w_i + b) - mpl_f * (1 - tau_w)]

    Both roots are non-positive unless C < 0, and x > 0 for any u in (0, 1),
    so an interior equilibrium exists exactly while C < 0. mpl_f is a constant
    here because the Euler equation pins r and the formal sector is CRS.
    """
    r = 1.0 / p.beta - 1.0 + p.delta
    mpl_f = (1.0 - p.alpha) * (p.alpha / r) ** (p.alpha / (1.0 - p.alpha))
    return 1.0 - (p.p_informal_prod + p.p_ben_level) / mpl_f


@pytest.mark.parametrize("benefit,expected", [(0.30, 0.45934), (0.40, 0.42330)])
def test_existence_ceiling_matches_closed_form(benefit, expected):
    assert _existence_ceiling(Params(p_ben_level=benefit)) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("benefit", [0.30, 0.40])
def test_no_equilibrium_above_the_ceiling_from_any_start(benefit):
    """Above the ceiling the failure is non-existence, not a bad initial guess."""
    ceiling = _existence_ceiling(Params(p_ben_level=benefit))
    rng = np.random.default_rng(0)
    starts = [np.exp(rng.normal(0.0, 1.2, len(VARIABLES))) for _ in range(40)]

    below = sum(_converges(Params(p_ben_level=benefit, tau_w=ceiling - 0.004), x0)
                for x0 in starts)
    above = sum(_converges(Params(p_ben_level=benefit, tau_w=ceiling + 0.004), x0)
                for x0 in starts)
    assert below > 0, "should still solve just below the ceiling"
    assert above == 0, "no start may converge above the ceiling"


@pytest.mark.parametrize("tau_w", [0.05, 0.20, 0.35, 0.42])
def test_informal_wage_is_exogenous(tau_w):
    """The closed-form ceiling needs w_i to be a parameter, not an outcome.

    v1's informal sector is linear in labour, so its marginal product is the
    constant p_informal_prod however much labour it absorbs and the Walrasian
    informal wage equals it. Informal labour adjusts on the quantity margin
    instead. If this ever stops holding, the existence condition in the README
    stops being closed-form and becomes a fixed point (as it already is in v2).
    """
    p = Params(tau_w=tau_w)
    s = solve(p)
    assert s["w_i"] == p.p_informal_prod


def test_informal_quantity_absorbs_the_tax_instead():
    lo = solve(Params(tau_w=0.05))
    hi = solve(Params(tau_w=0.42))
    assert hi["l_i"] > lo["l_i"] * 1.5          # quantity moves a lot
    assert hi["w_i"] == lo["w_i"]                # price not at all


def _converges(p: Params, x0) -> bool:
    try:
        solve(p, guess=x0, strict=False)
        return True
    except SolveError:
        return False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
