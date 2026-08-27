"""
Regression tests for the dual labour market model (v2).

These pin the calibration, the accounting identities, and the three mechanisms
the write-up's argument rests on. If a change breaks the ranking of financing
instruments, ``test_vat_dominates_labour_tax_for_the_poor`` fails.

Run with:  python -m pytest test_dual.py -q
"""

import numpy as np
import pytest

from turkey_tank.dual import (Params, SolveError, Targets, balancing_rates,
                              calibrate, residuals, solve)

CAL, BASE = calibrate()


# --- calibration ------------------------------------------------------------

def test_hits_every_target():
    t = Targets()
    assert BASE.unemployment == pytest.approx(t.unemployment, abs=1e-9)
    assert BASE.informal_share == pytest.approx(t.informal_share, abs=1e-9)
    assert BASE.wage_gap == pytest.approx(t.wage_gap, abs=1e-9)
    assert BASE["theta"] == pytest.approx(t.tightness, abs=1e-9)
    assert BASE["p"] == pytest.approx(t.informal_price, abs=1e-9)


def test_unemployment_is_turkstat_comparable():
    """u is a share of the whole labour force, not just the formal part."""
    assert BASE["l_f"] + BASE["l_i"] + BASE["u"] == pytest.approx(1.0, abs=1e-12)
    assert 0.08 < BASE.unemployment < 0.09


def test_untargeted_moments_are_plausible():
    assert 0.25 < BASE["gov_rev"] / BASE["y_tot"] < 0.36      # revenue/GDP ~31%
    assert 0.3 < BASE["p_find"] < 0.8                          # quarterly job finding
    assert 0.0 < CAL.home_prod / BASE["w_f"] < 0.25
    assert BASE.informal_output_share < BASE.informal_share    # informal is low-productivity


# --- accounting -------------------------------------------------------------

def test_walras_holds_everywhere():
    for p in (CAL, CAL.at(tau_c=0.35), CAL.at(tau_w=0.5), CAL.at(rr=0.3)):
        assert abs(solve(p).walras_gap) < 1e-9


def test_informal_labour_income_equals_informal_output_value():
    assert BASE["w_i"] * BASE["l_i"] == pytest.approx(BASE["p"] * BASE["y_i"], rel=1e-12)


def test_government_budget_identity():
    assert BASE["gov_rev"] + BASE["lump_tax"] == pytest.approx(BASE["gov_exp"], abs=1e-12)


def test_solution_is_an_actual_root():
    assert np.max(np.abs(residuals(BASE.x, BASE.params))) < 1e-10


# --- the three mechanisms the argument needs --------------------------------

def test_vat_now_reaches_the_labour_market():
    """The v1 defect: tau_c entered no equation determining employment."""
    hi = solve(CAL.at(tau_c=0.35), guess=BASE)
    assert hi.informal_share > BASE.informal_share + 0.02
    assert hi["p"] > BASE["p"]          # informal goods get relatively dearer
    assert hi["l_f"] < BASE["l_f"]      # and formal employment shrinks
    assert hi["l_i"] > BASE["l_i"]


def test_vat_moves_quantities_not_wages():
    """Wages and tightness are invariant to VAT, and that is structural.

    The formal sector is CRS and the interest rate is pinned by beta, so
    mpl_f is a constant; free entry, Nash bargaining, the surplus definition
    and the informal-search indifference then close a six-equation block in
    (w_f, w_i, S_w, theta, p_find, p_fill) that contains no price and no
    quantity. VAT reallocates workers between formal, informal and unemployed
    without touching what any of them is paid. Worth stating plainly rather
    than letting a reader infer a wage channel that is not there.
    """
    hi = solve(CAL.at(tau_c=0.35), guess=BASE)
    for key in ("w_f", "w_i", "theta", "p_find", "p_fill", "s_w"):
        assert hi[key] == pytest.approx(BASE[key], rel=1e-9), key
    # ...while the allocation of labour does move
    assert hi["l_i"] > BASE["l_i"]
    assert hi.unemployment < BASE.unemployment


def test_labour_tax_pushes_workers_informal():
    hi = solve(CAL.at(tau_w=0.50), guess=BASE)
    assert hi.informal_share > BASE.informal_share + 0.05
    assert hi["l_f"] < BASE["l_f"]


def test_more_generous_benefits_raise_unemployment():
    hi = solve(CAL.at(rr=0.30), guess=BASE)
    assert hi.unemployment > BASE.unemployment + 0.01


def test_labour_tax_leaks_more_than_vat():
    """Same story, told by the two Laffer sweeps in the write-up."""
    lab = solve(CAL.at(tau_w=CAL.tau_w + 0.10), guess=BASE)
    vat = solve(CAL.at(tau_c=CAL.tau_c + 0.10), guess=BASE)
    assert (lab.informal_share - BASE.informal_share) > \
           (vat.informal_share - BASE.informal_share)


# --- budget closure ---------------------------------------------------------

def test_labour_tax_has_two_balancing_rates():
    """Endpoint bracketing would report 'no solution' here; both ends are positive."""
    roots = balancing_rates(CAL.at(rr=0.20), "tau_w", BASE["lump_tax"],
                            CAL.tau_w, 0.70, guess=BASE)
    assert len(roots) == 2
    assert roots[0] == pytest.approx(0.3863, abs=1e-3)
    assert roots[1] > 0.60                       # past the Laffer peak


def test_vat_has_one_balancing_rate():
    roots = balancing_rates(CAL.at(rr=0.20), "tau_c", BASE["lump_tax"],
                            CAL.tau_c, 0.60, guess=BASE)
    assert len(roots) == 1
    assert roots[0] == pytest.approx(0.2060, abs=1e-3)


def test_vat_dominates_labour_tax_for_the_poor():
    """The write-up's central claim, and the reason it must not silently break."""
    expanded = CAL.at(rr=0.20)
    tw = balancing_rates(expanded, "tau_w", BASE["lump_tax"], CAL.tau_w, 0.70, guess=BASE)[0]
    tc = balancing_rates(expanded, "tau_c", BASE["lump_tax"], CAL.tau_c, 0.60, guess=BASE)[0]
    lab = solve(expanded.at(tau_w=tw), guess=BASE)
    vat = solve(expanded.at(tau_c=tc), guess=BASE)
    assert vat.real_c_spd > lab.real_c_spd        # the poor do better under VAT
    assert vat["y_tot"] > lab["y_tot"]            # so does output


@pytest.mark.parametrize("sigma", [1.5, 2.0, 3.0, 4.0])
def test_ranking_is_robust_to_substitutability(sigma):
    cal, base = calibrate(Params(sigma=sigma))
    expanded = cal.at(rr=0.20)
    tw = balancing_rates(expanded, "tau_w", base["lump_tax"], cal.tau_w, 0.70, guess=base)[0]
    tc = balancing_rates(expanded, "tau_c", base["lump_tax"], cal.tau_c, 0.60, guess=base)[0]
    lab = solve(expanded.at(tau_w=tw), guess=base)
    vat = solve(expanded.at(tau_c=tc), guess=base)
    assert vat.real_c_spd > lab.real_c_spd
    assert vat["y_tot"] > lab["y_tot"]


def test_laffer_peak_is_interior_for_labour_tax_only():
    rates = np.linspace(0.05, 0.70, 30)
    rev = [solve(CAL.at(tau_w=float(r)), guess=BASE)["gov_rev"] for r in rates]
    best = int(np.argmax(rev))
    assert 0 < best < len(rev) - 1                # labour tax turns over
    assert rates[best] > CAL.tau_w                # ...but above Turkey's current rate

    vrates = np.linspace(0.0, 0.60, 20)
    vrev = [solve(CAL.at(tau_c=float(r)), guess=BASE)["gov_rev"] for r in vrates]
    assert int(np.argmax(vrev)) == len(vrev) - 1  # VAT still rising at the top


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
