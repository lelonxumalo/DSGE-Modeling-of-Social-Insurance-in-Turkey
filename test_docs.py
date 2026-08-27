"""
Keep the README's numbers honest.

The January write-up went stale because its figures were copied out of console
output by hand and never checked again. The README quotes the same kind of
figures, so it gets the same treatment: every number in its headline table and
calibration fit is recomputed here from a fresh solve.

The check is structural rather than textual. A presence test ("does '8.60%'
appear anywhere?") cannot catch a corrupted table cell, because the same figure
usually appears elsewhere in the document -- which is exactly how a deliberately
broken cell slipped through an earlier version of this guard. So it parses the
table row and compares cell by cell.

Run with:  python -m pytest test_docs.py -q
"""

import re
from pathlib import Path

import pytest

from turkey_tank.dual import balancing_rates, calibrate, solve

README = Path(__file__).resolve().parent / "README.md"


@pytest.fixture(scope="module")
def figures():
    """Every number the README quotes, recomputed from the model."""
    pars, base = calibrate()
    target = base["lump_tax"]
    expanded = pars.at(rr=0.20)

    unfunded = solve(expanded, guess=base)
    tw = balancing_rates(expanded, "tau_w", target, pars.tau_w, 0.70, guess=base)[0]
    tc = balancing_rates(expanded, "tau_c", target, pars.tau_c, 0.60, guess=base)[0]
    labour = solve(expanded.at(tau_w=tw), guess=base)
    vat = solve(expanded.at(tau_c=tc), guess=base)

    def dc(s):
        return (s.real_c_spd / base.real_c_spd - 1) * 100

    def dy(s):
        return (s["y_tot"] / base["y_tot"] - 1) * 100

    return {
        "rows": [
            ("Baseline", [f"{base.unemployment:.2%}", f"{base.informal_share:.2%}"]),
            ("Unfunded", [f"{unfunded.unemployment:.2%}",
                          f"{unfunded.informal_share:.2%}",
                          f"-{abs(dy(unfunded)):.2f}%", f"{dc(unfunded):+.2f}%"]),
            ("Payroll tax", [f"{labour.unemployment:.2%}",
                             f"{labour.informal_share:.2%}",
                             f"-{abs(dy(labour)):.2f}%", f"-{abs(dc(labour)):.2f}%"]),
            ("VAT", [f"{vat.unemployment:.2%}", f"{vat.informal_share:.2%}",
                     f"-{abs(dy(vat)):.2f}%", f"{dc(vat):+.2f}%"]),
        ],
        "prose": {
            "formal/informal wage gap": f"{base.wage_gap:.2f}x",
            "revenue / GDP": f"{base['gov_rev'] / base['y_tot']:.1%}",
            "job-finding rate": f"{base['p_find']:.1%}",
            "vacancy costs / output":
                f"{pars.vac_cost * base['v'] / base['y_f']:.1%}",
            "budget-closing payroll rate": f"{tw:.2%}",
            "budget-closing VAT rate": f"{tc:.2%}",
        },
    }


def _text():
    return README.read_text(encoding="utf-8").replace("−", "-").replace("→", "->")


def _cells(line):
    return [re.sub(r"\*\*|`", "", c).strip() for c in line.strip().strip("|").split("|")]


@pytest.mark.parametrize("index", range(4))
def test_headline_table_row_is_current(figures, index):
    """Each scenario row must carry the figures the model actually produces."""
    label, expected = figures["rows"][index]
    # Several tables carry these labels, so accept the row holding all the
    # expected cells rather than the first one bearing the label.
    candidates = [_cells(ln) for ln in _text().splitlines()
                  if label in ln and "|" in ln]
    assert candidates, f"no table row labelled {label!r} in README.md"
    if any(all(want in cells for want in expected) for cells in candidates):
        return
    best = max(candidates, key=lambda c: sum(w in c for w in expected))
    missing = [w for w in expected if w not in best]
    raise AssertionError(
        f"README headline row {label!r} is missing {missing}; "
        f"closest row has {[c for c in best if c][1:]}")


@pytest.mark.parametrize("name", [
    "formal/informal wage gap", "revenue / GDP", "job-finding rate",
    "vacancy costs / output", "budget-closing payroll rate",
    "budget-closing VAT rate",
])
def test_quoted_figure_is_current(figures, name):
    value = figures["prose"][name]
    assert value in _text(), f"README no longer quotes {name} as {value}"


def test_readme_does_not_describe_v1_as_current():
    """The v1 sections must stay labelled, since their results are withdrawn."""
    text = _text()
    assert "## Legacy scripts (v1)" in text
    assert "withdrawn" in text.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
