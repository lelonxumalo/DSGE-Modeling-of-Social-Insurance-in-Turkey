"""
Steady-state TANK-SM model for Turkey (dual formal/informal labour market).

The model is a square system of 23 nonlinear equations in 23 unknowns, solved
with ``scipy.optimize.root``.  It was originally written in GAMSPy; GAMS is no
longer required.

Porting notes
-------------
GAMSPy declared most variables ``type="positive"``.  Those bounds do real work:
``p_fill = m_eff * theta**(-0.5)`` is undefined for negative tightness, and an
unconstrained Newton step walks straight into it.  We reproduce the bounds by
solving in logs for every non-free variable.  ``method="lm"`` is used because
scipy's default ``"hybr"`` stalls on this system.

Accounting fixes
----------------
Two bugs in the original GAMSPy scripts are fixed here.  Pass
``Params(legacy_accounting=True)`` to reproduce the old (published) numbers.

1. ``firm_profit`` did not net out the capital rental bill, so savers received
   ``r * k`` twice -- roughly a third of GDP.
2. The government paid ``p_ben_level * u`` in benefits but only spenders
   received transfers, so half of every benefit lira vanished.

Together these broke the aggregate resource constraint by ``r*k - 0.5*b*u``.
``Solution.check()`` now asserts ``C + I == Y - vacancy costs`` after every
solve, which is the check that would have caught both.

Wage rule
---------
Nash bargaining with a labour tax.  Splitting the surplus with worker weight
``eta`` when the worker's income is taxed at ``tau_w`` and the outside option
is untaxed gives

    (1-eta) * [(1-tau_w)*w - z]  ==  eta * [mpl + theta*kappa - w]

which rearranges to the form used below::

    w * (1 - tau_w*(1-eta))  ==  (1-eta)*z + eta*(mpl + theta*kappa)

This is a *bargaining incidence* channel: a higher labour tax raises the
pre-tax wage a firm must pay, which suppresses vacancy posting.  It is not a
"workers exit to informality" channel -- informal labour never enters the
block that determines ``u``.  See the modelling caveats in the README.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.optimize import root

__all__ = [
    "Params", "Solution", "SolveError", "solve", "residuals",
    "VARIABLES", "DEFAULT_GUESS", "write_csv",
]


class SolveError(RuntimeError):
    """Raised when no equilibrium could be found at the given parameters."""


@dataclass(frozen=True)
class Params:
    # --- deep parameters ---
    beta: float = 0.985           # discount factor (high Turkish real rates)
    alpha: float = 0.40           # capital share, formal sector
    delta: float = 0.025          # depreciation
    phi: float = 1.0              # inverse Frisch elasticity

    # --- labour market ---
    p_informal_prod: float = 1.2  # informal productivity (== informal wage)
    p_sep_rate: float = 0.07      # formal separation rate (high turnover)
    p_match_eff: float = 0.35     # matching efficiency
    p_vac_cost: float = 0.35      # vacancy posting cost
    p_bargain: float = 0.5        # worker bargaining power (eta)
    p_match_elast: float = 0.5    # matching elasticity

    # --- fiscal ---
    tau_w: float = 0.35           # formal labour tax (OECD tax wedge)
    tau_c: float = 0.0            # VAT; 0.18 in the VAT experiment
    p_ben_level: float = 0.3      # unemployment benefit LEVEL, not a rate.
                                  # Implied replacement rate is b / w_f.

    # --- population ---
    pop_savers: float = 0.5

    # --- reproduce the pre-fix published results ---
    legacy_accounting: bool = False

    @property
    def pop_spenders(self) -> float:
        return 1.0 - self.pop_savers

    def at(self, **changes) -> "Params":
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)


VARIABLES = (
    "y_f", "y_i", "y_tot", "k", "inv", "r",
    "l_f", "l_i", "u", "v", "theta", "w_f", "w_i", "p_find", "p_fill",
    "c_sav", "c_spd", "lf_sav", "lf_spd", "li_spd",
    "gov_rev", "gov_exp", "lump_tax",
)

# lump_tax was GAMS type="free"; everything else was type="positive".
_FREE = ("lump_tax",)
_LOG_IDX = np.array([VARIABLES.index(v) for v in VARIABLES if v not in _FREE])

# Savers supply no informal labour.  In the GAMSPy version this was an equation
# pinning li_sav to zero; here it is imposed by construction, which keeps the
# system square without carrying a variable whose log is undefined.

DEFAULT_GUESS: dict[str, float] = {
    "y_f": 2.0, "y_i": 0.15, "y_tot": 2.15, "k": 10.0, "inv": 0.25, "r": 0.04,
    "l_f": 0.85, "l_i": 0.20, "u": 0.15, "v": 0.15, "theta": 1.0,
    "w_f": 1.2, "w_i": 0.65, "p_find": 0.6, "p_fill": 0.6,
    "c_sav": 1.5, "c_spd": 0.9, "lf_sav": 0.425, "lf_spd": 0.425,
    "li_spd": 0.20, "gov_rev": 0.3, "gov_exp": 0.05, "lump_tax": 0.0,
}


def residuals(x: np.ndarray, p: Params) -> np.ndarray:
    """Return the 23 equation residuals at point ``x``."""
    d = dict(zip(VARIABLES, x))
    eta, kappa, tc = p.p_bargain, p.p_vac_cost, 1.0 + p.tau_c
    ps, pv = p.pop_spenders, p.pop_savers

    mpl_f = (1.0 - p.alpha) * d["y_f"] / d["l_f"]
    discount = 1.0 / p.beta - (1.0 - p.p_sep_rate)          # == r + s
    capital_bill = d["r"] * d["k"]

    if p.legacy_accounting:
        firm_profit = d["y_f"] - d["w_f"] * d["l_f"] - d["v"] * kappa
        sav_transfers = 0.0
    else:
        firm_profit = (d["y_f"] - d["w_f"] * d["l_f"]
                       - capital_bill - d["v"] * kappa)
        sav_transfers = p.p_ben_level * d["u"] * pv
    spd_transfers = p.p_ben_level * d["u"] * ps

    return np.array([
        # --- production & capital ---
        d["y_f"] - d["k"] ** p.alpha * d["l_f"] ** (1.0 - p.alpha),
        d["y_i"] - p.p_informal_prod * d["l_i"],
        d["y_tot"] - d["y_f"] - d["y_i"],
        d["r"] - p.alpha * d["y_f"] / d["k"],
        d["w_i"] - p.p_informal_prod,                        # Walrasian informal wage
        1.0 - p.beta * (1.0 + d["r"] - p.delta),             # Euler
        d["inv"] - p.delta * d["k"],

        # --- search & matching ---
        d["l_f"] * p.p_sep_rate
        - p.p_match_eff * d["u"] ** p.p_match_elast
        * d["v"] ** (1.0 - p.p_match_elast),
        d["theta"] - d["v"] / d["u"],
        d["p_fill"] - p.p_match_eff * d["theta"] ** (-p.p_match_elast),
        d["p_find"] - p.p_match_eff * d["theta"] ** (1.0 - p.p_match_elast),
        d["l_f"] + d["u"] - 1.0,                             # formal labour force
        kappa / d["p_fill"] - (mpl_f - d["w_f"]) / discount,  # job creation
        d["w_f"] * (1.0 - p.tau_w * (1.0 - eta))
        - ((1.0 - eta) * (d["w_i"] + p.p_ben_level)
           + eta * (mpl_f + d["theta"] * kappa)),            # Nash wage

        # --- households ---
        d["lf_sav"] - d["l_f"] * pv,
        d["lf_spd"] - d["l_f"] * ps,
        d["l_i"] - d["li_spd"],
        d["w_i"] - d["c_spd"] * tc * (d["lf_spd"] + d["li_spd"]) ** p.phi,
        d["c_spd"] * tc - ((1.0 - p.tau_w) * d["w_f"] * d["lf_spd"]
                           + d["w_i"] * d["li_spd"] + spd_transfers),
        d["c_sav"] * tc - (capital_bill
                           + (1.0 - p.tau_w) * d["w_f"] * d["lf_sav"]
                           + firm_profit + sav_transfers
                           - d["inv"] - d["lump_tax"]),

        # --- government ---
        d["gov_rev"] - (p.tau_w * d["w_f"] * d["l_f"]
                        + p.tau_c * (d["c_sav"] + d["c_spd"])),
        d["gov_exp"] - p.p_ben_level * d["u"],
        d["gov_rev"] + d["lump_tax"] - d["gov_exp"],
    ])


@dataclass
class Solution:
    values: dict[str, float]
    params: Params
    residual_norm: float
    x: np.ndarray = field(repr=False)

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    # --- headline statistics ------------------------------------------------
    @property
    def unemployment_formal(self) -> float:
        """u as the model defines it: share of the *formal* labour force."""
        return self["u"]

    @property
    def unemployment_total(self) -> float:
        """u as a share of everyone working or searching (TurkStat-comparable)."""
        return self["u"] / (self["l_f"] + self["l_i"] + self["u"])

    @property
    def informal_employment_share(self) -> float:
        return self["l_i"] / (self["l_f"] + self["l_i"])

    @property
    def informal_output_share(self) -> float:
        return self["y_i"] / self["y_tot"]

    @property
    def replacement_rate(self) -> float:
        """What ``p_ben_level`` actually replaces, as a fraction of the wage."""
        return self.params.p_ben_level / self["w_f"]

    @property
    def consumption(self) -> float:
        return self["c_sav"] + self["c_spd"]

    @property
    def inequality_ratio(self) -> float:
        return self["c_sav"] / self["c_spd"]

    @property
    def walras_gap(self) -> float:
        """C + I - (Y - vacancy costs).  Must be zero."""
        return (self.consumption + self["inv"]
                - (self["y_tot"] - self["v"] * self.params.p_vac_cost))

    def check(self, tol: float = 1e-8) -> "Solution":
        if self.residual_norm > tol:
            raise SolveError(f"residual norm {self.residual_norm:.3e} > {tol:.1e}")
        if not self.params.legacy_accounting and abs(self.walras_gap) > tol:
            raise SolveError(
                "resource constraint violated: C + I - (Y - vac) = "
                f"{self.walras_gap:.6e}"
            )
        return self

    def summary(self) -> dict[str, float]:
        """Flat, CSV-friendly record of parameters and headline results."""
        out: dict[str, float] = {f"param_{k}": v for k, v in vars(self.params).items()}
        out.update({k: self.values[k] for k in VARIABLES})
        out.update(
            unemployment_formal=self.unemployment_formal,
            unemployment_total=self.unemployment_total,
            informal_employment_share=self.informal_employment_share,
            informal_output_share=self.informal_output_share,
            replacement_rate=self.replacement_rate,
            consumption=self.consumption,
            inequality_ratio=self.inequality_ratio,
            walras_gap=self.walras_gap,
        )
        return out

    def report(self, title: str = "MODEL RESULTS") -> None:
        print("\n" + "=" * 48)
        print(f" {title}")
        print("=" * 48)
        rows = [
            ("Total GDP", self["y_tot"], "{:.4f}"),
            ("Formal output", self["y_f"], "{:.4f}"),
            ("Informal output", self["y_i"], "{:.4f}"),
            ("Informal share of output", self.informal_output_share, "{:.2%}"),
            ("Informal share of employment", self.informal_employment_share, "{:.2%}"),
            ("Unemployment (formal LF)", self.unemployment_formal, "{:.2%}"),
            ("Unemployment (total LF)", self.unemployment_total, "{:.2%}"),
            ("Formal real wage", self["w_f"], "{:.4f}"),
            ("Informal real wage", self["w_i"], "{:.4f}"),
            ("Implied replacement rate", self.replacement_rate, "{:.2%}"),
            ("Savers consumption", self["c_sav"], "{:.4f}"),
            ("Spenders consumption", self["c_spd"], "{:.4f}"),
            ("Inequality ratio (sav/spd)", self.inequality_ratio, "{:.2f}"),
            ("Tax revenue", self["gov_rev"], "{:.4f}"),
            ("Social transfers", self["gov_exp"], "{:.4f}"),
            ("Lump-sum on savers", self["lump_tax"], "{:+.4f}"),
            ("Walras gap", self.walras_gap, "{:.2e}"),
        ]
        for label, value, fmt in rows:
            print(f"{label:.<34} {fmt.format(value)}")


def _unpack(z: np.ndarray) -> np.ndarray:
    x = z.copy()
    x[_LOG_IDX] = np.exp(z[_LOG_IDX])
    return x


def solve(p: Params, guess: "Solution | np.ndarray | None" = None,
          strict: bool = True) -> Solution:
    """Solve the steady state.

    ``guess`` may be a previous :class:`Solution`, which is how the sweeps do
    continuation: each solve starts from the last one that converged.
    """
    if guess is None:
        x0 = np.array([DEFAULT_GUESS[v] for v in VARIABLES], dtype=float)
    elif isinstance(guess, Solution):
        x0 = guess.x.copy()
    else:
        x0 = np.asarray(guess, dtype=float)

    z0 = x0.copy()
    z0[_LOG_IDX] = np.log(np.maximum(x0[_LOG_IDX], 1e-9))

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        res = root(lambda z: residuals(_unpack(z), p), z0,
                   method="lm", tol=1e-13)

    x = _unpack(res.x)
    if not np.all(np.isfinite(x)):
        raise SolveError("solver diverged (non-finite iterate)")

    norm = float(np.max(np.abs(residuals(x, p))))
    if not res.success or norm > 1e-8:
        raise SolveError(
            f"no equilibrium at tau_w={p.tau_w:.4f}, tau_c={p.tau_c:.4f}, "
            f"b={p.p_ben_level:.4f}: {res.message} (residual {norm:.2e})"
        )

    sol = Solution(values=dict(zip(VARIABLES, x)), params=p,
                   residual_norm=norm, x=x)
    return sol.check() if strict else sol


def write_csv(path: "str | Path", rows: list) -> Path:
    """Write scenario rows to CSV so published tables can be regenerated."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out
