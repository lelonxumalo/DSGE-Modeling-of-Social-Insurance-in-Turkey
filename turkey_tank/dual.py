"""
Dual labour market model for Turkey, v2.

This supersedes ``turkey_tank/model.py`` (kept only to reproduce the original
write-up).  Three things changed, each because a claim in that write-up could
not be supported by the v1 code.

1. **Coherent labour force.**  v1 imposed ``l_f + u == 1``, normalising the
   *formal* labour force, with informal workers outside the identity -- so
   ``u`` was not a rate comparable to TurkStat's 8-9%, and informal work was
   moonlighting by the formally employed rather than an alternative to a formal
   job.  Here ``l_f + l_i + u == 1``: every worker is formally employed,
   informally employed, or unemployed and searching.

2. **Informality is a margin.**  Workers not in a formal job choose between
   informal work and searching.  Indifference between the two,

       w_i == benefit + home_production + beta * p_find * S_worker

   pins the split: the informal wage must compensate for giving up the option
   value of search.  Informal output has decreasing returns (``gamma < 1``), so
   the sector absorbs labour at a falling wage rather than infinitely elastic.

3. **VAT is evaded informally.**  Households consume a CES bundle of formal and
   informal goods; VAT applies to the formal good only.  Raising VAT therefore
   shifts demand toward informal goods, lifts their relative price ``p`` and
   with it ``w_i``, which raises the outside option in bargaining and destroys
   formal jobs.  In v1 VAT applied to all consumption and entered no equation
   that determined employment, so it was non-distortionary by construction and
   the "VAT preserves jobs" result was an artefact.  Here VAT has a real cost,
   and the question the write-up asked is actually answerable.

What taxes can and cannot move
------------------------------
The formal sector is CRS and the interest rate is pinned by ``beta``, so the
formal marginal product of labour is a constant.  Free entry, the Nash wage,
the surplus definition and the informal-search indifference then close a
six-equation block in ``(w_f, w_i, s_w, theta, p_find, p_fill)`` containing no
price and no quantity.  So **VAT moves the allocation of labour, not wages**:
it shifts workers between formal work, informal work and unemployment while
leaving what each is paid untouched.  The labour tax, which does enter that
block, moves both.  State this plainly rather than letting a reader infer a
wage channel for VAT that is not there -- ``test_vat_moves_quantities_not_wages``
pins it.

Wage rule
---------
Nash bargaining over the match surplus with a labour tax.  The worker's threat
point is non-employment, whose value is pinned by the informal option, so the
usual ``theta*kappa`` term is absent -- the search option value is already
priced into the indifference condition above.  Splitting

    eta * J * (1-tau_w)  ==  (1-eta) * S_worker

gives

    w_f * (1-tau_w)  ==  (1-eta)*w_i + eta*(1-tau_w)*mpl_f

Calibration
-----------
Parameters marked "calibrated" in :class:`Params` are solved for, not chosen:
see :func:`calibrate`, which pins them to Turkish targets.  The defaults below
are the output of that routine.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq, least_squares

from turkey_tank.model import SolveError, write_csv  # reuse error type + CSV helper

__all__ = ["Params", "Targets", "Solution", "SolveError", "solve", "calibrate",
           "balancing_rates", "VARIABLES", "residuals", "write_csv"]


@dataclass(frozen=True)
class Params:
    # --- deep parameters, set a priori ---
    beta: float = 0.985           # quarterly discount factor
    alpha: float = 0.40           # capital share, formal sector
    delta: float = 0.025          # depreciation
    eps: float = 0.5              # matching elasticity
    eta: float = 0.5              # worker bargaining power
    sep: float = 0.07             # quarterly formal separation rate
    gamma: float = 0.85           # returns to labour in the informal sector
    sigma: float = 2.0            # CES elasticity, formal vs informal goods

    # --- policy ---
    tau_w: float = 0.375          # labour tax wedge
    tau_c: float = 0.20           # VAT, formal goods only
    rr: float = 0.15              # transfer to the unemployed, share of w_f

    # --- population ---
    pop_savers: float = 0.5

    # --- calibrated to targets (defaults are calibrate() output) ---
    match_eff: float = 0.5310     # A
    vac_cost: float = 1.3855      # kappa
    z_i: float = 1.6459           # informal TFP
    omega: float = 0.8660         # CES weight on the formal good
    home_prod: float = 0.2240     # flow value of home production

    @property
    def pop_spenders(self) -> float:
        return 1.0 - self.pop_savers

    def at(self, **changes) -> "Params":
        return replace(self, **changes)


@dataclass(frozen=True)
class Targets:
    """Turkish moments the model is calibrated to match.

    Approximate figures for 2024-25; verify against the current TurkStat and
    OECD releases before quoting them.
    """
    unemployment: float = 0.086       # TurkStat headline rate, ~8.6%
    informal_share: float = 0.27      # unregistered share of employment, ~27%
    wage_gap: float = 1.75            # formal / informal wage
    tightness: float = 1.0            # v/u, a normalisation
    informal_price: float = 1.0       # relative price of informal goods, a normalisation


VARIABLES = (
    "y_f", "y_i", "y_tot", "k", "inv", "r",
    "l_f", "l_i", "u", "v", "theta", "p_find", "p_fill",
    "w_f", "w_i", "s_w", "p", "pidx",
    "cf_sav", "ci_sav", "cf_spd", "ci_spd", "inc_sav", "inc_spd",
    "gov_rev", "gov_exp", "lump_tax",
)

_FREE = ("lump_tax",)
_LOG_IDX = np.array([VARIABLES.index(x) for x in VARIABLES if x not in _FREE])

DEFAULT_GUESS = {
    "y_f": 2.98, "y_i": 0.39, "y_tot": 3.38, "k": 29.66, "inv": 0.74, "r": 0.0402,
    "l_f": 0.645, "l_i": 0.27, "u": 0.085, "v": 0.085, "theta": 1.0,
    "p_find": 0.53, "p_fill": 0.53, "w_f": 2.56, "w_i": 1.46, "s_w": 1.63,
    "p": 1.0, "pidx": 1.17,
    "cf_sav": 1.06, "ci_sav": 0.197, "cf_spd": 1.06, "ci_spd": 0.197,
    "inc_sav": 1.47, "inc_spd": 1.47,
    "gov_rev": 0.90, "gov_exp": 0.033, "lump_tax": -0.87,
}


def residuals(x: np.ndarray, p: Params) -> np.ndarray:
    d = dict(zip(VARIABLES, x))
    ps, pv = p.pop_spenders, p.pop_savers
    disc = 1.0 - p.beta * (1.0 - p.sep)          # match-surplus discount
    mpl_f = (1.0 - p.alpha) * d["y_f"] / d["l_f"]

    benefit = p.rr * d["w_f"]                    # transfer per unemployed worker
    labour_inc = ((1.0 - p.tau_w) * d["w_f"] * d["l_f"]
                  + d["w_i"] * d["l_i"] + benefit * d["u"])
    firm_profit = (d["y_f"] - d["w_f"] * d["l_f"]
                   - d["r"] * d["k"] - p.vac_cost * d["v"])

    p_f = 1.0 + p.tau_c                          # consumer price, formal good
    p_i = d["p"]                                 # consumer price, informal good

    def demand(income, weight, price):
        real = income / d["pidx"]
        return weight * (price / d["pidx"]) ** (-p.sigma) * real

    return np.array([
        # --- capital and formal production ---
        1.0 - p.beta * (1.0 + d["r"] - p.delta),
        d["r"] - p.alpha * d["y_f"] / d["k"],
        d["y_f"] - d["k"] ** p.alpha * d["l_f"] ** (1.0 - p.alpha),
        d["inv"] - p.delta * d["k"],

        # --- informal production (decreasing returns; workers get the average product) ---
        d["y_i"] - p.z_i * d["l_i"] ** p.gamma,
        d["w_i"] - d["p"] * p.z_i * d["l_i"] ** (p.gamma - 1.0),

        # --- labour force: formal, informal, or searching ---
        d["l_f"] + d["l_i"] + d["u"] - 1.0,
        p.sep * d["l_f"] - p.match_eff * d["u"] ** p.eps * d["v"] ** (1.0 - p.eps),
        d["theta"] - d["v"] / d["u"],
        d["p_fill"] - p.match_eff * d["theta"] ** (-p.eps),
        d["p_find"] - p.match_eff * d["theta"] ** (1.0 - p.eps),

        # --- job creation and wage setting ---
        p.vac_cost / d["p_fill"] - (mpl_f - d["w_f"]) / disc,
        d["w_f"] * (1.0 - p.tau_w)
        - ((1.0 - p.eta) * d["w_i"] + p.eta * (1.0 - p.tau_w) * mpl_f),
        d["s_w"] - (d["w_f"] * (1.0 - p.tau_w) - d["w_i"]) / disc,

        # --- informal work vs. search: the margin that makes informality endogenous ---
        d["w_i"] - (benefit + p.home_prod + p.beta * d["p_find"] * d["s_w"]),

        # --- consumption: CES over formal (taxed) and informal (untaxed) goods ---
        d["pidx"] - (p.omega * p_f ** (1.0 - p.sigma)
                     + (1.0 - p.omega) * p_i ** (1.0 - p.sigma)) ** (1.0 / (1.0 - p.sigma)),
        d["inc_spd"] - labour_inc * ps,
        d["inc_sav"] - (labour_inc * pv + d["r"] * d["k"] + firm_profit
                        - d["inv"] - d["lump_tax"]),
        d["cf_spd"] - demand(d["inc_spd"], p.omega, p_f),
        d["ci_spd"] - demand(d["inc_spd"], 1.0 - p.omega, p_i),
        d["cf_sav"] - demand(d["inc_sav"], p.omega, p_f),
        d["ci_sav"] - demand(d["inc_sav"], 1.0 - p.omega, p_i),
        d["ci_sav"] + d["ci_spd"] - d["y_i"],        # informal goods clear -> pins p

        # --- government ---
        d["gov_rev"] - (p.tau_w * d["w_f"] * d["l_f"]
                        + p.tau_c * (d["cf_sav"] + d["cf_spd"])),
        d["gov_exp"] - benefit * d["u"],
        d["gov_rev"] + d["lump_tax"] - d["gov_exp"],

        # --- accounting ---
        d["y_tot"] - (d["y_f"] + d["p"] * d["y_i"]),
    ])


@dataclass
class Solution:
    values: dict
    params: Params
    residual_norm: float
    x: np.ndarray

    def __getitem__(self, k):
        return self.values[k]

    @property
    def unemployment(self) -> float:
        """Share of the labour force unemployed -- TurkStat-comparable."""
        return self["u"]

    @property
    def informal_share(self) -> float:
        """Unregistered share of total employment."""
        return self["l_i"] / (self["l_f"] + self["l_i"])

    @property
    def informal_output_share(self) -> float:
        return self["p"] * self["y_i"] / self["y_tot"]

    @property
    def wage_gap(self) -> float:
        return self["w_f"] / self["w_i"]

    @property
    def replacement_rate(self) -> float:
        return self.params.rr

    @property
    def consumption(self) -> float:
        """Real consumption index, both types."""
        return (self["inc_sav"] + self["inc_spd"]) / self["pidx"]

    @property
    def real_c_spd(self) -> float:
        return self["inc_spd"] / self["pidx"]

    @property
    def real_c_sav(self) -> float:
        return self["inc_sav"] / self["pidx"]

    @property
    def inequality_ratio(self) -> float:
        return self.real_c_sav / self.real_c_spd

    @property
    def walras_gap(self) -> float:
        """Formal goods market clearing, redundant by Walras' law. Must be zero."""
        return (self["cf_sav"] + self["cf_spd"] + self["inv"]
                + self.params.vac_cost * self["v"] - self["y_f"])

    @property
    def fiscal_balance(self) -> float:
        return self["lump_tax"]

    def check(self, tol: float = 1e-8) -> "Solution":
        if self.residual_norm > tol:
            raise SolveError(f"residual norm {self.residual_norm:.3e} > {tol:.1e}")
        if abs(self.walras_gap) > 1e-7:
            raise SolveError(f"formal goods market does not clear: {self.walras_gap:.3e}")
        return self

    def summary(self) -> dict:
        out = {f"param_{k}": v for k, v in vars(self.params).items()}
        out.update({k: self.values[k] for k in VARIABLES})
        out.update(
            unemployment=self.unemployment,
            informal_share=self.informal_share,
            informal_output_share=self.informal_output_share,
            wage_gap=self.wage_gap,
            replacement_rate=self.replacement_rate,
            real_c_sav=self.real_c_sav,
            real_c_spd=self.real_c_spd,
            inequality_ratio=self.inequality_ratio,
            walras_gap=self.walras_gap,
        )
        return out

    def report(self, title="MODEL RESULTS") -> None:
        print("\n" + "=" * 52)
        print(f" {title}")
        print("=" * 52)
        rows = [
            ("Unemployment rate", self.unemployment, "{:.2%}"),
            ("Informal share of employment", self.informal_share, "{:.2%}"),
            ("Informal share of GDP", self.informal_output_share, "{:.2%}"),
            ("Formal / informal wage gap", self.wage_gap, "{:.2f}x"),
            ("Job finding rate (quarterly)", self["p_find"], "{:.2%}"),
            ("Market tightness (v/u)", self["theta"], "{:.3f}"),
            ("Formal real wage", self["w_f"], "{:.4f}"),
            ("Informal real wage", self["w_i"], "{:.4f}"),
            ("Relative price, informal good", self["p"], "{:.4f}"),
            ("GDP", self["y_tot"], "{:.4f}"),
            ("Real consumption, savers", self.real_c_sav, "{:.4f}"),
            ("Real consumption, spenders", self.real_c_spd, "{:.4f}"),
            ("Inequality ratio", self.inequality_ratio, "{:.2f}"),
            ("Tax revenue", self["gov_rev"], "{:.4f}"),
            ("Transfers to unemployed", self["gov_exp"], "{:.4f}"),
            ("Lump-sum on savers", self["lump_tax"], "{:+.4f}"),
            ("Walras gap", self.walras_gap, "{:.2e}"),
        ]
        for label, value, fmt in rows:
            print(f"{label:.<36} {fmt.format(value)}")


def _unpack(z):
    x = z.copy()
    x[_LOG_IDX] = np.exp(z[_LOG_IDX])
    return x


def _least_squares(f, z0, tol: float = 1e-8):
    """Solve ``f(z) == 0``, returning the unpacked levels or None.

    Capital is two orders of magnitude larger than most variables, which leaves
    the Jacobian badly scaled; a plain ``root(method="lm")`` stalls at a
    least-squares local minimum around 1e-4 and reports it as non-existence.
    Trust-region reflective with ``x_scale="jac"`` rescales columns by the
    Jacobian and converges to machine precision, so failures here are genuine.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        res = least_squares(f, z0, method="trf", x_scale="jac",
                            xtol=1e-15, ftol=1e-15, gtol=1e-15)
    x = _unpack(res.x)
    if not np.all(np.isfinite(x)) or np.max(np.abs(f(res.x))) > tol:
        return None
    return x


def solve(p: Params, guess=None, strict: bool = True) -> Solution:
    if guess is None:
        x0 = np.array([DEFAULT_GUESS[v] for v in VARIABLES], float)
    elif isinstance(guess, Solution):
        x0 = guess.x.copy()
    else:
        x0 = np.asarray(guess, float)

    z0 = x0.copy()
    z0[_LOG_IDX] = np.log(np.maximum(x0[_LOG_IDX], 1e-9))

    x = _least_squares(lambda z: residuals(_unpack(z), p), z0)
    if x is None:
        raise SolveError(
            f"no equilibrium at tau_w={p.tau_w:.4f}, tau_c={p.tau_c:.4f}, "
            f"rr={p.rr:.4f}")
    norm = float(np.max(np.abs(residuals(x, p))))

    sol = Solution(dict(zip(VARIABLES, x)), p, norm, x)
    return sol.check() if strict else sol


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

_CALIB = ("match_eff", "vac_cost", "z_i", "omega", "home_prod")


def calibrate(p: Params | None = None, t: Targets | None = None,
              verbose: bool = False) -> tuple[Params, Solution]:
    """Solve jointly for the model and the five parameters that hit ``t``.

    Rather than a nested loop, the target conditions are appended to the
    equation system and the calibrated parameters become unknowns, so one solve
    delivers both.
    """
    p = p or Params()
    t = t or Targets()

    def unpack_all(z):
        x = _unpack(z[:len(VARIABLES)])
        pars = p.at(**dict(zip(_CALIB, np.exp(z[len(VARIABLES):]))))
        return x, pars

    def f(z):
        x, pars = unpack_all(z)
        d = dict(zip(VARIABLES, x))
        return np.concatenate([
            residuals(x, pars),
            [d["u"] - t.unemployment,
             d["l_i"] / (d["l_f"] + d["l_i"]) - t.informal_share,
             d["w_f"] / d["w_i"] - t.wage_gap,
             d["theta"] - t.tightness,
             d["p"] - t.informal_price],
        ])

    x0 = np.array([DEFAULT_GUESS[v] for v in VARIABLES], float)
    z0 = np.concatenate([
        np.where(np.isin(np.arange(len(VARIABLES)), _LOG_IDX),
                 np.log(np.maximum(x0, 1e-9)), x0),
        np.log([getattr(p, c) for c in _CALIB]),
    ])

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        res = least_squares(f, z0, method="trf", x_scale="jac",
                            xtol=1e-15, ftol=1e-15, gtol=1e-15)

    norm = float(np.max(np.abs(f(res.x))))
    if norm > 1e-8:
        raise SolveError(f"calibration failed: {res.message} (residual {norm:.2e})")

    x, pars = unpack_all(res.x)
    sol = Solution(dict(zip(VARIABLES, x)), pars,
                   float(np.max(np.abs(residuals(x, pars)))), x).check()
    if verbose:
        print("Calibrated parameters:")
        for c in _CALIB:
            print(f"  {c:<12} = {getattr(pars, c):.6f}")
    return pars, sol


# ---------------------------------------------------------------------------
# Budget closure
# ---------------------------------------------------------------------------

def balancing_rates(p: Params, name: str, target_lump: float,
                    lo: float, hi: float, n: int = 120,
                    guess: "Solution | None" = None) -> list:
    """Every rate of ``name`` in ``[lo, hi]`` that returns ``lump_tax`` to target.

    Returns a list because there can be more than one: the budget gap is not
    monotone in the tax rate, so a Laffer curve gives a low root (the useful
    one) and a high root on the far side of the peak.  Scanning for sign changes
    rather than bracketing the endpoints matters -- endpoint bracketing reports
    "no solution" whenever both ends sit on the same side of the curve.
    """
    pts = []
    for value in np.linspace(lo, hi, n):
        try:
            pts.append((float(value),
                        solve(p.at(**{name: float(value)}), guess=guess)["lump_tax"]
                        - target_lump))
        except SolveError:
            break

    out = []
    for (a, fa), (b, fb) in zip(pts, pts[1:]):
        if fa == 0.0:
            out.append(a)
        elif np.sign(fa) != np.sign(fb):
            out.append(float(brentq(
                lambda v: solve(p.at(**{name: float(v)}), guess=guess)["lump_tax"]
                - target_lump, a, b, xtol=1e-13)))
    return out
