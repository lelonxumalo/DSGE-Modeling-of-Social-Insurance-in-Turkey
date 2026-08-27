"""Sweep and budget-closure helpers shared by the experiment scripts.

The GAMSPy scripts closed the budget with a hand-rolled loop::

    new_tau = curr_tau + learning_rate * gap

which can oscillate, stops at a 1e-4 tolerance, and silently reports whatever
rate it happened to be sitting on when the iteration cap hit.  It also cannot
distinguish "did not converge" from "no such rate exists".

:func:`find_balancing_rate` scans a grid, brackets a sign change and hands it
to Brent's method, so it either returns a rate accurate to machine precision or
reports honestly that no rate in the range balances the budget -- and, in that
case, how far it got and where the equilibrium ceases to exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from turkey_tank.model import Params, Solution, SolveError, solve

__all__ = ["SweepPoint", "RateSearch", "sweep", "find_balancing_rate",
           "feasible_ceiling"]


@dataclass
class SweepPoint:
    value: float
    solution: Solution | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.solution is not None


def sweep(p: Params, name: str, values, strict: bool = True) -> list[SweepPoint]:
    """Solve ``p`` over ``values`` of parameter ``name``, using continuation.

    Each solve starts from the previous converged solution.  Points where no
    equilibrium exists come back with ``solution=None`` rather than raising --
    in this model a failure at a high tax rate is a result, not a bug.
    """
    points: list[SweepPoint] = []
    guess: Solution | None = None
    for value in values:
        try:
            sol = solve(p.at(**{name: float(value)}), guess=guess, strict=strict)
            guess = sol
            points.append(SweepPoint(float(value), sol))
        except SolveError as exc:
            points.append(SweepPoint(float(value), None, str(exc)))
    return points


def _bisect_ceiling(attempt, last_ok: float, infeasible: list,
                    tol: float = 1e-9) -> float:
    """Bisect the boundary where the equilibrium stops existing.

    A grid only brackets it to one grid step; the exact ceiling is an
    economically meaningful number (past it the job-creation and bargaining
    conditions share no solution), so it is worth pinning down.
    """
    above = [r for r in infeasible if r > last_ok]
    if not above:
        return last_ok
    ok, bad = last_ok, min(above)
    while bad - ok > tol:
        mid = 0.5 * (ok + bad)
        try:
            attempt(mid)
            ok = mid
        except SolveError:
            bad = mid
    return ok


def feasible_ceiling(p: Params, name: str, points: list) -> float | None:
    """Highest value of ``name`` at which an equilibrium still exists.

    Takes the :class:`SweepPoint` list from :func:`sweep` and refines the
    grid-bracketed boundary by bisection.
    """
    ok = [pt.value for pt in points if pt.ok]
    if not ok:
        return None
    bad = [pt.value for pt in points if not pt.ok]
    return _bisect_ceiling(lambda r: solve(p.at(**{name: float(r)}), strict=False),
                           max(ok), bad)


@dataclass
class RateSearch:
    """Outcome of searching for a budget-balancing tax rate."""
    name: str
    target: float
    rate: float | None = None
    solution: Solution | None = None
    feasible_max: float | None = None   # highest rate with an equilibrium
    best_gap: float | None = None       # closest approach to the target
    scanned: list = field(default_factory=list, repr=False)

    @property
    def found(self) -> bool:
        return self.rate is not None

    def describe(self) -> str:
        if self.found:
            return (f"{self.name} = {self.rate:.4%} balances the budget "
                    f"(gap {self.solution['lump_tax'] - self.target:+.2e})")
        ceiling = ("no equilibrium anywhere in the range"
                   if self.feasible_max is None
                   else f"equilibrium ceases to exist above {self.name} "
                        f"= {self.feasible_max:.4%}")
        return (f"NO feasible {self.name} balances the budget: {ceiling}; "
                f"closest gap {self.best_gap:+.4f}")


def find_balancing_rate(p: Params, name: str, target: float,
                        lo: float, hi: float, n: int = 240) -> RateSearch:
    """Find the rate of parameter ``name`` that returns ``lump_tax`` to ``target``.

    Returns a :class:`RateSearch` describing either the root or why there is
    none.  ``target`` is normally the baseline ``lump_tax``, i.e. "pay for the
    increment without leaning further on savers".
    """
    out = RateSearch(name=name, target=target)

    def gap(rate: float) -> float:
        return solve(p.at(**{name: float(rate)}), strict=False)["lump_tax"] - target

    grid = np.linspace(lo, hi, n)
    feasible: list[tuple[float, float]] = []
    infeasible: list[float] = []
    for rate in grid:
        try:
            feasible.append((float(rate), gap(rate)))
        except SolveError:
            infeasible.append(float(rate))

    out.scanned = feasible
    if not feasible:
        return out

    out.feasible_max = _bisect_ceiling(gap, max(r for r, _ in feasible), infeasible)
    out.best_gap = min((g for _, g in feasible), key=abs)

    for (r0, g0), (r1, g1) in zip(feasible, feasible[1:]):
        if g0 == 0.0 or np.sign(g0) != np.sign(g1):
            rate = brentq(gap, r0, r1, xtol=1e-13)
            out.rate = float(rate)
            out.solution = solve(p.at(**{name: out.rate}))
            out.best_gap = out.solution["lump_tax"] - target
            break

    return out
