"""
The Laffer Curve - Diagnosing Fiscal Limits
===========================================

Sweeps the labour tax rate to trace government revenue, formal unemployment
and the size of the informal sector, and answers: how much revenue can be
extracted from formal labour before the formal sector collapses?

Solve failures at high tax rates are results, not bugs.  Beyond a threshold
the job-creation and bargaining conditions have no common solution -- the
formal sector genuinely ceases to exist -- and the sweep records that as the
end of the feasible range rather than swallowing it in a bare ``except``.

Output: results/turkey_laffer_curve.png
        results/laffer_curve.csv
"""

import os
import sys
from pathlib import Path

# This script now lives one level down; make the model package importable when
# it is run directly (python legacy/N_name.py) rather than as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from turkey_tank import Params, write_csv
from turkey_tank.experiments import feasible_ceiling, sweep


def main():
    base_p = Params()
    tax_rates = np.linspace(0.05, 0.70, 25)

    print("--- Running Laffer curve sweep (5% to 70% labour tax) ---")
    points = sweep(base_p, "tau_w", tax_rates)

    rev, unemp, informal = [], [], []
    for pt in points:
        if not pt.ok:
            print(f"Tax: {pt.value:.2f} | NO EQUILIBRIUM (formal sector collapses)")
            rev.append(np.nan)
            unemp.append(np.nan)
            informal.append(np.nan)
            continue
        s = pt.solution
        r = s["gov_rev"]
        # A "solved" point with no formal sector left is not a revenue figure.
        if s["u"] > 0.99 or r < 0:
            r = np.nan
        rev.append(r)
        unemp.append(s["u"] * 100)
        informal.append(s.informal_output_share * 100)
        print(f"Tax: {pt.value:.2f} | Rev: {r:.4f} | U: {s['u']:.2%} "
              f"| Informal output: {s.informal_output_share:.1%}")

    ceiling = feasible_ceiling(base_p, "tau_w", points)
    if ceiling is not None and ceiling < tax_rates[-1]:
        print(f"\nEquilibrium ceases to exist above tau_w = {ceiling:.2%}.")

    # =========================================================================
    # OUTPUT
    # =========================================================================
    os.makedirs("results", exist_ok=True)
    write_csv("results/laffer_curve.csv", [
        {"tau_w": pt.value, "feasible": pt.ok,
         **(pt.solution.summary() if pt.ok
            else {k: float("nan") for k in points[0].solution.summary()})}
        for pt in points
    ])
    print("[SUCCESS] Data saved to results/laffer_curve.csv")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:blue"
    ax1.set_xlabel("Labour Tax Rate (tau_w)")
    ax1.set_ylabel("Gov Revenue (Model Units)", color=color, fontweight="bold")
    ax1.plot(tax_rates, rev, color=color, marker="o", linewidth=2, label="Revenue")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.6)

    valid = [r for r in rev if not np.isnan(r)]
    if valid:
        max_rev = max(valid)
        peak_tax = tax_rates[rev.index(max_rev)]
        ax1.axvline(peak_tax, color="black", linestyle="--", alpha=0.8)
        ax1.text(peak_tax, max_rev, f" Peak Rev\n @ Tax={peak_tax:.2f}",
                 ha="center", va="bottom", fontweight="bold")
        print(f"\nRevenue peaks at tau_w = {peak_tax:.2%} (revenue {max_rev:.4f}).")

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Unemployment Rate (%)", color=color, fontweight="bold")
    ax2.plot(tax_rates, unemp, color=color, linestyle="--", linewidth=2,
             label="Unemployment")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("The Turkey Laffer Curve: Fiscal Limits & Unemployment")
    plt.tight_layout()
    plt.savefig("results/turkey_laffer_curve.png", dpi=300)
    print("[SUCCESS] Saved Laffer curve to results/turkey_laffer_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
