"""
Fiscal Closure Comparison: Who Pays for Social Protection?
==========================================================

Compares three ways of funding a +33% unemployment benefit expansion:

  A. Baseline            -- current policy.
  B. Debt / lump-sum     -- benefits absorbed by the lump-sum tax on savers.
  C. Labour tax financed -- tau_w rises until savers are back to their
                            baseline lump-sum burden.

Scenario C is solved by bracketing and Brent's method rather than the old
gradient loop, so it reports either an exact rate or an honest "no such rate
exists" together with the point where the equilibrium ceases to exist.

Output: results/turkey_tax_experiment.png
        results/financing_experiments.csv
"""

import os

import matplotlib.pyplot as plt

from turkey_tank import Params, solve, write_csv
from turkey_tank.experiments import find_balancing_rate


def main():
    base_p = Params()

    # =========================================================================
    # A. BASELINE
    # =========================================================================
    print("--- Scenario A: Baseline ---")
    base = solve(base_p)
    print(f"  U: {base['u']:.2%} | fiscal balance (lump on savers): "
          f"{base['lump_tax']:+.4f}")

    # =========================================================================
    # B. DEBT / LUMP-SUM FINANCED (the rich pay)
    # =========================================================================
    print("\n--- Scenario B: Debt / lump-sum financing ---")
    debt_p = base_p.at(p_ben_level=0.40)
    debt = solve(debt_p, guess=base)
    print(f"  U: {debt['u']:.2%} | lump on savers: {debt['lump_tax']:+.4f} "
          f"(vs {base['lump_tax']:+.4f} baseline)")

    # =========================================================================
    # C. LABOUR TAX FINANCED
    # =========================================================================
    print("\n--- Scenario C: Labour tax financing ---")
    print("Goal: raise tau_w until the lump-sum burden on savers returns to "
          "its baseline level.")
    search = find_balancing_rate(debt_p, "tau_w", target=base["lump_tax"],
                                 lo=0.35, hi=0.70)
    print("  " + search.describe())

    if search.found:
        tax = search.solution
        tax_label = f"Labour Tax Fund\n(tau_w = {search.rate:.1%})"
    else:
        # Report the best attainable point instead of a rate that does not exist.
        tax = solve(debt_p.at(tau_w=search.feasible_max))
        tax_label = f"Labour Tax Fund\n(max feasible {search.feasible_max:.1%})"
        print(f"  Reporting the ceiling instead: tau_w = {search.feasible_max:.2%}, "
              f"U = {tax['u']:.2%}, residual gap {search.best_gap:+.4f}")

    # =========================================================================
    # OUTPUT
    # =========================================================================
    os.makedirs("results", exist_ok=True)
    write_csv("results/financing_experiments.csv", [
        {"scenario": "baseline", "balances_budget": True, **base.summary()},
        {"scenario": "debt_lumpsum", "balances_budget": True, **debt.summary()},
        {"scenario": "labour_tax", "balances_budget": search.found, **tax.summary()},
    ])
    print("\n[SUCCESS] Data saved to results/financing_experiments.csv")

    def pchg(new, old):
        return ((new - old) / old) * 100

    scenarios = ["Baseline", "Debt/Rich Fund", tax_label]
    u_vals = [base["u"] * 100, debt["u"] * 100, tax["u"] * 100]
    welf_vals = [0, pchg(debt["c_spd"], base["c_spd"]),
                 pchg(tax["c_spd"], base["c_spd"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Fiscal Closure Matters: Who Pays for Social Protection?",
                 fontsize=16)

    bars1 = ax1.bar(scenarios, u_vals, color=["gray", "#d62728", "#8c0000"],
                    alpha=0.8)
    ax1.set_ylabel("Unemployment Rate (%)")
    ax1.set_title("Unemployment Impact")
    ax1.set_ylim(0, max(u_vals) * 1.2)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f"{bar.get_height():.2f}%", ha="center", va="bottom",
                 weight="bold")

    bars2 = ax2.bar(scenarios, welf_vals, color=["gray", "green", "red"],
                    alpha=0.8)
    ax2.set_ylabel("% Change in Consumption (Poor)")
    ax2.set_title("Welfare Impact (Poor Households)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h,
                 f"{h:+.2f}%", ha="center",
                 va="bottom" if h >= 0 else "top", weight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/turkey_tax_experiment.png", dpi=300)
    print("[SUCCESS] Chart saved to results/turkey_tax_experiment.png")
    plt.show()


if __name__ == "__main__":
    main()
