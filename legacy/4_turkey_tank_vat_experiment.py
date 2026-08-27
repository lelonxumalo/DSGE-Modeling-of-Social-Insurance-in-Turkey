"""
VAT Financing Experiment (Consumption Tax)
==========================================

Tests whether financing social protection via VAT is less damaging to the
Turkish labour market than a labour tax.

READ THIS BEFORE INTERPRETING THE OUTPUT
----------------------------------------
In this model VAT *cannot* affect employment, by construction.  The eleven
equations that pin down (y_f, k, r, l_f, u, v, theta, w_f, w_i, p_find,
p_fill) contain neither ``tau_c`` nor any consumption variable, so the labour
block is recursive and unemployment is invariant to the VAT rate.  The script
asserts this explicitly below.

So the "VAT preserves jobs" comparison is really "tau_w = 0.35 versus
tau_w = 0.40", both at the higher benefit level -- any instrument other than
the labour tax would give the identical unemployment rate.  Worse, because
both the spender budget and the labour-supply condition involve only the gross
expenditure c*(1+tau_c), under log utility income and substitution effects
cancel exactly and ``tau_c`` is a pure rescaling of consumption: informal
labour does not move either.

To make this a real result, VAT has to be evaded in the informal sector --
i.e. apply tau_c to formal consumption only, so raising it shifts demand
toward informal output.  That is the channel that actually limits VAT in a
dual economy, and it is not yet in the model.

Output: results/turkey_vat_experiment.png
        results/vat_experiment.csv
"""

import os
import sys
from pathlib import Path

# This script now lives one level down; make the model package importable when
# it is run directly (python legacy/N_name.py) rather than as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from turkey_tank import Params, solve, write_csv
from turkey_tank.experiments import find_balancing_rate

VAT_BASE = 0.18


def main():
    base_p = Params(tau_c=VAT_BASE)

    # =========================================================================
    # 1. BASELINE
    # =========================================================================
    print("--- Baseline ---")
    base = solve(base_p)
    print(f"  U: {base['u']:.2%} | fiscal balance: {base['lump_tax']:+.4f}")

    # =========================================================================
    # 2. SCENARIO B: LABOUR TAX FINANCING
    # =========================================================================
    print("\n--- Scenario B: Labour tax financing ---")
    higher_benefits = base_p.at(p_ben_level=0.40)

    # Held at the Laffer peak for comparability with the published figure.
    labour = solve(higher_benefits.at(tau_w=0.40), guess=base)
    labour_gap = labour["lump_tax"] - base["lump_tax"]
    print(f"  At tau_w = 40%: U = {labour['u']:.2%}, residual gap "
          f"{labour_gap:+.4f} (NOT budget-balanced)")

    # Is there any labour tax rate that does balance it?
    labour_search = find_balancing_rate(higher_benefits, "tau_w",
                                        target=base["lump_tax"],
                                        lo=0.35, hi=0.70)
    print("  " + labour_search.describe())

    # =========================================================================
    # 3. SCENARIO C: VAT FINANCING
    # =========================================================================
    print("\n--- Scenario C: VAT financing ---")
    vat_search = find_balancing_rate(higher_benefits, "tau_c",
                                     target=base["lump_tax"],
                                     lo=VAT_BASE, hi=0.60)
    print("  " + vat_search.describe())
    if not vat_search.found:
        raise SystemExit("VAT financing found no balancing rate -- inspect the scan.")
    vat = vat_search.solution
    print(f"  VAT {VAT_BASE:.1%} -> {vat_search.rate:.2%} "
          f"({(vat_search.rate - VAT_BASE) * 100:+.2f}pp)")

    # The recursion claim, checked rather than asserted.
    unfunded = solve(higher_benefits, guess=base)   # same tau_w, VAT untouched
    assert abs(unfunded["u"] - vat["u"]) < 1e-10, "VAT moved unemployment?"
    assert abs(unfunded["l_i"] - vat["l_i"]) < 1e-10, "VAT moved informal labour?"
    print(f"\n  [CHECK] Raising VAT by "
          f"{(vat_search.rate - VAT_BASE) * 100:.2f}pp moved unemployment by "
          f"{abs(unfunded['u'] - vat['u']):.2e} and informal labour by "
          f"{abs(unfunded['l_i'] - vat['l_i']):.2e}.")
    print("  The labour block is recursive: VAT is non-distortionary here by "
          "construction, not by result. See the module docstring.")

    # =========================================================================
    # 4. OUTPUT
    # =========================================================================
    os.makedirs("results", exist_ok=True)
    write_csv("results/vat_experiment.csv", [
        {"scenario": "baseline", "balances_budget": True, **base.summary()},
        {"scenario": "labour_tax_40pct", "balances_budget": False,
         **labour.summary()},
        {"scenario": "vat_financed", "balances_budget": True, **vat.summary()},
    ])
    print("\n[SUCCESS] Data saved to results/vat_experiment.csv")

    def pchg(new, old):
        return ((new - old) / old) * 100

    scenarios = ["Baseline",
                 "Labour Tax 40%\n(not balanced)",
                 f"VAT {vat_search.rate:.1%}\n(balanced)"]
    u_vals = [base["u"] * 100, labour["u"] * 100, vat["u"] * 100]
    welf_vals = [0, pchg(labour["c_spd"], base["c_spd"]),
                 pchg(vat["c_spd"], base["c_spd"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Labour Tax vs. VAT: Financing Social Protection in Turkey",
                 fontsize=16)

    bars1 = ax1.bar(scenarios, u_vals, color=["gray", "#8c0000", "#2ca02c"],
                    alpha=0.8)
    ax1.set_ylabel("Unemployment Rate (%)")
    ax1.set_title("Impact on Jobs")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f"{bar.get_height():.2f}%", ha="center", va="bottom",
                 weight="bold")

    bars2 = ax2.bar(scenarios, welf_vals, color=["gray", "red", "green"],
                    alpha=0.8)
    ax2.set_ylabel("% Change in Consumption (Poor)")
    ax2.set_title("Impact on Poverty (Spenders)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h,
                 f"{h:+.2f}%", ha="center",
                 va="bottom" if h >= 0 else "top", weight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/turkey_vat_experiment.png", dpi=300)
    print("[SUCCESS] Chart saved to results/turkey_vat_experiment.png")
    plt.show()


if __name__ == "__main__":
    main()
