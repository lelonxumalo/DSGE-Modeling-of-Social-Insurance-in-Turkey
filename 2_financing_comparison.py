"""
How should Turkey pay for a bigger safety net?

Expands the transfer to the unemployed by a third (15% -> 20% of the formal
wage) and funds it three ways:

  * unfunded  -- savers absorb it through the lump-sum residual
  * labour tax -- tau_w rises until savers are back to their baseline burden
  * VAT        -- tau_c rises until the same

Both budget-closure searches scan for sign changes rather than bracketing the
endpoints, because the fiscal gap is not monotone in the tax rate: the labour
tax has two balancing rates, one either side of the Laffer peak.

Output: results/v2_financing_comparison.png
        results/v2_financing_comparison.csv
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from turkey_tank.dual import balancing_rates, calibrate, solve, write_csv

BASE_RR, NEW_RR = 0.15, 0.20
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
NEUTRAL, LABOUR, VAT = "#9a9992", "#2a78d6", "#eb6834"


def main():
    pars, base = calibrate()
    target = base["lump_tax"]
    expanded = pars.at(rr=NEW_RR)

    print(f"Baseline: unemployment {base.unemployment:.2%}, informality "
          f"{base.informal_share:.2%}, fiscal balance {target:+.4f}")
    print(f"\nExpanding the transfer to the unemployed from {BASE_RR:.0%} to "
          f"{NEW_RR:.0%} of the formal wage (+33%).")

    unfunded = solve(expanded, guess=base)
    print(f"  Unfunded: unemployment {unfunded.unemployment:.2%}, "
          f"fiscal gap {unfunded['lump_tax'] - target:+.4f}")

    tw_roots = balancing_rates(expanded, "tau_w", target, pars.tau_w, 0.70, guess=base)
    tc_roots = balancing_rates(expanded, "tau_c", target, pars.tau_c, 0.60, guess=base)
    print(f"\n  Labour tax rates that balance the budget: "
          f"{', '.join(f'{r:.2%}' for r in tw_roots) or 'none'}")
    if len(tw_roots) > 1:
        print(f"    -> two roots; {tw_roots[0]:.2%} is the efficient one, "
              f"{tw_roots[-1]:.2%} sits past the Laffer peak.")
    print(f"  VAT rates that balance the budget: "
          f"{', '.join(f'{r:.2%}' for r in tc_roots) or 'none'}")

    labour = solve(expanded.at(tau_w=tw_roots[0]), guess=base)
    vat = solve(expanded.at(tau_c=tc_roots[0]), guess=base)

    scenarios = [
        ("Baseline", base, NEUTRAL),
        ("Unfunded\n(savers absorb)", unfunded, NEUTRAL),
        (f"Labour tax\n{pars.tau_w:.1%} to {tw_roots[0]:.1%}", labour, LABOUR),
        (f"VAT\n{pars.tau_c:.1%} to {tc_roots[0]:.1%}", vat, VAT),
    ]

    def pchg(s, attr):
        a = getattr(s, attr) if isinstance(attr, str) else attr(s)
        b = getattr(base, attr) if isinstance(attr, str) else attr(base)
        return (a / b - 1) * 100

    print("\n" + "=" * 78)
    print(f"{'':<28}{'Unemp':>8}{'Informal':>10}{'GDP %':>9}{'C poor %':>10}{'C rich %':>10}")
    print("=" * 78)
    for label, s, _ in scenarios:
        flat = label.replace("\n", " ")
        print(f"{flat:<28}{s.unemployment:>8.2%}{s.informal_share:>10.2%}"
              f"{pchg(s, lambda x: x['y_tot']):>+9.2f}{pchg(s, 'real_c_spd'):>+10.2f}"
              f"{pchg(s, 'real_c_sav'):>+10.2f}")

    print("\nThe unemployment column barely moves across financing choices: in a")
    print("dual labour market the cost of taxation shows up as informality and")
    print("lost output, not as measured unemployment.")

    # ---------------------------------------------------------------- output
    os.makedirs("results", exist_ok=True)
    write_csv("results/v2_financing_comparison.csv",
              [{"scenario": lab.replace("\n", " "), **s.summary()}
               for lab, s, _ in scenarios])
    print("\n[SUCCESS] Data saved to results/v2_financing_comparison.csv")

    labels = [lab for lab, _, _ in scenarios]
    colors = [c for _, _, c in scenarios]
    panels = [
        ("Unemployment rate", [s.unemployment * 100 for _, s, _ in scenarios],
         "%", "{:.2f}%"),
        ("Real GDP vs. baseline", [pchg(s, lambda x: x["y_tot"]) for _, s, _ in scenarios],
         "%", "{:+.2f}%"),
        ("Consumption of the poor vs. baseline",
         [pchg(s, "real_c_spd") for _, s, _ in scenarios], "%", "{:+.2f}%"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2))
    fig.patch.set_facecolor("#fcfcfb")
    fig.suptitle("Financing a bigger safety net: unemployment barely notices; "
                 "output and the poor do",
                 fontsize=14, color=INK, y=0.99)

    for ax, (title, values, _, fmt) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        bars = ax.bar(range(len(values)), values, color=colors, width=0.62)
        ax.set_title(title, fontsize=11, color=INK, pad=12)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.5, color=MUTED)
        ax.axhline(0, color=GRID, linewidth=1.2, zorder=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", labelsize=8.5, colors=MUTED, length=0)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%" if abs(y) >= 2
                                     else f"{y:.2f}%")
        span = max(values) - min(min(values), 0) or 1
        for bar, value in zip(bars, values):
            off = span * 0.035 * (1 if value >= 0 else -1)
            ax.text(bar.get_x() + bar.get_width() / 2, value + off, fmt.format(value),
                    ha="center", va="bottom" if value >= 0 else "top",
                    fontsize=9.5, color=INK, fontweight="bold")
        lo = min(min(values), 0) - span * 0.18
        ax.set_ylim(lo, max(values) + span * 0.20)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig("results/v2_financing_comparison.png", dpi=200,
                facecolor=fig.get_facecolor())
    print("[SUCCESS] Chart saved to results/v2_financing_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
