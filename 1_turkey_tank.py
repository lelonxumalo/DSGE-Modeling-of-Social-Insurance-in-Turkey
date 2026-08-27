"""
Baseline TANK Model for Turkey (Social Protection & Informality)
================================================================

Solves the baseline steady state and runs the headline policy experiment:
a +33% increase in the unemployment benefit level (0.30 -> 0.40).

The model itself lives in ``turkey_tank/model.py`` -- this script only sets up
scenarios and reports.  Previously each of the four scripts carried its own
copy of the equations, and they had already diverged: this one was missing the
labour-tax wedge in the Nash wage rule that scripts 2-4 had, which put its
baseline unemployment at 10.6% against their 21.0%.

Output: results/turkey_policy_impact_dual_axis.png
        results/baseline_experiment.csv
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from turkey_tank import Params, solve, write_csv


def main():
    base_p = Params()

    # =========================================================================
    # 1. BASELINE
    # =========================================================================
    print("\n--- Solving Turkey TANK-SM baseline ---")
    base = solve(base_p)
    base.report("TURKEY SOCIAL PROTECTION MODEL - BASELINE")

    # =========================================================================
    # 2. POLICY EXPERIMENT: EXPANDING SOCIAL PROTECTION
    # =========================================================================
    print("\n" + "=" * 48)
    print(" POLICY EXPERIMENT: INCREASE UNEMPLOYMENT BENEFITS (+33%)")
    print("=" * 48)

    shock = solve(base_p.at(p_ben_level=0.40), guess=base)
    shock.report("AFTER SHOCK (benefit level 0.30 -> 0.40)")

    def pchg(new, old):
        return ((new - old) / old) * 100

    c_spd_change = pchg(shock["c_spd"], base["c_spd"])
    li_change = pchg(shock["l_i"], base["l_i"])
    u_change = shock["u"] - base["u"]
    cost_increase = pchg(shock["gov_exp"], base["gov_exp"])

    print("\nPolicy: benefit level 0.30 -> 0.40")
    print(f"  (replacement rate {base.replacement_rate:.1%} "
          f"-> {shock.replacement_rate:.1%} of the formal wage)")
    print("-" * 48)
    print(f"Spender consumption:  {c_spd_change:+.2f}%   (did we help the poor?)")
    print(f"Informal labour:      {li_change:+.2f}%   (did we push people informal?)")
    print(f"Unemployment rate:    {u_change * 100:+.2f}pp (did formal jobs vanish?)")
    print(f"Fiscal cost:          {cost_increase:+.2f}%   (budget impact)")

    if c_spd_change > 0 and li_change > 0:
        print("\nCONCLUSION: trade-off detected -- consumption up, informality up.")
    elif c_spd_change < 0:
        print("\nCONCLUSION: policy failure -- job losses outweigh the benefit rise.")

    # =========================================================================
    # 3. OUTPUT
    # =========================================================================
    os.makedirs("results", exist_ok=True)
    write_csv("results/baseline_experiment.csv", [
        {"scenario": "baseline", **base.summary()},
        {"scenario": "benefits_+33pct", **shock.summary()},
    ])
    print("\n[SUCCESS] Data saved to results/baseline_experiment.csv")

    # --- Dashboard ---
    scenarios = ["Baseline", "Policy Shock (+33% Benefits)"]
    u_vals = [base["u"] * 100, shock["u"] * 100]
    li_vals = [base["l_i"], shock["l_i"]]
    values = [c_spd_change, cost_increase]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Impact of Expanding Social Protection in Turkey (TANK Model)",
                 fontsize=16)

    width = 0.35
    x = np.arange(len(scenarios))

    color1 = "#d62728"
    rects1 = ax1.bar(x - width / 2, u_vals, width,
                     label="Unemployment Rate (%)", color=color1, alpha=0.9)
    ax1.set_ylabel("Unemployment Rate (%)", color=color1, fontsize=12, weight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, max(u_vals) * 1.35)

    ax1b = ax1.twinx()
    color2 = "#ff7f0e"
    rects2 = ax1b.bar(x + width / 2, li_vals, width,
                      label="Informal Labour (Level)", color=color2, alpha=0.9)
    ax1b.set_ylabel("Informal Labour Level (Normalised)", color=color2,
                    fontsize=12, weight="bold")
    ax1b.tick_params(axis="y", labelcolor=color2)
    ax1b.set_ylim(0, max(li_vals) * 1.35)

    ax1.set_title("Labour Market Structural Changes")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects, ax, fmt="{:.2f}"):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(fmt.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold")

    autolabel(rects1, ax1, "{:.2f}%")
    autolabel(rects2, ax1b, "{:.3f}")

    metrics = ["Spender Consumption\n(Welfare)", "Gov Expenditure\n(Fiscal Cost)"]
    bars = ax2.bar(metrics, values, color=["green", "red"], alpha=0.7)
    ax2.set_ylabel("Percentage Change (%)")
    ax2.set_title("Policy Efficiency: Cost vs. Benefit")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f"{height:+.2f}%", ha="center",
                 va="bottom" if height >= 0 else "top",
                 fontsize=12, weight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join("results", "turkey_policy_impact_dual_axis.png")
    plt.savefig(output_path, dpi=300)
    print(f"[SUCCESS] Visualisation saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
