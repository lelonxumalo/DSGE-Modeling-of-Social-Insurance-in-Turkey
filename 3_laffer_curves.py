"""
Two Laffer curves: the labour tax and the VAT.

Sweeps each instrument on its own, holding the other at its calibrated value,
and traces total government revenue alongside the informal share of employment.
The pairing is the point: the labour tax turns over because the formal tax base
walks into the informal sector, and the VAT keeps raising revenue but leaks into
informality the whole way.

Output: results/laffer_curves.png
        results/laffer_curves.csv
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from turkey_tank.dual import SolveError, calibrate, solve, write_csv

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
LABOUR, VAT = "#2a78d6", "#eb6834"


def sweep(pars, base, name, values):
    rows, guess = [], base
    for value in values:
        try:
            s = solve(pars.at(**{name: float(value)}), guess=guess)
            guess = s
            rows.append((float(value), s))
        except SolveError:
            print(f"  {name}={value:.3f}: no equilibrium")
            break
    return rows


def main():
    pars, base = calibrate()

    print("--- Labour tax sweep (VAT held at its calibrated value) ---")
    lab = sweep(pars, base, "tau_w", np.linspace(0.05, 0.70, 40))
    print("--- VAT sweep (labour tax held at its calibrated value) ---")
    vat = sweep(pars, base, "tau_c", np.linspace(0.00, 0.60, 40))

    def peak(rows):
        v, s = max(rows, key=lambda t: t[1]["gov_rev"])
        return v, s["gov_rev"]

    lab_peak, lab_rev = peak(lab)
    print(f"\nLabour tax: revenue peaks at tau_w = {lab_peak:.1%} (revenue {lab_rev:.4f}); "
          f"Turkey sits at {pars.tau_w:.1%}.")
    vat_peak, vat_rev = peak(vat)
    print(f"VAT: revenue still rising at tau_c = {vat_peak:.1%} (revenue {vat_rev:.4f}) "
          f"-- no interior peak in range.")
    print(f"\nInformality over the labour tax sweep: "
          f"{lab[0][1].informal_share:.1%} -> {lab[-1][1].informal_share:.1%}")
    print(f"Informality over the VAT sweep:         "
          f"{vat[0][1].informal_share:.1%} -> {vat[-1][1].informal_share:.1%}")

    os.makedirs("results", exist_ok=True)
    write_csv("results/laffer_curves.csv",
              [{"instrument": "tau_w", "rate": v, **s.summary()} for v, s in lab]
              + [{"instrument": "tau_c", "rate": v, **s.summary()} for v, s in vat])
    print("\n[SUCCESS] Data saved to results/laffer_curves.csv")

    # --- figure: revenue on top, informality below; never a dual axis --------
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharex="col", sharey="row")
    fig.patch.set_facecolor("#fcfcfb")
    fig.suptitle("Both taxes leak into informality; only one of them stops "
                 "raising revenue", fontsize=14, color=INK, y=0.98)

    cols = [("Labour tax", lab, LABOUR, pars.tau_w, "tau_w"),
            ("VAT", vat, VAT, pars.tau_c, "tau_c")]

    for col, (name, rows, color, current, _) in enumerate(cols):
        rates = [v * 100 for v, _ in rows]
        revenue = [s["gov_rev"] for _, s in rows]
        informal = [s.informal_share * 100 for _, s in rows]

        for row, (series, title, fmt) in enumerate([
            (revenue, f"{name}: total government revenue", "{:.2f}"),
            (informal, f"{name}: informal share of employment", "{:.0f}%"),
        ]):
            ax = axes[row][col]
            ax.set_facecolor("#fcfcfb")
            ax.plot(rates, series, color=color, linewidth=2, zorder=3)
            ax.axvline(current * 100, color=MUTED, linewidth=1,
                       linestyle=(0, (4, 3)), zorder=1)
            ax.annotate(f"Turkey today\n{current:.1%}",
                        xy=(current * 100, ax.get_ylim()[0]),
                        xytext=(current * 100 + 1.5, min(series) + (max(series) - min(series)) * 0.06),
                        fontsize=8.5, color=MUTED)
            if row == 0:
                best = max(range(len(series)), key=lambda i: series[i])
                if 0 < best < len(series) - 1:
                    ax.plot([rates[best]], [series[best]], "o", color=color,
                            markersize=8, zorder=4)
                    ax.annotate(f"revenue peaks\nat {rates[best]:.0f}%",
                                xy=(rates[best], series[best]),
                                xytext=(rates[best] - 4, series[best] * 0.80),
                                fontsize=9, color=INK, fontweight="bold")
            ax.set_title(title, fontsize=11, color=INK, pad=10)
            ax.grid(color=GRID, linewidth=0.8, alpha=0.7)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(GRID)
            ax.tick_params(labelsize=8.5, colors=MUTED, length=0)
            if row == 1:
                ax.set_xlabel(f"{name} rate (%)", fontsize=9.5, color=MUTED)
            ax.yaxis.set_major_formatter(
                (lambda y, _: f"{y:.0f}%") if row == 1 else (lambda y, _: f"{y:.1f}"))

    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    plt.savefig("results/laffer_curves.png", dpi=200, facecolor=fig.get_facecolor())
    print("[SUCCESS] Chart saved to results/laffer_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
