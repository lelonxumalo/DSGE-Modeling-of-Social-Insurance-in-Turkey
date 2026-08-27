"""
Calibrate the dual labour market model to Turkish data and report the fit.

Five parameters (matching efficiency, vacancy cost, informal TFP, the CES
weight on formal goods, home production) are solved for jointly with the model
so that it reproduces five moments. Everything else is set a priori.

Output: results/calibration.csv
"""

import os

from turkey_tank.dual import Params, Targets, calibrate, write_csv

# Moments the calibration is asked to hit, and where they come from. These are
# approximate 2024-25 figures -- re-check them against the current releases
# before quoting the model's results anywhere.
SOURCES = {
    "unemployment": "TurkStat labour force survey, headline rate",
    "informal_share": "TurkStat, unregistered share of total employment",
    "wage_gap": "formal/informal earnings ratio, survey estimates",
    "tightness": "normalisation (v/u = 1)",
    "informal_price": "normalisation (informal good is the numeraire pair)",
}


def main():
    targets = Targets()
    pars, sol = calibrate(verbose=True)

    print("\n" + "=" * 74)
    print(" CALIBRATION FIT")
    print("=" * 74)
    print(f"{'Moment':<34}{'Target':>10}{'Model':>10}   Source")
    print("-" * 74)
    fit = [
        ("Unemployment rate", targets.unemployment, sol.unemployment, "{:.2%}",
         SOURCES["unemployment"]),
        ("Informal share of employment", targets.informal_share, sol.informal_share,
         "{:.2%}", SOURCES["informal_share"]),
        ("Formal / informal wage gap", targets.wage_gap, sol.wage_gap, "{:.2f}",
         SOURCES["wage_gap"]),
        ("Market tightness (v/u)", targets.tightness, sol["theta"], "{:.2f}",
         SOURCES["tightness"]),
        ("Relative price, informal good", targets.informal_price, sol["p"], "{:.2f}",
         SOURCES["informal_price"]),
    ]
    for label, target, model, fmt, src in fit:
        print(f"{label:<34}{fmt.format(target):>10}{fmt.format(model):>10}   {src}")

    print("\n" + "=" * 74)
    print(" NOT TARGETED -- these are model output, and a check on plausibility")
    print("=" * 74)
    implied = [
        ("Job finding rate (quarterly)", sol["p_find"], "{:.1%}"),
        ("Vacancy filling rate (quarterly)", sol["p_fill"], "{:.1%}"),
        ("Informal share of GDP", sol.informal_output_share, "{:.1%}"),
        ("Informal share of consumption", 1 - pars.omega, "{:.1%}"),
        ("Vacancy costs / formal output", pars.vac_cost * sol["v"] / sol["y_f"], "{:.1%}"),
        ("Home production / formal wage", pars.home_prod / sol["w_f"], "{:.1%}"),
        ("Consumption ratio, savers/spenders", sol.inequality_ratio, "{:.2f}"),
        ("Tax revenue / GDP", sol["gov_rev"] / sol["y_tot"], "{:.1%}"),
    ]
    for label, value, fmt in implied:
        print(f"{label:<38}{fmt.format(value):>10}")

    print("\nNote: the informal share of GDP is well below the informal share of")
    print("employment because informal work is low-productivity and uses no capital.")
    print("It is not comparable to MIMIC-style 'shadow economy' estimates of ~30% of")
    print("GDP, which measure a different object (including under-reporting by")
    print("registered firms).")

    sol.report("CALIBRATED BASELINE")

    os.makedirs("results", exist_ok=True)
    write_csv("results/calibration.csv", [{"scenario": "baseline", **sol.summary()}])
    print("\n[SUCCESS] Data saved to results/calibration.csv")


if __name__ == "__main__":
    main()
