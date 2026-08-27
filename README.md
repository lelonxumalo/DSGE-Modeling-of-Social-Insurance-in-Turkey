# DSGE Modeling of Social Insurance in Turkey

A Dynamic Stochastic General Equilibrium (DSGE) model analyzing social protection policies in Turkey using a Two-Agent New Keynesian (TANK) framework with dual labor markets (formal and informal sectors). See writeup on my [blog](https://lelonxumalo.blogspot.com/2026/01/fiscal-limits-of-social-protection-in.html)

## Overview

This project implements a computational model to study the impact of social insurance expansions on Turkey's labor market, focusing on the trade-offs between:
- **Formal employment** (regulated, taxed, with search frictions)
- **Informal employment** (unregulated, untaxed, flexible)
- **Unemployment** and social protection
- **Fiscal sustainability** and policy financing methods

## Features

- **Dual Labor Market**: Formal sector with search-and-matching frictions vs. informal sector with flexible wages
- **Household Heterogeneity**: Savers (capital owners) vs. Spenders (liquidity-constrained)
- **Fiscal Policy**: Labor taxes, unemployment benefits, and various financing mechanisms
- **Turkish Calibration**: Parameters calibrated to Turkey's economic context (2024-2025)

## Scripts

### 1. `1_turkey_tank.py` - Baseline Model
The core TANK-SM model calibrated for Turkey.

**Key Features:**
- Formal sector with Nash bargaining and vacancy posting
- Informal sector as "survival" option
- 35% labor tax rate and unemployment benefits
- Policy experiment: +33% increase in unemployment benefits

**Output:** `results/turkey_policy_impact_dual_axis.png`

**Run:**
```bash
python 1_turkey_tank.py
```

### 2. `2_financing_experiments.py` - Fiscal Closure Comparison
Compares different methods of financing social protection expansion.

**Scenarios:**
- **Baseline**: Current policy
- **Debt-financed**: Benefits funded via lump-sum taxes on savers
- **Labor tax-financed**: Benefits funded by raising formal sector labor taxes

**Analysis:** Shows who bears the cost and the differential impact on unemployment

**Output:** `results/turkey_tax_experiment.png`

**Run:**
```bash
python 2_financing_experiments.py
```

### 3. `3_turkey_tank_laffer.py` - Laffer Curve Analysis
Traces the relationship between labor tax rates (0-70%) and government revenue.

**Key Questions:**
- What is the revenue-maximizing tax rate?
- At what point does the formal sector collapse?
- How does informality grow with tax rates?

**Output:** `results/turkey_laffer_curve.png`

**Run:**
```bash
python 3_turkey_tank_laffer.py
```

### 4. `4_turkey_tank_vat_experiment.py` - VAT vs. Labor Tax
Tests whether consumption taxes (VAT) are less distortive than labor taxes for funding social protection.

**Hypothesis:** VAT is broader and doesn't directly affect the formal sector hiring decision.

**Output:** `results/turkey_vat_experiment.png`

**Run:**
```bash
python 4_turkey_tank_vat_experiment.py
```

## Installation

### Requirements
- Python 3.9+ (developed on 3.13)

No solver licence is needed. The model was previously written in GAMSPy and
required a GAMS backend; it is now a plain `scipy.optimize.root` solve.

### Setup

1. Clone this repository:
```bash
git clone https://github.com/lelonxumalo/DSGE-Modeling-of-Social-Insurance-in-Turkey.git
cd DSGE-Modeling-of-Social-Insurance-in-Turkey
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Verify the install:
```bash
python -m pytest test_model.py -q
```

## Model Structure

### Households
- **Savers (50%)**: Own capital, work in formal sector, smooth consumption over time
- **Spenders (50%)**: Hand-to-mouth, work in both formal and informal sectors

### Production
- **Formal Sector**: Capital-intensive (Cobb-Douglas), subject to labor taxes
- **Informal Sector**: Labor-only, lower productivity, untaxed

### Labor Market
- **Formal**: Search-and-matching with job separation rate of 7%
- **Informal**: Walrasian spot market (flexible wages)
- **Unemployment**: Benefits provided at 30% of wage (baseline)

### Government
- **Revenue**: Labor income taxes (35% baseline)
- **Expenditure**: Unemployment benefits
- **Closure**: Balanced budget via lump-sum adjustments

## Key Parameters (Turkey Calibration)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `beta` | 0.985 | Discount factor (reflects high real interest rates) |
| `alpha` | 0.40 | Capital share (formal sector) |
| `tau_w` | 0.35 | Labor tax rate (OECD tax wedge) |
| `p_sep_rate` | 0.07 | Formal job separation rate (high turnover) |
| `p_match_eff` | 0.35 | Matching efficiency (structural unemployment) |
| `p_informal_prod` | 1.2 | Informal sector productivity |
| `p_ben_level` | 0.3 | Unemployment benefit **level** (not a rate). The implied replacement rate is `p_ben_level / w_f` &asymp; 11% |

## Model code

The equations live in `turkey_tank/model.py` and are shared by all four
experiment scripts. Sweep and budget-closure helpers are in
`turkey_tank/experiments.py`. Each script writes both a PNG and a CSV so the
numbers behind published tables can be regenerated.

Two accounting bugs were fixed after the write-up was published:

1. `firm_profit` did not net out the capital rental bill, so savers were paid
   `r*k` twice (about a third of GDP).
2. The government paid `p_ben_level * u` in benefits while only spenders
   received transfers, so half of every benefit lira vanished.

Together these broke the aggregate resource constraint. `Solution.check()` now
asserts `C + I == Y - vacancy costs` after every solve. Pass
`Params(legacy_accounting=True)` to reproduce the pre-fix published numbers;
`test_model.py` pins both sets.

## Modelling caveats

Read these before quoting results.

- **The labour block is recursive.** The eleven equations determining
  `y_f, k, r, l_f, u, v, theta, w_f, w_i, p_find, p_fill` contain no
  consumption variable and no `tau_c`, so **VAT cannot move unemployment in
  this model**. A "VAT preserves jobs" comparison is really a comparison of
  labour tax rates. Making VAT distortionary requires exempting informal
  consumption, which is not yet implemented.
- **`u` is not a headline unemployment rate.** `l_f + u == 1` normalises the
  *formal* labour force; informal workers sit outside that identity. Solutions
  report both `unemployment_formal` and a TurkStat-comparable
  `unemployment_total`.
- **Informal work is a second job, not an exit.** Informal labour is supplied
  by formally-employed spenders and never feeds back into `u`, so the model
  does not contain a "workers exit the formal sector" margin.
- **The outside option double-counts.** `w_i + p_ben_level` gives a separated
  worker the informal wage *and* the benefit simultaneously, which amplifies
  the disemployment effect of benefits.
- **Calibration is not target-matched.** Parameters were set by hand, not
  solved to hit Turkish moments.

## Results

All visualizations are saved in the `results/` folder:
- Labor market dynamics (unemployment, informality)
- Welfare impacts (consumption changes)
- Fiscal trade-offs (revenue vs. costs)
- Policy comparison charts

## Research Questions Addressed

1. **Does expanding social protection help or hurt the poor?**
   - Trade-off: Higher benefits vs. job destruction

2. **Who should pay for social insurance?**
   - Rich (debt/lump-sum) vs. workers (labor taxes)

3. **Is there a fiscal limit to labor taxation?**
   - Laffer curve analysis

4. **Can VAT reduce labor market distortions?**
   - Consumption tax as alternative financing

## Citation

If you use this model in your research, please cite:

```
Nxumalo, Mpumelelo. 2025. Fiscal Limits of Social Protection in Turkey: A General Equilibrium Analysis. https://lelonxumalo.blogspot.com/2026/01/fiscal-limits-of-social-protection-in.html
```

## License

MIT

## Acknowledgments

This model builds on the TANK framework and incorporates insights from Turkish labor market research.
