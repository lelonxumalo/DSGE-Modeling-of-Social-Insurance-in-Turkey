# Social Insurance and Informality in Turkey

A steady-state two-agent model of Turkey's dual labour market with search and
matching, used to study who bears the cost of expanding social protection.

The model is deterministic: there are no stochastic shocks, no transition
dynamics and no expectations. It solves a square nonlinear system for a single
steady state and compares steady states across policies.

> **Status.** This repository contains one model, calibrated to Turkish data and
> covered by tests. An earlier version of it produced the January 2026 write-up;
> those results were **withdrawn** — two accounting errors and a VAT that could
> not affect employment by construction — and that code has been removed. It
> remains reachable at the `v1-january-writeup` tag if the retracted numbers ever
> need auditing.

## Overview

The project studies how social insurance expansions land on Turkey's labour
market, and in particular the trade-offs between:

- **Formal employment** (regulated, taxed, hired through search frictions)
- **Informal employment** (unregistered, untaxed, lower-paid)
- **Unemployment** and social protection
- **Fiscal sustainability** and the choice of financing instrument

## Features

- **Dual labour market with a real margin**: $l_f + l_i + u = 1$, and workers not
  in a formal job choose between informal work and searching, so `u` is a rate
  comparable to TurkStat's.
- **Household heterogeneity**: savers (capital owners, who absorb the fiscal
  residual) vs. spenders (hand-to-mouth).
- **VAT that can be evaded**: households buy a CES bundle of formal and informal
  goods and VAT applies only to the formal one, so consumption taxes reach the
  labour market through the informal margin.
- **Target-matched calibration**: five parameters are solved for jointly with the
  model to reproduce five Turkish moments. See [Calibration](#calibration).

Scripts are numbered in reading order:

```bash
python 1_calibrate.py              # calibration fit against Turkish data
python 2_financing_comparison.py   # payroll tax vs. VAT financing
python 3_laffer_curves.py          # both Laffer curves
```

See [Repository layout](#repository-layout) for how the pieces fit together.

Every number this README quotes is recomputed from a fresh solve by
`test_docs.py`, cell by cell against the headline table — which is how the
January write-up's numbers are kept from going stale again.

## Headline results

Raising the transfer to the unemployed by a third (15% → 20% of the formal wage):

| Scenario | Unemployment | Informal share | Real GDP | Consumption of the poor |
|---|---|---|---|---|
| Baseline | 8.60% | 27.00% | — | — |
| Unfunded (savers absorb) | 9.20% | 26.64% | -0.23% | +0.98% |
| Payroll tax 37.5% → 38.63% | 9.19% | 27.25% | **-0.91%** | **-0.60%** |
| VAT 20.0% → 20.60% | 9.18% | 26.82% | -0.35% | +0.52% |

Unemployment is nearly identical across all three. The cost of the financing
choice lands on informality and output instead — and under payroll-tax financing
the poor end up worse off than before the transfer was raised. The ranking
survives every robustness check in `test_dual.py`; the magnitudes do not.

## Model structure

### Households
- **Savers (50%)**: own the capital stock and the formal firms, and absorb the
  government's lump-sum residual.
- **Spenders (50%)**: hand-to-mouth.
- Both face the same labour market; formal jobs, informal jobs and unemployment
  are allocated across types in population proportion.

### Production
- **Formal**: Cobb-Douglas in capital and labour, constant returns, taxed.
- **Informal**: labour only, decreasing returns ($\gamma = 0.85$), untaxed.
  Workers are paid the average product, so all informal value added accrues to
  informal workers.

### Labour market
- **Identity**: $l_f + l_i + u = 1$ — every worker is formally employed,
  informally employed, or unemployed and searching.
- **Formal**: search and matching; firms post vacancies until expected profit
  covers the cost; the wage splits the match surplus by Nash bargaining with a
  tax wedge.
- **Informal**: workers not in a formal job are indifferent between informal work
  and searching, which is what pins down the size of the sector.
- **Unemployment**: the transfer is $rr \cdot w_f$, i.e. 15% of the formal wage at
  baseline and 20% in the expansion experiment. This is an *effective* rate —
  Turkey's statutory UI replaces about 40% of prior earnings but reaches a small
  minority of the unemployed. The remaining value of non-employment is home
  production, calibrated as a residual.

### Consumption
- CES bundle of formal and informal goods, elasticity $\sigma = 2$.
- **VAT applies to the formal good only.** This is the channel through which
  consumption taxes reach employment; without it VAT is non-distortionary by
  construction, which is the defect that sank the January write-up.

### Government
- **Revenue**: labour tax on formal wages, plus VAT on formal consumption.
- **Expenditure**: transfers to the unemployed.
- **Closure**: a lump-sum tax on savers clears the budget.

## Calibration

Five parameters are solved for **jointly with the model** — the target conditions
are appended to the equation system rather than wrapped in an outer loop — so a
single solve delivers both. Values below are the output of `calibrate()`; run
`python 1_calibrate.py` to regenerate them.

### Solved to hit the targets

| Parameter | Value | Description |
|-----------|-------|-------------|
| `match_eff` | 0.5431 | Matching efficiency |
| `vac_cost` | 1.4170 | Vacancy posting cost |
| `z_i` | 1.1838 | Informal sector productivity |
| `omega` | 0.8977 | CES weight on the formal good |
| `home_prod` | 0.2046 | Flow value of home production |

### Set a priori

| Parameter | Value | Description |
|-----------|-------|-------------|
| `beta` | 0.985 | Quarterly discount factor (high Turkish real rates) |
| `alpha` | 0.40 | Capital share, formal sector |
| `delta` | 0.025 | Depreciation |
| `eps` | 0.5 | Matching elasticity |
| `eta` | 0.5 | Worker bargaining power |
| `sep` | 0.07 | Quarterly formal separation rate |
| `gamma` | 0.85 | Returns to labour, informal sector |
| `sigma` | 2.0 | CES elasticity, formal vs. informal goods |
| `tau_w` | 0.375 | Labour tax wedge (OECD *Taxing Wages*) |
| `tau_c` | 0.20 | VAT, standard rate in force since July 2023 |
| `rr` | 0.15 | Transfer to the unemployed, share of the formal wage |
| `pop_savers` | 0.50 | Saver share of the population |

### Fit

| Moment | Target | Model | |
|--------|--------|-------|---|
| Unemployment rate | 8.6% | 8.60% | targeted |
| Informal share of employment | 27% | 27.00% | targeted |
| Formal / informal wage gap | 1.75x | 1.75x | targeted |
| Market tightness (v/u) | 1.00 | 1.00 | normalisation |
| Relative price, informal good | 1.00 | 1.00 | normalisation |
| **Tax revenue / GDP** | ~31% | **31.3%** | **not targeted** |
| Quarterly job-finding rate | — | 54.3% | not targeted |
| Vacancy costs / formal output | — | 3.9% | not targeted |

Revenue-to-GDP is the useful external check: nothing in the calibration aims at
it and it lands in Turkey's actual low-thirties range.

Targets are approximate 2024–25 figures — the unemployment rate and unregistered
employment share from TurkStat's labour force survey, the tax wedge from OECD
*Taxing Wages*, and the wage gap from survey estimates. **Verify them against
current releases before quoting the model's results.**

## Model code

`turkey_tank/dual.py` holds the whole model: equations, calibration and budget
closure.

It solves in logs for the positive variables, reproducing the bounds that
GAMSPy's `type="positive"` used to supply, and uses
`least_squares(method="trf", x_scale="jac")` rather than `root`: capital is two
orders of magnitude larger than most variables, and `root(method="lm")` stalls
on the resulting badly-scaled Jacobian and reports the stall as non-existence.

Each script writes a CSV alongside its chart, so published tables can be
regenerated rather than re-read from console output.

## Repository layout

```
1_calibrate.py              solve the model and report the calibration fit
2_financing_comparison.py   fund a benefit expansion three ways, compare
3_laffer_curves.py          sweep each tax instrument on its own

turkey_tank/dual.py         the model: equations, calibration, budget closure

results/                    every chart and CSV the scripts produce
test_dual.py                calibration, identities, mechanisms, robustness
test_docs.py                recomputes every figure this README quotes
```

**The numbers are reading order, not a dependency chain.** Each script is
self-contained: it calls `calibrate()` itself, solves from scratch, and writes
its own outputs. Nothing reads another script's CSV, so they can be run in any
order or individually. The model is deterministic, so every script recomputes
the same calibrated parameters — running `1_calibrate.py` first is a convention
for readers, not a precondition.

The shape of a run is the same in all three:

1. `calibrate()` solves the 27-equation system jointly with five target
   conditions, returning the calibrated `Params` and the baseline `Solution`.
2. The script varies one or two policy parameters — `rr`, `tau_w`, `tau_c` —
   and re-solves, passing the previous solution as the starting guess so
   continuation carries it through difficult regions.
3. `Solution.check()` asserts market clearing after every solve, so a bad
   equilibrium raises rather than propagating into a chart.
4. Results go to `results/` as both a PNG and a CSV, and the console report is
   part of the output.

Budget-closing scenarios add a step: `balancing_rates()` scans for the tax rate
that returns the fiscal residual to its baseline. It returns a *list*, because
the payroll tax has two such rates, one either side of its Laffer peak.

## Modelling caveats

Read these before quoting results.

- **VAT moves quantities, not wages.** The formal sector is CRS and the interest
  rate is pinned by `beta`, so the formal marginal product of labour is constant.
  Free entry, the Nash wage, the surplus definition and the informal-search
  indifference close a six-equation block in `(w_f, w_i, s_w, theta, p_find,
  p_fill)` containing no price and no quantity. VAT reallocates workers between
  the three labour states without changing what any of them is paid.
- **VAT regressivity is understated.** Both household types consume the same CES
  bundle, so the model misses that poorer households spend more of their income
  on necessities. A real VAT rise would hit the poor harder than this says.
- **Enforcement is fixed.** Informality responds to taxes but not to audit
  intensity, penalties or formalisation incentives — so the model cannot evaluate
  raising the payroll tax *and* tightening enforcement.
- **Steady-state only.** No transition path, and no inflation channel, which for
  a VAT increase in Turkey is a real omission.
- **No participation margin.** The labour force is fixed at 1, so Turkey's low
  female participation rate is invisible to the model.
- **Two parameters lack a Turkish anchor**: `sigma` and `eta`. `test_dual.py`
  sweeps both; the instrument ranking holds throughout, the magnitudes do not,
  and at $\eta = 0.35$ payroll financing stops hurting the poor outright.

## Research questions addressed

1. **Does expanding social protection help or hurt the poor?**
   It depends entirely on how it is financed. Unfunded it helps (+0.98%); funded
   by VAT it still helps (+0.52%); funded by payroll taxes it *hurts* (-0.60%),
   because the higher wedge falls on the formal wages most of them earn.

2. **Who should pay for social insurance?**
   The broad consumption base beats the payroll base. The same revenue costs
   0.91% of output raised through the payroll tax against 0.35% through VAT, and
   the ranking survives every robustness check tested.

3. **Is there a fiscal limit to labour taxation?**
   Yes, but not where the January write-up put it. Total revenue peaks at a wedge of
   about **53%**, so Turkey's ~37.5% is comfortably on the right side and there
   is no fiscal cliff. The binding argument is the marginal cost of those funds,
   not insolvency — walking the wedge from 5% to 70% takes informal employment
   from 15% to 56%.

4. **Can VAT reduce labour market distortions?**
   Yes, relative to the payroll tax — but VAT is not free. It leaks into
   informality too, taking the informal share from 21% to 38% across its sweep,
   against 15% to 56% for the payroll tax. It is a flatter leak, not no leak.

## Results

| File | Contents |
|------|----------|
| `financing_comparison.png` | Unemployment, GDP and consumption of the poor under each financing route |
| `laffer_curves.png` | Revenue and informality against each instrument, on shared scales |
| `calibration.csv` | Calibrated baseline, full solution vector |
| `financing_comparison.csv` | The four scenarios |
| `laffer_curves.csv` | Both sweeps, with an `instrument` column |

Every CSV carries all parameters (`param_*`), all solved variables and the
derived statistics.

## Installation

### Requirements
- Python 3.9+ (developed on 3.13)

No solver licence is needed. The model was previously written in GAMSPy and
required a GAMS backend; it is now plain `scipy.optimize`.

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
python -m pytest -q            # full suite
```

> **Note on the repository name.** The directory and clone URL still say
> `DSGE-Modeling-of-Social-Insurance-in-Turkey`, which is inaccurate for the same
> reason the old title was: nothing here is dynamic or stochastic. See
> [Renaming this repository](#renaming-this-repository).

## Development

```bash
python -m pytest -q                        # all tests
python -m pytest test_dual.py -q           # the model
python -m pytest test_docs.py -q           # this README's numbers
python -m pytest test_dual.py -q -k vat    # a single group
```

`test_dual.py` pins the calibration, the accounting identities, the three
mechanisms the argument rests on, and the robustness sweeps. `test_docs.py`
recomputes every figure this README quotes. `Solution.check()` asserts market
clearing after every solve.

Scripts call `plt.show()`, which blocks; set `MPLBACKEND=Agg` to run them
non-interactively. Charts and CSVs are written before the show call.

## Renaming this repository

The name `DSGE-Modeling-of-Social-Insurance-in-Turkey` describes a model this
never was. If you rename it — `turkey-dual-labour-market` or
`turkey-social-insurance-model` would both be accurate — here is what happens.

**What keeps working.** GitHub permanently redirects the old URL to the new one
for web traffic, clones, fetches and pushes, and for the API. Stars, watchers,
forks, issues, pull requests and releases all move with the repository. Existing
local clones keep working through the redirect, though `git remote set-url` is
worth running anyway.

**What breaks.**

- The redirect dies the moment anyone — including you — creates a new repository
  under the old name. That is the only way to lose it, and it is silent.
- GitHub Pages URLs change if Pages is enabled on a `github.io` path.
- Anything hardcoding the old URL: the clone command in this README, the
  citation URL below, and the links in the write-up draft.
- The local working directory name, which is independent of the remote and has to
  be renamed separately.
- Any DOI or archive snapshot (Zenodo and similar) pins the old name in already-
  minted records; those cannot be rewritten.

**Note to add to the new repository's README if you do rename:**

> Formerly `DSGE-Modeling-of-Social-Insurance-in-Turkey`. The old name described
> the model inaccurately — it is deterministic and steady-state, with no
> stochastic shocks or dynamics. GitHub redirects the old URL, but please update
> bookmarks and remotes: `git remote set-url origin <new-url>`. Work citing the
> old name refers to the same repository.

## Citation

```
Nxumalo, Mpumelelo. 2026. Turkey's Informality Tax: Social Protection and the Marginal Cost of Public Funds.
https://lelonxumalo.blogspot.com/2026/01/fiscal-limits-of-social-protection-in.html
```

The post at that URL is the January 2026 write-up, whose results are withdrawn.
A redraft against the current model exists but is not yet posted; the write-up
and its tooling are kept out of this repository, which ships the model only.

## License

MIT

## Acknowledgments

Builds on the TANK framework and on Turkish labour market research.
