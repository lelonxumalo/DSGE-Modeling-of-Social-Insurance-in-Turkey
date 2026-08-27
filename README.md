# Social Insurance and Informality in Turkey

A steady-state two-agent model of Turkey's dual labour market with search and
matching, used to study who bears the cost of expanding social protection.

The model is deterministic: there are no stochastic shocks, no transition
dynamics and no expectations. It solves a square nonlinear system for a single
steady state and compares steady states across policies.

> **Status.** `turkey_tank/dual.py` (**v2**) is current and calibrated to Turkish
> data — use it for everything. `turkey_tank/model.py` (**v1**) is frozen and kept
> only to reproduce the January 2026 write-up, whose results are **withdrawn and
> superseded** by v2. Every section below is labelled v1 or v2; if it isn't
> labelled, it applies to both.

## Overview

The project studies how social insurance expansions land on Turkey's labour
market, and in particular the trade-offs between:

- **Formal employment** (regulated, taxed, hired through search frictions)
- **Informal employment** (unregistered, untaxed, lower-paid)
- **Unemployment** and social protection
- **Fiscal sustainability** and the choice of financing instrument

## Features (v2)

- **Dual labour market with a real margin**: $l_f + l_i + u = 1$, and workers not
  in a formal job choose between informal work and searching, so `u` is a rate
  comparable to TurkStat's.
- **Household heterogeneity**: savers (capital owners, who absorb the fiscal
  residual) vs. spenders (hand-to-mouth).
- **VAT that can be evaded**: households buy a CES bundle of formal and informal
  goods and VAT applies only to the formal one, so consumption taxes reach the
  labour market through the informal margin.
- **Target-matched calibration**: five parameters are solved for jointly with the
  model to reproduce five Turkish moments. See
  [Calibration](#calibration-v2).

## Which model to use

| | v2 (`turkey_tank/dual.py`) | v1 (`turkey_tank/model.py`) |
|---|---|---|
| Labour force | $l_f + l_i + u = 1$ — `u` is TurkStat-comparable | $l_f + u = 1$ — informal workers outside the identity |
| Informality | a margin: informal work vs. search | moonlighting by the formally employed |
| VAT | applies to formal goods only, so it can be evaded | uniform, and cannot affect employment at all |
| Calibration | solved to hit Turkish moments | parameters set by hand |
| Baseline unemployment | 8.60% (matches data) | 21% (not a comparable rate) |
| Solver | `least_squares(method="trf", x_scale="jac")` | `root(method="lm")` |
| Use it for | everything | reproducing the original write-up only |

**Start here** (v2):

```bash
python calibrate_to_turkey.py      # calibration fit against Turkish data
python financing_comparison.py     # payroll tax vs. VAT financing
python laffer_curves.py            # both Laffer curves
```

Every number this README quotes is recomputed from a fresh solve by
`test_docs.py`, cell by cell against the headline table — which is how the
January write-up's numbers are kept from going stale again.

### Headline results (v2)

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

## Model structure (v2)

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
  construction (which is exactly the v1 defect).

### Government
- **Revenue**: labour tax on formal wages, plus VAT on formal consumption.
- **Expenditure**: transfers to the unemployed.
- **Closure**: a lump-sum tax on savers clears the budget.

## Calibration (v2)

Five parameters are solved for **jointly with the model** — the target conditions
are appended to the equation system rather than wrapped in an outer loop — so a
single solve delivers both. Values below are the output of `calibrate()`; run
`python calibrate_to_turkey.py` to regenerate them.

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

| Module | Role |
|--------|------|
| `turkey_tank/dual.py` | **v2, primary.** Equations, calibration, budget closure. |
| `turkey_tank/model.py` | v1, frozen. Kept to reproduce the original write-up. |
| `turkey_tank/experiments.py` | Sweep and budget-closure helpers for v1. |

Both models solve in logs for the positive variables, reproducing the variable
bounds that GAMSPy's `type="positive"` used to supply. v2 uses
`least_squares(method="trf", x_scale="jac")`: capital is two orders of magnitude
larger than most variables, and `root(method="lm")` stalls on the resulting
badly-scaled Jacobian and reports the stall as non-existence.

Each script writes a CSV alongside its chart, so published tables can be
regenerated rather than re-read from console output.

## Modelling caveats (v2)

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

## Research questions addressed (v2)

1. **Does expanding social protection help or hurt the poor?**
   It depends entirely on how it is financed. Unfunded it helps (+0.98%); funded
   by VAT it still helps (+0.52%); funded by payroll taxes it *hurts* (-0.60%),
   because the higher wedge falls on the formal wages most of them earn.

2. **Who should pay for social insurance?**
   The broad consumption base beats the payroll base. The same revenue costs
   0.91% of output raised through the payroll tax against 0.35% through VAT, and
   the ranking survives every robustness check tested.

3. **Is there a fiscal limit to labour taxation?**
   Yes, but not where the v1 write-up put it. Total revenue peaks at a wedge of
   about **53%**, so Turkey's ~37.5% is comfortably on the right side and there
   is no fiscal cliff. The binding argument is the marginal cost of those funds,
   not insolvency — walking the wedge from 5% to 70% takes informal employment
   from 15% to 56%.

4. **Can VAT reduce labour market distortions?**
   Yes, relative to the payroll tax — but VAT is not free. It leaks into
   informality too, taking the informal share from 21% to 38% across its sweep,
   against 15% to 56% for the payroll tax. It is a flatter leak, not no leak.

## Results

### v2 (current)

| File | Contents |
|------|----------|
| `v2_financing_comparison.png` | Unemployment, GDP and consumption of the poor under each financing route |
| `v2_laffer_curves.png` | Revenue and informality against each instrument, on shared scales |
| `v2_calibration.csv` | Calibrated baseline, full solution vector |
| `v2_financing_comparison.csv` | The four scenarios |
| `v2_laffer_curves.csv` | Both sweeps, with an `instrument` column |

### v1 (legacy)

| File | Contents |
|------|----------|
| `turkey_policy_impact_dual_axis.png` | Labour market and welfare impact of the benefit expansion |
| `turkey_tax_experiment.png` | Unemployment and welfare under each fiscal closure |
| `turkey_laffer_curve.png` | Revenue and unemployment against the labour tax rate |
| `turkey_vat_experiment.png` | Labour tax vs. VAT financing |
| `baseline_experiment.csv`, `financing_experiments.csv`, `laffer_curve.csv`, `vat_experiment.csv` | Scenario data for the above |

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
python -m pytest -q            # full suite: test_dual.py (v2) + test_model.py (v1)
```

> **Note on the repository name.** The directory and clone URL still say
> `DSGE-Modeling-of-Social-Insurance-in-Turkey`, which is inaccurate for the same
> reason the old title was: nothing here is dynamic or stochastic. See
> [Renaming this repository](#renaming-this-repository).

## Legacy scripts (v1)

**These reproduce the withdrawn January 2026 write-up and are not maintained.**
Their `u` is a share of the *formal* labour force only, so it is not an
unemployment rate comparable to data. Use v2 for anything current.

### 1. `1_turkey_tank.py` — baseline model
The core TANK-SM model. Formal sector with Nash bargaining and vacancy posting;
informal sector as a "survival" option; 35% labour tax; policy experiment raising
the benefit level 0.30 → 0.40.

**Result:** unemployment 21.02% → 25.40% (+4.38pp), spender consumption +0.43%,
informal labour +3.57%, fiscal cost +61%.

**Output:** `results/turkey_policy_impact_dual_axis.png`, `results/baseline_experiment.csv`

### 2. `2_financing_experiments.py` — fiscal closure comparison
Baseline vs. debt/lump-sum financing vs. labour-tax financing.

**Result:** no labour tax rate funds the expansion. No equilibrium exists above
`tau_w` ≈ 42.33% (see [below](#on-non-existence-v1)), and revenue never closes
the gap before then (closest approach +0.077).

**Output:** `results/turkey_tax_experiment.png`, `results/financing_experiments.csv`

### 3. `3_turkey_tank_laffer.py` — Laffer curve
Sweeps the labour tax over 25 points from 5% to 70%.

**Result:** revenue peaks at `tau_w` ≈ 40% (0.787 model units) with unemployment
at 28.3%; by 46% unemployment is 78% and revenue has fallen to 0.279. No
equilibrium exists above `tau_w` ≈ 45.93%.

**Output:** `results/turkey_laffer_curve.png`, `results/laffer_curve.csv`

### 4. `4_turkey_tank_vat_experiment.py` — VAT vs. labour tax
**Result:** VAT at 21.07% (up 3.07pp from 18%) balances the budget where no
labour tax rate can.

**But:** VAT cannot move unemployment in v1 *by construction* — the labour block
is recursive and contains no consumption variable. The script asserts this,
showing a 3.07pp VAT rise moves unemployment by ~5e-15. This is the defect v2
fixes by exempting informal consumption from VAT.

**Output:** `results/turkey_vat_experiment.png`, `results/vat_experiment.csv`

### On non-existence (v1)

Earlier versions of this README claimed the formal sector "ceases to exist" above
a ceiling found by bisecting until the solver failed. Solver failure is not proof
of non-existence, so here is the argument.

Eliminating the v1 system down to a single quadratic in $x = l_f/u$ gives

$$Q x^{2} + G P x + C = 0, \qquad Q, G, P > 0$$

$$C = (1-\eta)\left[(w_i + b) - \mathrm{mpl}_f\,(1-\tau_w)\right]$$

This is a threshold in exogenous parameters only because **both** `mpl_f` and
`w_i` are constants in v1:

- `mpl_f` because the Euler equation pins `r` and the formal sector has constant
  returns, so the capital–labour ratio, and with it the marginal product, is
  independent of everything else.
- `w_i` because v1's informal sector is **linear** in labour, $y_i = z\,l_i$, so
  its marginal product is the constant `z` however much labour it absorbs, and
  the Walrasian informal wage equals it — `w_i == p_informal_prod`,
  `turkey_tank/model.py:153`. Informal labour adjusts purely on the quantity
  margin: across the tax sweep `l_i` runs from 0.34 to 0.73 while `w_i` stays at
  1.200000000000 to every digit. `test_model.py` pins this.

`b` is the exogenous `p_ben_level`, so every term on the right-hand side is a
parameter. With $Q > 0$ and $GP > 0$ both roots are non-positive unless
$C < 0$, and $x > 0$ for any $u \in (0,1)$. So an interior equilibrium exists
**exactly** while

$$\tau_w \;<\; 1 - \frac{w_i + b}{\mathrm{mpl}_f}$$

which is 45.93% at the baseline benefit and 42.33% at the expanded one — the
after-tax marginal product must still cover the worker's outside option.
`test_model.py` checks the closed form and confirms that 40 random starting
values all converge just below each ceiling and none converges just above it.

The ceilings the scripts *report* (42.13%, 45.77%) sit a little below the
analytical ones because the two roots merge as $C \to 0^{-}$ and the Jacobian becomes
singular, so the solver gives up slightly early. The scripts' numbers are honest
about what was found numerically; the closed form above is the actual boundary.

**This does not carry over to v2.** There informal production has decreasing
returns and its own relative price, $w_i = p\,z_i\,l_i^{\gamma-1}$, so `w_i` is
endogenous and responds to the tax. The corresponding boundary in v2 is a fixed
point to be solved, not a formula — which is why `balancing_rates()` scans
numerically rather than evaluating an expression.

### v1 caveats

$l_f + u = 1$ normalises the *formal* labour force, so `u` is not a comparable
unemployment rate; informal work is moonlighting rather than an exit margin;
$w_i + b$ gives a separated worker the informal wage *and* the benefit
at once; VAT is uniform and so cannot affect employment by construction; and no
parameter is target-matched. All are fixed in v2.

Two accounting bugs were also fixed after the write-up was published: `firm_profit`
did not net out the capital rental bill (savers were paid `r*k` twice), and the
government paid `p_ben_level * u` while only spenders received transfers. Pass
`Params(legacy_accounting=True)` to reproduce the pre-fix published numbers;
`test_model.py` pins both sets.

### Legacy headline numbers (v1 — superseded)

| | Baseline | +33% benefits, debt-financed | Labour tax financed | VAT financed |
|---|---|---|---|---|
| Unemployment (formal LF) | 21.02% | 25.40% | 84.17% at the ceiling | 25.40% |
| Balances the budget | — | no (savers absorb it) | impossible | yes, at `tau_c` = 21.07% |

**These are superseded by the v2 headline table above** and are retained only so
the withdrawn write-up can be audited. The published post reports the required
VAT as 20.6%, computed before the accounting fixes; the corrected v1 figure is
21.07%, and the post's labour-tax column was never budget-balanced.

## Development

```bash
python -m pytest -q                        # all tests
python -m pytest test_dual.py -q           # v2 only
python -m pytest test_model.py -q          # v1 only
python -m pytest test_docs.py -q           # this README's numbers
python -m pytest test_model.py -q -k vat   # a single group
```

`test_dual.py` pins the v2 calibration, the accounting identities, the three
mechanisms the write-up's argument rests on, and the robustness sweeps.
`test_model.py` pins v1 in both its corrected and `legacy_accounting=True` forms,
plus the closed-form existence ceiling. `test_docs.py` recomputes every figure
this README quotes. `Solution.check()` asserts market clearing after every solve
in both models.

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

The January 2026 post at that URL is still the **v1** write-up, whose results are
withdrawn. A v2 redraft exists but is not yet posted; the write-up and its
tooling are kept out of this repository, which ships the model only.

## License

MIT

## Acknowledgments

Builds on the TANK framework and on Turkish labour market research.
