"""
TANK Model for Turkey (Social Protection & Informality)
================================================================

This module adapts the TANK-SM model to the Turkish context.
It introduces a Dual Labor Market (Formal/Informal) and a fiscal authority
to analyze social protection policies.

Key Features for Turkey:
1. Dual Labor Market: 
   - Formal: Search & Matching frictions, Taxed, High Productivity.
   - Informal: Walrasian (Flexible), Untaxed, Low Productivity.
2. Fiscal Block:
   - Labor Income Tax (Social Security Contributions).
   - Unemployment Benefits (Social Protection).
3. Household Heterogeneity:
   - Savers: Own capital, work primarily in Formal sector.
   - Spenders: Liquidity constrained, work in Formal (if matched) 
     and Informal (to smooth consumption).
"""

import gamspy
import pandas as pd
import numpy as np

def main():
    # Create a GAMS Container
    m = gamspy.Container()

    # =========================================================================
    # 1. DEFINE SETS
    # =========================================================================
    s_hh = gamspy.Set(m, "s_hh", records=["savers", "spenders"], description="Household Types")

    # =========================================================================
    # 2. CALIBRATION (TURKISH CONTEXT 2024-2025)
    # =========================================================================
    
    # --- Deep Parameters ---
    # Turkey has high real interest rates -> Lower Beta (~0.985 quarterly)
    beta = gamspy.Parameter(m, "beta", records=0.985, description="Discount Factor")
    
    # Capital share (Formal sector is capital intensive)
    alpha = gamspy.Parameter(m, "alpha", records=0.40, description="Capital Share (Formal)")
    delta = gamspy.Parameter(m, "delta", records=0.025, description="Depreciation Rate")
    
    # --- Labor Market Parameters (RECALIBRATED for Higher Informality) ---
    
    # 1. Informal Productivity
    # Increased to 1.2 to make informal work a viable "survival" option 
    # relative to the high formal wage. This encourages Spenders to supply informal labor.
    p_informal_prod = gamspy.Parameter(m, "p_informal_prod", records=1.2, description="Informal Productivity")
    
    # 2. Labor Disutility (Inverse Frisch Elasticity)
    phi = gamspy.Parameter(m, "phi", records=1.0, description="Inverse Frisch Elasticity")
    
    # 3. Matching Frictions (The "Squeeze")
    # Higher separation rate reflects Turkey's high labor turnover
    p_sep_rate = gamspy.Parameter(m, "p_sep_rate", records=0.07, description="Formal Job Separation Rate")
    
    # Lower efficiency makes matches harder to form, creating structural unemployment
    p_match_eff = gamspy.Parameter(m, "p_match_eff", records=0.35, description="Matching Efficiency")
    
    # Higher vacancy costs discourage formal hiring, limiting formal job slots
    p_vac_cost = gamspy.Parameter(m, "p_vac_cost", records=0.35, description="Vacancy Posting Cost")
    
    p_bargain = gamspy.Parameter(m, "p_bargain", records=0.5, description="Worker Bargaining Power")
    p_match_elast = gamspy.Parameter(m, "p_match_elast", records=0.5, description="Matching Elasticity")

    # --- Fiscal Policy (Social Protection) ---
    # OECD Tax Wedge for Turkey ~35%
    tau_w = gamspy.Parameter(m, "tau_w", records=0.35, description="Formal Labor Tax Rate")
    
    # Unemployment Benefit Level
    # Calibrated to be a meaningful safety net, but low enough to maintain search incentives
    p_ben_level = gamspy.Parameter(m, "p_ben_level", records=0.3, description="Unemployment Benefit Level")

    # Population Share
    # 50% "Spenders" (Liquidity Constrained) reflects high wealth inequality
    pop_data = pd.DataFrame([["savers", 0.5], ["spenders", 0.5]], columns=["s_hh", "value"])
    p_pop_share = gamspy.Parameter(m, "p_pop_share", domain=[s_hh], records=pop_data)

    # =========================================================================
    # 3. VARIABLES
    # =========================================================================
    
    # --- Production & Capital ---
    y_f = gamspy.Variable(m, "y_f", type="positive", description="Formal Output")
    y_i = gamspy.Variable(m, "y_i", type="positive", description="Informal Output")
    y_tot = gamspy.Variable(m, "y_tot", type="positive", description="Total GDP")
    k = gamspy.Variable(m, "k", type="positive", description="Capital Stock")
    i = gamspy.Variable(m, "i", type="positive", description="Investment")
    r = gamspy.Variable(m, "r", type="positive", description="Rental Rate")
    
    # --- Labor Market (Aggregate) ---
    l_f = gamspy.Variable(m, "l_f", type="positive", description="Formal Employment")
    l_i = gamspy.Variable(m, "l_i", type="positive", description="Informal Employment")
    u = gamspy.Variable(m, "u", type="positive", description="Unemployment (Formal)")
    v = gamspy.Variable(m, "v", type="positive", description="Vacancies")
    theta = gamspy.Variable(m, "theta", type="positive", description="Market Tightness")
    w_f = gamspy.Variable(m, "w_f", type="positive", description="Formal Real Wage")
    w_i = gamspy.Variable(m, "w_i", type="positive", description="Informal Real Wage (Imputed)")
    
    # --- Probabilities ---
    p_find = gamspy.Variable(m, "p_find", type="positive", description="Job Finding Prob")
    p_fill = gamspy.Variable(m, "p_fill", type="positive", description="Vacancy Filling Prob")
    
    # --- Households ---
    c = gamspy.Variable(m, "c", domain=[s_hh], type="positive", description="Consumption by Type")
    l_f_hh = gamspy.Variable(m, "l_f_hh", domain=[s_hh], type="positive", description="Formal Labor by Type")
    l_i_hh = gamspy.Variable(m, "l_i_hh", domain=[s_hh], type="positive", description="Informal Labor by Type")
    
    # --- Fiscal ---
    gov_rev = gamspy.Variable(m, "gov_rev", type="positive", description="Government Revenue")
    gov_exp = gamspy.Variable(m, "gov_exp", type="positive", description="Government Expenditure")
    lump_tax = gamspy.Variable(m, "lump_tax", type="free", description="Lump Sum Tax/Transfer (Balancing Item)")

    # =========================================================================
    # 4. EQUATIONS
    # =========================================================================
    
    # --- 1. Production Block ---
    # Formal Production (Cobb-Douglas)
    eq_yf = gamspy.Equation(m, "eq_yf", type="REGULAR")
    eq_yf[...] = y_f == (k**alpha) * (l_f**(1-alpha))
    
    # Informal Production (Linear/Diminishing Returns - simplified as Linear here for stability)
    # Informal sector uses only Labor (Spenders' labor)
    eq_yi = gamspy.Equation(m, "eq_yi", type="REGULAR")
    eq_yi[...] = y_i == p_informal_prod * l_i

    # Total GDP
    eq_ytot = gamspy.Equation(m, "eq_ytot", type="REGULAR")
    eq_ytot[...] = y_tot == y_f + y_i

    # Factor Prices (Formal)
    eq_r = gamspy.Equation(m, "eq_r", type="REGULAR")
    eq_r[...] = r == alpha * (y_f / k)

    mpl_f = (1 - alpha) * (y_f / l_f) # Aux definition for bargaining

    # Factor Prices (Informal - Walrasian Spot Market)
    eq_wi = gamspy.Equation(m, "eq_wi", type="REGULAR")
    eq_wi[...] = w_i == p_informal_prod

    # --- 2. Search & Matching (Formal Sector) ---
    # Matching Technology
    eq_match = gamspy.Equation(m, "eq_match", type="REGULAR")
    eq_match[...] = l_f * p_sep_rate == p_match_eff * (u**p_match_elast) * (v**(1-p_match_elast))
    
    # Definitions
    eq_theta = gamspy.Equation(m, "eq_theta", type="REGULAR")
    eq_theta[...] = theta == v / u
    
    eq_pfill = gamspy.Equation(m, "eq_pfill", type="REGULAR")
    eq_pfill[...] = p_fill == p_match_eff * (theta**(-p_match_elast))
    
    eq_pfind = gamspy.Equation(m, "eq_pfind", type="REGULAR")
    eq_pfind[...] = p_find == p_match_eff * (theta**(1-p_match_elast))

    # Labor Force Identity (Formal Labor Force = 1 normalized)
    # Note: Informal labor is "outside" the formal definition of u in this specific setup
    # to avoid complex flows. Informal is treated as "working while searching" or "secondary".
    eq_lf_ident = gamspy.Equation(m, "eq_lf_ident", type="REGULAR")
    eq_lf_ident[...] = l_f + u == 1

    # Job Creation Condition (Vacancy Posting)
    # Firm posts vacancies until Cost = Expected Profit
    eq_job_creat = gamspy.Equation(m, "eq_job_creat", type="REGULAR")
    eq_job_creat[...] = (p_vac_cost / p_fill) == (mpl_f - w_f) / (1/beta - (1-p_sep_rate))

    # Nash Bargaining Wage (Formal)
    # Surplus split. Note: Taxes drive a wedge here.
    # We use a simplified rule: Weighted avg of Product and Outside Option
    # Outside Option = Informal Wage + Benefits (Opportunity cost)
    outside_option = w_i + p_ben_level
    eq_wage = gamspy.Equation(m, "eq_wage", type="REGULAR")
    eq_wage[...] = w_f == (1-p_bargain)*outside_option + p_bargain*(mpl_f + theta*p_vac_cost)

    # --- 3. Households ---
    
    # Euler Equation (Savers only)
    eq_euler = gamspy.Equation(m, "eq_euler", type="REGULAR")
    eq_euler[...] = 1 == beta * (1 + r - delta)
    
    eq_k_accum = gamspy.Equation(m, "eq_k_accum", type="REGULAR")
    eq_k_accum[...] = i == delta * k

    # Allocation of Aggregate Labor to Households (Proportional)
    eq_lf_hh = gamspy.Equation(m, "eq_lf_hh", domain=[s_hh], type="REGULAR")
    eq_lf_hh[s_hh] = l_f_hh[s_hh] == l_f * p_pop_share[s_hh]

    # Informal Labor Supply
    # Assumption: Savers do NOT work informally (High opportunity cost/Preference)
    eq_li_savers = gamspy.Equation(m, "eq_li_savers", type="REGULAR")
    eq_li_savers[...] = l_i_hh["savers"] == 0
    
    # Spenders supply informal labor until MRS = w_i
    # Utility U = log(C) - (L_total)^(1+phi)/(1+phi)
    # MRS = C * L^phi = w_i
    total_l_spenders = l_f_hh["spenders"] + l_i_hh["spenders"]
    eq_li_spenders = gamspy.Equation(m, "eq_li_spenders", type="REGULAR")
    eq_li_spenders[...] = w_i == c["spenders"] * (total_l_spenders**phi)
    
    # Aggregate Informal Labor
    eq_li_agg = gamspy.Equation(m, "eq_li_agg", type="REGULAR")
    eq_li_agg[...] = l_i == gamspy.Sum(s_hh, l_i_hh[s_hh])

    # Budget Constraint: Spenders (Hand-to-Mouth)
    # Income = Net Formal Wage + Informal Wage + Unemployment Benefits
    # Note: Only employed formal workers pay tax. Unemployed get benefits.
    sp_formal_inc = (1 - tau_w) * w_f * l_f_hh["spenders"]
    sp_informal_inc = w_i * l_i_hh["spenders"]
    # Benefits allocated based on unemployment share (u * pop_share)
    sp_transfers = p_ben_level * (u * p_pop_share["spenders"])
    
    eq_budget_sp = gamspy.Equation(m, "eq_budget_sp", type="REGULAR")
    eq_budget_sp[...] = c["spenders"] == sp_formal_inc + sp_informal_inc + sp_transfers

    # Budget Constraint: Savers
    # Income = Capital Income + Net Formal Wage + Profits + Net Lump Sum - Investment
    # Firm Profits = (Y_f - w_f*l_f) - vac_costs
    firm_profit = y_f - (w_f * l_f) - (v * p_vac_cost)
    sv_formal_inc = (1 - tau_w) * w_f * l_f_hh["savers"]
    
    eq_budget_sv = gamspy.Equation(m, "eq_budget_sv", type="REGULAR")
    eq_budget_sv[...] = c["savers"] == (r * k) + sv_formal_inc + firm_profit - i - lump_tax

    # --- 4. Government Block ---
    # Revenue = Labor Tax on Formal Wages
    eq_gov_rev = gamspy.Equation(m, "eq_gov_rev", type="REGULAR")
    eq_gov_rev[...] = gov_rev == tau_w * w_f * l_f

    # Expenditure = Unemployment Benefits
    eq_gov_exp = gamspy.Equation(m, "eq_gov_exp", type="REGULAR")
    eq_gov_exp[...] = gov_exp == p_ben_level * u
    
    # Government Balance (lump_tax clears the deficit/surplus to Savers)
    # In a full dynamic model, this would be debt. Here, balanced budget rule.
    eq_gov_bal = gamspy.Equation(m, "eq_gov_bal", type="REGULAR")
    eq_gov_bal[...] = gov_rev + lump_tax == gov_exp

# =========================================================================
    # 5. INITIALIZATION & SOLVE
    # =========================================================================
    
    # --- A. Set Non-Zero Initial Guesses ---
    # Critical: Do not leave any divisor or power-base as 0!
    
    # Labor Market (Quantities)
    l_f.setRecords(0.85)
    u.setRecords(0.15)
    l_i.setRecords(0.20)
    
    # Implicitly, if theta=1, then v = u. 
    # We must set v > 0 to avoid division by zero in theta = v/u
    v.setRecords(0.15)  
    theta.setRecords(1.0) 

    # Probabilities (Derived from theta=1 and p_match_eff=0.6)
    p_find.setRecords(0.6)
    p_fill.setRecords(0.6) 

    # Capital & Production
    k.setRecords(10.0)
    y_f.setRecords(2.0)  # Approx guess based on inputs
    y_i.setRecords(0.15)
    y_tot.setRecords(2.15)
    r.setRecords(0.04)   # Standard quarterly return

    # Prices
    w_f.setRecords(1.2)
    w_i.setRecords(0.65) # Matches productivity param
    
    # Consumption (Must be > 0 for log utility or MRS)
    # Spenders consume ~ wages, Savers consume ~ wages + capital income
    c_guess = pd.DataFrame([
        ["savers", 1.5],
        ["spenders", 0.9]
    ], columns=["s_hh", "level"])
    c.setRecords(c_guess)

    # Disaggregated Labor Guesses
    l_f_hh_guess = pd.DataFrame([
        ["savers", 0.425], # Half of 0.85
        ["spenders", 0.425]
    ], columns=["s_hh", "level"])
    l_f_hh.setRecords(l_f_hh_guess)

    l_i_hh_guess = pd.DataFrame([
        ["savers", 0.0],
        ["spenders", 0.20]
    ], columns=["s_hh", "level"])
    l_i_hh.setRecords(l_i_hh_guess)

    # Fiscal
    gov_rev.setRecords(0.3)
    gov_exp.setRecords(0.05)
    lump_tax.setRecords(0.0)

    # --- B. Solve ---
    # Model Definition
    turkey_tank_model = gamspy.Model(
        m,
        "turkey_tank_model",
        problem="CNS", # Constrained Nonlinear System
        equations=m.getEquations(),
    )

    print("\n--- Solving Turkey TANK-SM Model (With Safe Init) ---")
    turkey_tank_model.solve()
    
    # =========================================================================
    # 6. REPORTING
    # =========================================================================
    print("\n" + "="*40)
    print(" TURKEY SOCIAL PROTECTION MODEL RESULTS")
    print("="*40)
    
    # Helper to print
    def pr_var(name, var, desc):
        val = var.records['level'].item() if not var.records.empty else 0.0
        print(f"{desc:.<30} {val:.4f}")

    print("\n--- Macro Aggregates ---")
    pr_var("y_tot", y_tot, "Total GDP")
    pr_var("y_f", y_f, "Formal Output")
    pr_var("y_i", y_i, "Informal Output")
    informal_share = (y_i.records['level'].item() / y_tot.records['level'].item()) * 100
    print(f"Informal Econ Share (%)...... {informal_share:.2f}%")
    
    print("\n--- Labor Market ---")
    pr_var("u", u, "Formal Unemployment Rate")
    pr_var("w_f", w_f, "Formal Real Wage")
    pr_var("w_i", w_i, "Informal Real Wage")
    wage_gap = (w_f.records['level'].item() / w_i.records['level'].item())
    print(f"Formal Wage Premium.......... {wage_gap:.2f}x")

    print("\n--- Inequality (Consumption) ---")
    c_sav = c.records.loc[c.records['s_hh']=='savers','level'].item()
    c_spd = c.records.loc[c.records['s_hh']=='spenders','level'].item()
    print(f"Savers Consumption........... {c_sav:.4f}")
    print(f"Spenders Consumption......... {c_spd:.4f}")
    print(f"Inequality Ratio (Sav/Spd)... {c_sav/c_spd:.2f}")

    print("\n--- Fiscal Status ---")
    pr_var("gov_rev", gov_rev, "Tax Revenue (Labor)")
    pr_var("gov_exp", gov_exp, "Social Transfers")
    
    print("\n--- Policy Analysis Check ---")
    print("If you increase 'p_ben_level' (Benefits):")
    print("1. Spender Consumption (C_spd) should rise (Direct Effect).")
    print("2. Formal Wage (w_f) should rise (Outside Option increases).")
    print("3. Formal Employment (l_f) may fall (Higher labor costs).")
    print("4. Informal Labor (l_i) may rise (Leakage effect).")

# =========================================================================
    # 7. POLICY EXPERIMENT: EXPANDING SOCIAL PROTECTION
    # =========================================================================
    print("\n" + "="*40)
    print(" POLICY EXPERIMENT: INCREASE UNEMPLOYMENT BENEFITS (+33%)")
    print("="*40)
    
    # 1. Store Baseline Results
    base_u = u.records['level'].item()
    base_li = l_i.records['level'].item()
    base_c_spd = c.records.loc[c.records['s_hh']=='spenders','level'].item()
    base_gov_exp = gov_exp.records['level'].item()

    # 2. Apply Shock
    # Increase Benefit Level from 0.30 -> 0.40
    p_ben_level.setRecords(0.40)
    
    # 3. Solve New Equilibrium
    turkey_tank_model.solve()
    
    # 4. Retrieve New Results
    new_u = u.records['level'].item()
    new_li = l_i.records['level'].item()
    new_c_spd = c.records.loc[c.records['s_hh']=='spenders','level'].item()
    new_gov_exp = gov_exp.records['level'].item()

    # 5. Calculate Deltas
    # Did poverty (consumption) improve?
    c_spd_change = ((new_c_spd - base_c_spd) / base_c_spd) * 100
    
    # Did informality worsen?
    li_change = ((new_li - base_li) / base_li) * 100
    
    # Did unemployment worsen?
    u_change = new_u - base_u  # Absolute change in rate
    
    # Fiscal Cost?
    cost_increase = ((new_gov_exp - base_gov_exp) / base_gov_exp) * 100

    # 6. Print Impact Report
    print(f"Policy: Increase Benefit Level (0.3 -> 0.4)")
    print(f"------------------------------------------------")
    print(f"Spender Consumption:  {c_spd_change:+.2f}%  (Did we help the poor?)")
    print(f"Informal Labor:       {li_change:+.2f}%  (Did we push people to informality?)")
    print(f"Unemployment Rate:    {u_change:+.4f}   (Did formal jobs disappear?)")
    print(f"Fiscal Cost:          {cost_increase:+.2f}%  (Budget impact)")
    
    if c_spd_change > 0 and li_change > 0:
        print("\nCONCLUSION: Trade-off Detected.")
        print("The policy helped consumption but increased informality.")
        print("This confirms the 'Leakage' hypothesis for Turkey.")
    elif c_spd_change < 0:
         print("\nCONCLUSION: Policy Failure.")
         print("Job losses outweighed the benefit increase.")

# =========================================================================
    # 8. VISUALIZATION: DASHBOARD (Dual Axis Fix)
    # =========================================================================
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Ensure 'results' folder exists
    output_folder = "results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Data Preparation
    scenarios = ['Baseline', 'Policy Shock (+33% Benefits)']
    u_vals = [base_u * 100, new_u * 100]  # Rate %
    li_vals = [base_li, new_li]           # Level
    
    # Fiscal & Welfare Data
    values = [c_spd_change, cost_increase]
    
    # Create Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Impact of Expanding Social Protection in Turkey (TANK Model)', fontsize=16)

    # --- Plot 1: Labor Market (Dual Axis) ---
    width = 0.35
    x = np.arange(len(scenarios))
    
    # Left Axis: Unemployment (Red)
    color1 = '#d62728' # Red
    rects1 = ax1.bar(x - width/2, u_vals, width, label='Unemployment Rate (%)', color=color1, alpha=0.9)
    ax1.set_ylabel('Unemployment Rate (%)', color=color1, fontsize=12, weight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 15) # Fix scale to make room
    
    # Right Axis: Informal Labor (Orange)
    ax1b = ax1.twinx() # Create secondary axis
    color2 = '#ff7f0e' # Orange
    rects2 = ax1b.bar(x + width/2, li_vals, width, label='Informal Labor (Level)', color=color2, alpha=0.9)
    ax1b.set_ylabel('Informal Labor Level (Normalized)', color=color2, fontsize=12, weight='bold')
    ax1b.tick_params(axis='y', labelcolor=color2)
    ax1b.set_ylim(0, 0.8) # Fix scale to make bars visible
    
    ax1.set_title('Labor Market Structural Changes')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    
    # Legend (Combine handles from both axes)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Label Bars
    def autolabel(rects, ax, fmt='{:.2f}'):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(fmt.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
    
    autolabel(rects1, ax1, '{:.2f}%')
    autolabel(rects2, ax1b, '{:.3f}') # 3 decimal places for small changes

    # --- Plot 2: Cost-Benefit Analysis (% Change) ---
    metrics = ['Spender Consumption\n(Welfare)', 'Gov Expenditure\n(Fiscal Cost)']
    colors = ['green', 'red']
    
    bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
    
    ax2.set_ylabel('Percentage Change (%)')
    ax2.set_title('Policy Efficiency: Cost vs. Benefit')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.2f}%',
                ha='center', va='bottom', fontsize=12, weight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # SAVE
    output_path = os.path.join(output_folder, "turkey_policy_impact_dual_axis.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n[SUCCESS] Visualization saved to: {output_path}")
    plt.show()
            
if __name__ == "__main__":
    main()