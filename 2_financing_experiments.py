"""
Policy Experiment - Financing Methods
===============================================================

This script models the impact of different fiscal financing methods
(debt-financed vs. tax-financed) on unemployment and welfare in a
TANK model adapted for Turkey.  
"""

import gamspy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # --- 1. MODEL SETUP ---
    m = gamspy.Container()
    s_hh = gamspy.Set(m, "s_hh", records=["savers", "spenders"])

    # Parameters
    beta = gamspy.Parameter(m, "beta", records=0.985)
    alpha = gamspy.Parameter(m, "alpha", records=0.40)
    delta = gamspy.Parameter(m, "delta", records=0.025)
    p_informal_prod = gamspy.Parameter(m, "p_informal_prod", records=1.2)
    phi = gamspy.Parameter(m, "phi", records=1.0)
    p_sep_rate = gamspy.Parameter(m, "p_sep_rate", records=0.07)
    p_match_eff = gamspy.Parameter(m, "p_match_eff", records=0.35)
    p_vac_cost = gamspy.Parameter(m, "p_vac_cost", records=0.35)
    p_bargain = gamspy.Parameter(m, "p_bargain", records=0.5)
    p_match_elast = gamspy.Parameter(m, "p_match_elast", records=0.5)
    
    # Fiscal Params
    tau_w = gamspy.Parameter(m, "tau_w", records=0.35)
    p_ben_level = gamspy.Parameter(m, "p_ben_level", records=0.3)
    
    pop_data = pd.DataFrame([["savers", 0.5], ["spenders", 0.5]], columns=["s_hh", "value"])
    p_pop_share = gamspy.Parameter(m, "p_pop_share", domain=[s_hh], records=pop_data)

    # Variables
    y_f = gamspy.Variable(m, "y_f", type="positive")
    y_i = gamspy.Variable(m, "y_i", type="positive")
    y_tot = gamspy.Variable(m, "y_tot", type="positive")
    k = gamspy.Variable(m, "k", type="positive")
    i = gamspy.Variable(m, "i", type="positive")
    r = gamspy.Variable(m, "r", type="positive")
    l_f = gamspy.Variable(m, "l_f", type="positive")
    l_i = gamspy.Variable(m, "l_i", type="positive")
    u = gamspy.Variable(m, "u", type="positive")
    v = gamspy.Variable(m, "v", type="positive")
    theta = gamspy.Variable(m, "theta", type="positive")
    w_f = gamspy.Variable(m, "w_f", type="positive")
    w_i = gamspy.Variable(m, "w_i", type="positive")
    p_find = gamspy.Variable(m, "p_find", type="positive")
    p_fill = gamspy.Variable(m, "p_fill", type="positive")
    c = gamspy.Variable(m, "c", domain=[s_hh], type="positive")
    l_f_hh = gamspy.Variable(m, "l_f_hh", domain=[s_hh], type="positive")
    l_i_hh = gamspy.Variable(m, "l_i_hh", domain=[s_hh], type="positive")
    gov_rev = gamspy.Variable(m, "gov_rev", type="positive")
    gov_exp = gamspy.Variable(m, "gov_exp", type="positive")
    lump_tax = gamspy.Variable(m, "lump_tax", type="free")

    # Equations
    eq_yf = gamspy.Equation(m, "eq_yf", type="REGULAR")
    eq_yf[...] = y_f == (k**alpha) * (l_f**(1-alpha))
    eq_yi = gamspy.Equation(m, "eq_yi", type="REGULAR"); eq_yi[...] = y_i == p_informal_prod * l_i
    eq_ytot = gamspy.Equation(m, "eq_ytot", type="REGULAR"); eq_ytot[...] = y_tot == y_f + y_i
    eq_r = gamspy.Equation(m, "eq_r", type="REGULAR"); eq_r[...] = r == alpha * (y_f / k)
    mpl_f = (1 - alpha) * (y_f / l_f)
    eq_wi = gamspy.Equation(m, "eq_wi", type="REGULAR"); eq_wi[...] = w_i == p_informal_prod
    eq_match = gamspy.Equation(m, "eq_match", type="REGULAR"); eq_match[...] = l_f * p_sep_rate == p_match_eff * (u**p_match_elast) * (v**(1-p_match_elast))
    eq_theta = gamspy.Equation(m, "eq_theta", type="REGULAR"); eq_theta[...] = theta == v / u
    eq_pfill = gamspy.Equation(m, "eq_pfill", type="REGULAR"); eq_pfill[...] = p_fill == p_match_eff * (theta**(-p_match_elast))
    eq_pfind = gamspy.Equation(m, "eq_pfind", type="REGULAR"); eq_pfind[...] = p_find == p_match_eff * (theta**(1-p_match_elast))
    eq_lf_ident = gamspy.Equation(m, "eq_lf_ident", type="REGULAR"); eq_lf_ident[...] = l_f + u == 1
    eq_job_creat = gamspy.Equation(m, "eq_job_creat", type="REGULAR"); eq_job_creat[...] = (p_vac_cost / p_fill) == (mpl_f - w_f) / (1/beta - (1-p_sep_rate))

    # *** CRITICAL FIX: WAGE BARGAINING WITH TAX WEDGE ***
    # Workers bargain for "Net Wage" (w_f * (1-tau)). 
    # If tau rises, w_f (cost to firm) must rise to satisfy the worker's outside option.
    outside_option = w_i + p_ben_level
    eq_wage = gamspy.Equation(m, "eq_wage", type="REGULAR")
    
    # Simple Nash derivation approximation with tax on worker:
    # w_f * (1 - tau_w*(1-p_bargain)) = (1-beta)*Outside + beta*(MPL + SaveCost)
    # This ensures that as tau_w -> 1, w_f -> Infinity (Firm pays more).
    eq_wage[...] = w_f * (1 - tau_w * (1 - p_bargain)) == (1 - p_bargain)*outside_option + p_bargain*(mpl_f + theta*p_vac_cost)

    eq_euler = gamspy.Equation(m, "eq_euler", type="REGULAR"); eq_euler[...] = 1 == beta * (1 + r - delta)
    eq_k_accum = gamspy.Equation(m, "eq_k_accum", type="REGULAR"); eq_k_accum[...] = i == delta * k
    eq_lf_hh = gamspy.Equation(m, "eq_lf_hh", domain=[s_hh], type="REGULAR"); eq_lf_hh[s_hh] = l_f_hh[s_hh] == l_f * p_pop_share[s_hh]
    eq_li_savers = gamspy.Equation(m, "eq_li_savers", type="REGULAR"); eq_li_savers[...] = l_i_hh["savers"] == 0
    total_l_spenders = l_f_hh["spenders"] + l_i_hh["spenders"]
    eq_li_spenders = gamspy.Equation(m, "eq_li_spenders", type="REGULAR"); eq_li_spenders[...] = w_i == c["spenders"] * (total_l_spenders**phi)
    eq_li_agg = gamspy.Equation(m, "eq_li_agg", type="REGULAR"); eq_li_agg[...] = l_i == gamspy.Sum(s_hh, l_i_hh[s_hh])
    sp_formal_inc = (1 - tau_w) * w_f * l_f_hh["spenders"]
    sp_informal_inc = w_i * l_i_hh["spenders"]
    sp_transfers = p_ben_level * (u * p_pop_share["spenders"])
    eq_budget_sp = gamspy.Equation(m, "eq_budget_sp", type="REGULAR"); eq_budget_sp[...] = c["spenders"] == sp_formal_inc + sp_informal_inc + sp_transfers
    firm_profit = y_f - (w_f * l_f) - (v * p_vac_cost)
    sv_formal_inc = (1 - tau_w) * w_f * l_f_hh["savers"]
    eq_budget_sv = gamspy.Equation(m, "eq_budget_sv", type="REGULAR"); eq_budget_sv[...] = c["savers"] == (r * k) + sv_formal_inc + firm_profit - i - lump_tax
    eq_gov_rev = gamspy.Equation(m, "eq_gov_rev", type="REGULAR"); eq_gov_rev[...] = gov_rev == tau_w * w_f * l_f
    eq_gov_exp = gamspy.Equation(m, "eq_gov_exp", type="REGULAR"); eq_gov_exp[...] = gov_exp == p_ben_level * u
    eq_gov_bal = gamspy.Equation(m, "eq_gov_bal", type="REGULAR"); eq_gov_bal[...] = gov_rev + lump_tax == gov_exp

    # Initialization
    l_f.setRecords(0.85); u.setRecords(0.15); l_i.setRecords(0.20)
    v.setRecords(0.15); theta.setRecords(1.0); p_find.setRecords(0.6); p_fill.setRecords(0.6)
    k.setRecords(10.0); y_f.setRecords(2.0); r.setRecords(0.04)
    w_f.setRecords(1.2); w_i.setRecords(0.65)
    c.setRecords(pd.DataFrame([["savers", 1.5], ["spenders", 0.9]], columns=["s_hh", "level"]))
    l_f_hh.setRecords(pd.DataFrame([["savers", 0.425], ["spenders", 0.425]], columns=["s_hh", "level"]))
    l_i_hh.setRecords(pd.DataFrame([["savers", 0.0], ["spenders", 0.20]], columns=["s_hh", "level"]))
    gov_rev.setRecords(0.3); gov_exp.setRecords(0.05); lump_tax.setRecords(0.0)

    model = gamspy.Model(m, "turkey_tax_model", problem="CNS", equations=m.getEquations())

    # =========================================================================
    # 2. RUN SCENARIO A: BASELINE
    # =========================================================================
    print("--- Running Scenario A: Baseline ---")
    model.solve()
    base_u = u.records['level'].item()
    base_c_spd = c.records.loc[c.records['s_hh']=='spenders','level'].item()
    base_lump = lump_tax.records['level'].item() # Store Baseline Surplus
    print(f"  Baseline Unemployment: {base_u:.2%}")
    print(f"  Baseline Fiscal Balance (LumpTax): {base_lump:.4f}")

    # =========================================================================
    # 3. RUN SCENARIO B: DEBT FINANCED (LUMP SUM)
    # =========================================================================
    print("\n--- Running Scenario B: Debt/Lump Sum Financing ---")
    p_ben_level.setRecords(0.40) # Increase benefits
    model.solve()
    debt_u = u.records['level'].item()
    debt_c_spd = c.records.loc[c.records['s_hh']=='spenders','level'].item()
    # Note: lump_tax will change here to absorb the cost. We ignore that change for Scenario C target.
    print(f"  Debt Unemployment: {debt_u:.2%}")

    # =========================================================================
    # 4. RUN SCENARIO C: TAX FINANCED (ITERATIVE LOOP)
    # =========================================================================
    print("\n--- Running Scenario C: Tax Financing (Finding Equilibrium) ---")
    print("Goal: Find Tax Rate that returns LumpTax to Baseline Level (Pay for the increment)")
    
    target_tolerance = 0.0001
    learning_rate = 0.2 # Lower rate for stability
    
    # Start loop with current tax (0.35)
    
    print(f"{'ITER':<5} {'TAX RATE':<10} {'DEFICIT GAP':<15} {'UNEMP':<10}")
    
    for i in range(30):
        model.solve()
        
        curr_lump = lump_tax.records['level'].item()
        curr_tau = tau_w.records['value'].item()
        curr_u = u.records['level'].item()
        
        # GAP = Current Surplus - Baseline Surplus
        # We want Gap == 0 (Meaning we restored the fiscal balance)
        # Since lump_tax is a "Revenue Item" in the eq (Rev + Lump = Exp) -> Lump = Exp - Rev
        # If Lump is HIGHER than baseline (less negative), we have a deficit relative to baseline.
        gap = curr_lump - base_lump
        
        print(f"{i:<5} {curr_tau:.4f}     {gap:.6f}        {curr_u:.2%}")
        
        if abs(gap) < target_tolerance:
            print("-> Convergence Reached.")
            break
            
        # Adjust Rule:
        # If Gap > 0 (LumpTax is higher/less negative -> Deficit increased), we need MORE Revenue -> Raise Tax
        new_tau = curr_tau + (learning_rate * gap)
        tau_w.setRecords(new_tau)

    # Store Scenario C
    tax_u = u.records['level'].item()
    tax_c_spd = c.records.loc[c.records['s_hh']=='spenders','level'].item()
    tax_final_rate = tau_w.records['value'].item()

    # =========================================================================
    # 5. VISUALIZATION & OUTPUT
    # =========================================================================
    
    def pchg(new, old): return ((new - old) / old) * 100
    
    scenarios = ['Baseline', 'Debt/Rich Fund', 'Labor Tax Fund']
    u_vals = [base_u*100, debt_u*100, tax_u*100]
    welf_vals = [0, pchg(debt_c_spd, base_c_spd), pchg(tax_c_spd, base_c_spd)]

    if not os.path.exists("results"): os.makedirs("results")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Fiscal Closure Matters: Who Pays for Social Protection?', fontsize=16)

    # Plot 1: Unemployment
    bars1 = ax1.bar(scenarios, u_vals, color=['gray', '#d62728', '#8c0000'], alpha=0.8)
    ax1.set_ylabel('Unemployment Rate (%)')
    ax1.set_title('Unemployment Impact')
    ax1.set_ylim(0, max(u_vals)*1.2)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.2f}%', ha='center', va='bottom', weight='bold')

    # Plot 2: Welfare
    bars2 = ax2.bar(scenarios, welf_vals, color=['gray', 'green', 'red'], alpha=0.8)
    ax2.set_ylabel('% Change in Consumption (Poor)')
    ax2.set_title('Welfare Impact (Poor Households)')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars2:
        h = bar.get_height()
        off = 0.05 if h >= 0 else -0.15
        va = 'bottom' if h >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2., h + off,
                f'{h:+.2f}%', ha='center', va=va, weight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/turkey_tax_experiment.png", dpi=300)
    print("\n[SUCCESS] Chart saved to results/turkey_tax_experiment.png")
    plt.show()

if __name__ == "__main__":
    main()