"""
The Laffer Curve - Diagnosing Fiscal Limits
===================================================
This script sweeps the labor tax rate (tau_w) from 0% to 70% to trace:
1. Total Government Revenue (The Laffer Curve)
2. Formal Unemployment Rate
3. Informal Sector Size

It helps answer: "What is the maximum revenue the Turkish government 
can theoretically extract from labor taxes before the formal sector collapses?"
"""

import gamspy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # --- 1. MODEL DEFINITION (Recalibrated & Corrected) ---
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
    tau_w = gamspy.Parameter(m, "tau_w", records=0.35) # We will vary this
    p_ben_level = gamspy.Parameter(m, "p_ben_level", records=0.3) # Fixed Benefit Level
    
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
    eq_yf = gamspy.Equation(m, "eq_yf", type="REGULAR"); eq_yf[...] = y_f == (k**alpha) * (l_f**(1-alpha))
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

    # CORRECTED WAGE EQUATION (With Tax Wedge)
    outside_option = w_i + p_ben_level
    eq_wage = gamspy.Equation(m, "eq_wage", type="REGULAR")
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

    model = gamspy.Model(m, "laffer_model", problem="CNS", equations=m.getEquations())

    # =========================================================================
    # 2. RUN LAFFER CURVE SWEEP
    # =========================================================================
    print("--- Running Laffer Curve Sweep (0% to 70% Tax) ---")
    
    # Range of Tax Rates to test
    tax_rates = np.linspace(0.05, 0.70, 25) # 25 points from 5% to 70%
    
    # Storage for results
    results_rev = []
    results_u = []
    results_informal = []
    
    for tax in tax_rates:
        # Update Tax Parameter
        tau_w.setRecords(tax)
        
        try:
            # Solve
            model.solve()
            
            # Store values
            rev = gov_rev.records['level'].item()
            unemp = u.records['level'].item() * 100 # In %
            inf_share = (y_i.records['level'].item() / y_tot.records['level'].item()) * 100
            
            # Sanity Check: If solver "succeeds" but gives garbage (u=100% or rev<0), filter it
            if unemp > 99 or rev < 0:
                rev = np.nan
            
            results_rev.append(rev)
            results_u.append(unemp)
            results_informal.append(inf_share)
            
            print(f"Tax: {tax:.2f} | Rev: {rev:.4f} | U: {unemp:.2f}%")
            
        except:
            print(f"Tax: {tax:.2f} | SOLVE FAILED (Economy collapsed)")
            results_rev.append(np.nan)
            results_u.append(np.nan)
            results_informal.append(np.nan)

    # =========================================================================
    # 3. VISUALIZATION
    # =========================================================================
    if not os.path.exists("results"): os.makedirs("results")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Revenue (Laffer Curve)
    color = 'tab:blue'
    ax1.set_xlabel('Labor Tax Rate (tau_w)')
    ax1.set_ylabel('Gov Revenue (Model Units)', color=color, fontweight='bold')
    ax1.plot(tax_rates, results_rev, color=color, marker='o', linewidth=2, label="Revenue")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight Peak
    # Find max revenue ignoring NaNs
    valid_rev = [x for x in results_rev if not np.isnan(x)]
    if valid_rev:
        max_rev = max(valid_rev)
        max_idx = results_rev.index(max_rev)
        peak_tax = tax_rates[max_idx]
        
        ax1.axvline(peak_tax, color='black', linestyle='--', alpha=0.8)
        ax1.text(peak_tax, max_rev, f' Peak Rev\n @ Tax={peak_tax:.2f}', 
                 ha='center', va='bottom', fontweight='bold')
    
    # Secondary Axis: Unemployment
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('Unemployment Rate (%)', color=color, fontweight='bold')
    ax2.plot(tax_rates, results_u, color=color, linestyle='--', linewidth=2, label="Unemployment")
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('The Turkey Laffer Curve: Fiscal Limits & Unemployment')
    plt.tight_layout()
    plt.savefig("results/turkey_laffer_curve.png", dpi=300)
    print("\n[SUCCESS] Saved Laffer Curve to results/turkey_laffer_curve.png")
    plt.show()

if __name__ == "__main__":
    main()