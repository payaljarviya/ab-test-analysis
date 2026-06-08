"""
A/B Test Design & Causal Analysis
===================================
Author: Payal Jarviya
MBA Candidate | SKK GSB Seoul | AI & Business Analytics

Business Problem:
    Running A/B tests without statistical rigour leads to either false positives
    (shipping changes that don't work) or false negatives (abandoning ideas that
    do work). This project demonstrates end-to-end experiment design and analysis:
    pre-test power analysis, sample size calculation, frequentist significance
    testing, Bayesian A/B testing, and a CUPED variance-reduction technique —
    all applied to a simulated e-commerce checkout optimisation experiment.

    Scenario: We are testing a simplified one-page checkout (Treatment) vs. the
    existing multi-step checkout (Control). Primary metric: conversion rate.
    Secondary metrics: average order value (AOV) and session duration.

Dataset: Synthetic experiment data (generated internally)

Instructions:
    pip install pandas numpy matplotlib seaborn scipy
    python ab_test_analysis.py
    Charts saved to output/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings, os, math
warnings.filterwarnings('ignore')

# ─── PALETTE ───────────────────────────────────────────────────────────────────
NAVY   = "#1a3a5c"
BLUE   = "#2E75B6"
GREEN  = "#27AE60"
RED    = "#C0392B"
ORANGE = "#E67E22"
GRAY   = "#95a5a6"
LIGHT  = "#D5E8F0"
BG     = "#f8f9fa"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': BG,
    'figure.facecolor': 'white',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
})

os.makedirs('output', exist_ok=True)


# ─── 1. POWER ANALYSIS ────────────────────────────────────────────────────────
def power_analysis(baseline_cr=0.04, mde=0.005, alpha=0.05, power=0.80):
    """
    Calculate the minimum sample size needed per group.
    baseline_cr : current conversion rate (e.g. 4%)
    mde         : minimum detectable effect (absolute, e.g. 0.5pp lift)
    alpha       : significance level (Type I error)
    power       : desired statistical power (1 - Type II error)
    """
    p1 = baseline_cr
    p2 = baseline_cr + mde
    p_bar = (p1 + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)  # two-tailed
    z_beta  = stats.norm.ppf(power)

    n = ((z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) +
          z_beta  * math.sqrt(p1*(1-p1) + p2*(1-p2))) ** 2) / (mde ** 2)

    print(f"\n{'='*60}")
    print(f"  POWER ANALYSIS")
    print(f"{'='*60}")
    print(f"  Baseline conversion rate : {p1:.1%}")
    print(f"  MDE (absolute)           : +{mde:.1%}  →  target {p2:.1%}")
    print(f"  Significance level (α)   : {alpha}")
    print(f"  Power (1-β)              : {power:.0%}")
    print(f"  Required n per group     : {math.ceil(n):,}")
    print(f"  Total sample needed      : {math.ceil(n)*2:,}")

    return math.ceil(n)


# ─── 2. GENERATE EXPERIMENT DATA ──────────────────────────────────────────────
def generate_experiment(n_per_group=5000, seed=42):
    np.random.seed(seed)

    # True treatment effect: +1.2pp conversion lift, +$3 AOV lift
    cr_control   = 0.040
    cr_treatment = 0.052
    aov_control   = 58.0
    aov_treatment = 61.0

    # Control group
    ctrl = pd.DataFrame({
        'user_id'  : [f'U{i:06d}' for i in range(n_per_group)],
        'group'    : 'Control',
        'converted': np.random.binomial(1, cr_control, n_per_group),
        'session_duration_s': np.random.lognormal(5.2, 0.6, n_per_group).round(0),
        'pre_exp_purchases'  : np.random.poisson(2.1, n_per_group),
    })
    ctrl['order_value'] = np.where(
        ctrl['converted'] == 1,
        np.random.lognormal(np.log(aov_control), 0.4, n_per_group).round(2),
        0.0
    )

    # Treatment group
    trt = pd.DataFrame({
        'user_id'  : [f'U{n_per_group+i:06d}' for i in range(n_per_group)],
        'group'    : 'Treatment',
        'converted': np.random.binomial(1, cr_treatment, n_per_group),
        'session_duration_s': np.random.lognormal(4.9, 0.5, n_per_group).round(0),
        'pre_exp_purchases'  : np.random.poisson(2.1, n_per_group),
    })
    trt['order_value'] = np.where(
        trt['converted'] == 1,
        np.random.lognormal(np.log(aov_treatment), 0.4, n_per_group).round(2),
        0.0
    )

    df = pd.concat([ctrl, trt], ignore_index=True)
    print(f"\n{'─'*50}")
    print(f"  Experiment data generated: {len(df):,} users ({n_per_group:,} per group)")
    print(f"  Control CR:   {ctrl['converted'].mean():.2%}")
    print(f"  Treatment CR: {trt['converted'].mean():.2%}")

    return df


# ─── 3. FREQUENTIST Z-TEST ────────────────────────────────────────────────────
def frequentist_test(df, metric='converted', alpha=0.05):
    ctrl = df[df['group'] == 'Control'][metric]
    trt  = df[df['group'] == 'Treatment'][metric]

    if metric == 'converted':
        # Proportion z-test
        n_c, n_t   = len(ctrl), len(trt)
        p_c, p_t   = ctrl.mean(), trt.mean()
        p_pool     = (ctrl.sum() + trt.sum()) / (n_c + n_t)
        se         = math.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))
        z_stat     = (p_t - p_c) / se
        p_val      = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        lift_abs   = p_t - p_c
        lift_rel   = (p_t - p_c) / p_c
        ci_lo = (p_t - p_c) - 1.96 * se
        ci_hi = (p_t - p_c) + 1.96 * se
        label = "Conversion Rate"
    else:
        # Two-sample t-test for continuous metrics (AOV among converters)
        ctrl_conv = ctrl[df[df['group']=='Control']['converted'] == 1]
        trt_conv  = trt[df[df['group']=='Treatment']['converted'] == 1]
        t_stat, p_val = stats.ttest_ind(ctrl_conv, trt_conv)
        z_stat = t_stat
        p_c, p_t = ctrl_conv.mean(), trt_conv.mean()
        lift_abs = p_t - p_c
        lift_rel = lift_abs / p_c
        se = stats.sem(trt_conv - ctrl_conv.mean()) if len(trt_conv) > 1 else 0
        ci_lo = lift_abs - 1.96 * se
        ci_hi = lift_abs + 1.96 * se
        label = "Average Order Value (converters)"

    significant = p_val < alpha

    print(f"\n  Frequentist Test — {label}")
    print(f"  {'─'*48}")
    print(f"  Control:     {p_c:.4f}")
    print(f"  Treatment:   {p_t:.4f}")
    print(f"  Lift:        {lift_abs:+.4f}  ({lift_rel:+.1%})")
    print(f"  95% CI:      [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Z-statistic: {z_stat:.3f}")
    print(f"  p-value:     {p_val:.4f}   {'✅ SIGNIFICANT' if significant else '❌ NOT SIGNIFICANT'} (α={alpha})")

    return {'metric': label, 'ctrl': p_c, 'trt': p_t, 'lift': lift_abs,
            'lift_rel': lift_rel, 'z': z_stat, 'p_val': p_val,
            'ci_lo': ci_lo, 'ci_hi': ci_hi, 'significant': significant}


# ─── 4. BAYESIAN A/B TEST ─────────────────────────────────────────────────────
def bayesian_ab_test(df, n_samples=50000, seed=42):
    """
    Beta-Binomial Bayesian A/B test.
    Prior: Beta(1, 1) — uninformative / uniform.
    Posterior: Beta(alpha + successes, beta + failures).
    """
    np.random.seed(seed)
    ctrl = df[df['group'] == 'Control']['converted']
    trt  = df[df['group'] == 'Treatment']['converted']

    alpha_c = 1 + ctrl.sum();  beta_c = 1 + (len(ctrl) - ctrl.sum())
    alpha_t = 1 + trt.sum();   beta_t = 1 + (len(trt)  - trt.sum())

    samples_c = np.random.beta(alpha_c, beta_c, n_samples)
    samples_t = np.random.beta(alpha_t, beta_t, n_samples)

    prob_t_better = (samples_t > samples_c).mean()
    expected_lift = (samples_t - samples_c).mean()
    credible_lo   = np.percentile(samples_t - samples_c, 2.5)
    credible_hi   = np.percentile(samples_t - samples_c, 97.5)

    print(f"\n  Bayesian Test — Conversion Rate")
    print(f"  {'─'*48}")
    print(f"  P(Treatment > Control)  : {prob_t_better:.1%}")
    print(f"  Expected lift (mean)    : {expected_lift:+.4f}")
    print(f"  95% Credible Interval   : [{credible_lo:+.4f}, {credible_hi:+.4f}]")

    return samples_c, samples_t, prob_t_better, expected_lift


# ─── 5. CUPED VARIANCE REDUCTION ──────────────────────────────────────────────
def cuped_adjustment(df, metric='converted', covariate='pre_exp_purchases'):
    """
    CUPED (Controlled-Experiment Using Pre-Experiment Data) reduces variance by
    removing variation explained by a pre-experiment covariate.
    """
    theta = (np.cov(df[metric], df[covariate], ddof=1)[0, 1] /
             np.var(df[covariate], ddof=1))

    grand_mean_cov = df[covariate].mean()
    df = df.copy()
    df['metric_cuped'] = df[metric] - theta * (df[covariate] - grand_mean_cov)

    var_original = df[metric].var()
    var_cuped    = df['metric_cuped'].var()
    variance_reduction = 1 - (var_cuped / var_original)

    ctrl = df[df['group'] == 'Control']['metric_cuped']
    trt  = df[df['group'] == 'Treatment']['metric_cuped']
    t_stat, p_val = stats.ttest_ind(trt, ctrl)

    print(f"\n  CUPED Variance Reduction")
    print(f"  {'─'*48}")
    print(f"  Theta (covariate coeff) : {theta:.4f}")
    print(f"  Variance reduction      : {variance_reduction:.1%}")
    print(f"  CUPED p-value           : {p_val:.4f}")

    return variance_reduction, p_val


# ─── 6. VISUALISATIONS ─────────────────────────────────────────────────────────
def plot_results_overview(df, freq_result):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('A/B Test Results — One-Page Checkout Optimisation', fontsize=14, fontweight='bold')

    # Conversion rates
    ax = axes[0]
    groups = ['Control', 'Treatment']
    rates  = [freq_result['ctrl'], freq_result['trt']]
    colors = [BLUE, GREEN]
    bars = ax.bar(groups, [r * 100 for r in rates], color=colors, width=0.5, edgecolor='white')
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{rate:.2%}", ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Conversion Rate (%)')
    ax.set_title(f'Conversion Rate\n{"✅ Significant" if freq_result["significant"] else "❌ Not Significant"} '
                 f'(p={freq_result["p_val"]:.4f})')
    ax.set_ylim(0, max(rates) * 1.3)

    # Lift CI
    ax = axes[1]
    lift = freq_result['lift']
    ci_lo, ci_hi = freq_result['ci_lo'], freq_result['ci_hi']
    ax.barh([0], [lift * 100], xerr=[[(lift - ci_lo) * 100], [(ci_hi - lift) * 100]],
            color=GREEN if lift > 0 else RED, height=0.3,
            error_kw={'capsize': 8, 'capthick': 2, 'elinewidth': 2, 'ecolor': NAVY})
    ax.axvline(0, color=NAVY, linewidth=1.5, linestyle='--')
    ax.set_xlabel('Absolute Lift (pp)')
    ax.set_title(f'Treatment Lift\n+{lift:.2%} ({lift/freq_result["ctrl"]:+.1%} relative)')
    ax.set_yticks([])

    # AOV comparison (converters only)
    ax = axes[2]
    for grp, color, label in [('Control', BLUE, 'Control'), ('Treatment', GREEN, 'Treatment')]:
        vals = df[(df['group'] == grp) & (df['converted'] == 1)]['order_value']
        ax.hist(vals, bins=30, color=color, alpha=0.6, label=f"{label} (${vals.mean():.0f} avg)",
                edgecolor='white')
    ax.set_xlabel('Order Value ($)')
    ax.set_ylabel('Converters')
    ax.set_title('AOV Distribution (Converters)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('output/ab_test_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: output/ab_test_results.png")


def plot_bayesian_posteriors(samples_c, samples_t, prob_t_better):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Bayesian A/B Analysis — Posterior Distributions', fontsize=14, fontweight='bold')

    # Posterior distributions
    ax = axes[0]
    ax.hist(samples_c * 100, bins=60, color=BLUE, alpha=0.6, density=True, label='Control')
    ax.hist(samples_t * 100, bins=60, color=GREEN, alpha=0.6, density=True, label='Treatment')
    ax.set_xlabel('Conversion Rate (%)')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Distributions\n(Beta-Binomial)')
    ax.legend()

    # Distribution of lift (treatment - control)
    ax = axes[1]
    lift_samples = (samples_t - samples_c) * 100
    ax.hist(lift_samples, bins=60, color=NAVY, alpha=0.8, density=True, edgecolor='white')
    ax.axvline(0, color=RED, linewidth=2, linestyle='--', label='No effect')
    ax.axvline(lift_samples.mean(), color=GREEN, linewidth=2, label=f'Mean lift: +{lift_samples.mean():.2f}pp')
    ci_lo, ci_hi = np.percentile(lift_samples, [2.5, 97.5])
    ax.axvspan(ci_lo, ci_hi, alpha=0.15, color=GREEN, label=f'95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]pp')
    ax.set_xlabel('Treatment − Control Lift (pp)')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of Lift\nP(Treatment > Control) = {prob_t_better:.1%}')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('output/bayesian_posteriors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: output/bayesian_posteriors.png")


def plot_power_curve(baseline_cr=0.040, alpha=0.05):
    """Show required sample size vs. MDE."""
    mdes = np.linspace(0.001, 0.020, 50)
    sample_sizes = []
    for mde in mdes:
        p1, p2 = baseline_cr, baseline_cr + mde
        p_bar = (p1 + p2) / 2
        z_a = stats.norm.ppf(1 - alpha/2)
        z_b = stats.norm.ppf(0.80)
        n = ((z_a * math.sqrt(2 * p_bar * (1-p_bar)) +
              z_b * math.sqrt(p1*(1-p1) + p2*(1-p2)))**2) / mde**2
        sample_sizes.append(math.ceil(n))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([m * 100 for m in mdes], [s * 2 for s in sample_sizes],
            color=BLUE, linewidth=2.5)
    ax.fill_between([m * 100 for m in mdes], [s * 2 for s in sample_sizes],
                    alpha=0.15, color=BLUE)
    ax.axvline(0.5, color=RED, linewidth=2, linestyle='--', label='MDE = 0.5pp (this experiment)')
    ax.set_xlabel('Minimum Detectable Effect (pp)', fontsize=11)
    ax.set_ylabel('Total Sample Size Required', fontsize=11)
    ax.set_title(f'Power Curve — Sample Size vs. MDE\n(α={alpha}, Power=80%, Baseline CR={baseline_cr:.0%})')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend()

    plt.tight_layout()
    plt.savefig('output/power_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: output/power_curve.png")


# ─── 7. BUSINESS SUMMARY ───────────────────────────────────────────────────────
def print_summary(df, freq_result, prob_t_better):
    ctrl = df[df['group'] == 'Control']
    trt  = df[df['group'] == 'Treatment']
    aov_ctrl = ctrl[ctrl['converted'] == 1]['order_value'].mean()
    aov_trt  = trt[trt['converted']  == 1]['order_value'].mean()
    monthly_visitors = 200_000
    incremental_conversions = freq_result['lift'] * monthly_visitors
    incremental_revenue = incremental_conversions * aov_trt

    print(f"\n{'='*60}")
    print(f"  A/B TEST BUSINESS SUMMARY")
    print(f"{'='*60}")
    print(f"  Test:          One-Page vs. Multi-Step Checkout")
    print(f"  Sample size:   {len(df):,}  ({len(ctrl):,} control | {len(trt):,} treatment)")
    print(f"\n  Primary Metric — Conversion Rate")
    print(f"    Control:     {freq_result['ctrl']:.2%}")
    print(f"    Treatment:   {freq_result['trt']:.2%}")
    print(f"    Lift:        {freq_result['lift']:+.2%}  ({freq_result['lift_rel']:+.1%} relative)")
    print(f"    p-value:     {freq_result['p_val']:.4f}  — {'✅ SIGNIFICANT' if freq_result['significant'] else '❌ NOT SIGNIFICANT'}")
    print(f"\n  Bayesian Confidence: P(Treatment better) = {prob_t_better:.1%}")
    print(f"\n  Secondary Metric — AOV (converters only)")
    print(f"    Control: ${aov_ctrl:.2f}  |  Treatment: ${aov_trt:.2f}")
    print(f"\n  Business Impact (assuming {monthly_visitors:,} monthly visitors):")
    print(f"    Incremental monthly conversions: ~{incremental_conversions:,.0f}")
    print(f"    Incremental monthly revenue:     ~${incremental_revenue:,.0f}")
    print(f"    Annualised revenue upside:        ~${incremental_revenue*12:,.0f}")
    print(f"\n  Recommendation: {'SHIP the one-page checkout.' if freq_result['significant'] else 'Do not ship — continue testing.'}")
    print(f"{'='*60}\n")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n  A/B Test Design & Causal Analysis")
    print("  Author: Payal Jarviya | SKK GSB Seoul\n")

    n_req = power_analysis(baseline_cr=0.04, mde=0.005, alpha=0.05, power=0.80)
    df = generate_experiment(n_per_group=max(n_req, 5000))

    print(f"\n{'='*60}")
    print(f"  HYPOTHESIS TESTS")
    print(f"{'='*60}")
    cr_result  = frequentist_test(df, metric='converted')
    aov_result = frequentist_test(df, metric='order_value')
    cuped_var_red, cuped_p = cuped_adjustment(df)

    samples_c, samples_t, prob_t_better, expected_lift = bayesian_ab_test(df)

    print("\n  Generating charts...")
    plot_results_overview(df, cr_result)
    plot_bayesian_posteriors(samples_c, samples_t, prob_t_better)
    plot_power_curve()

    print_summary(df, cr_result, prob_t_better)
    print("  Done.\n")
