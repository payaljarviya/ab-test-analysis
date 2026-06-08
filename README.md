# A/B Test Design & Causal Analysis

**Business Question:** Does the one-page checkout increase conversion — and if so, by how much, with what confidence, and what is the revenue impact?

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![SciPy](https://img.shields.io/badge/SciPy-Stats-8CAAE6?style=flat)](https://scipy.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)

---

## Experiment Summary

| Parameter | Value |
|-----------|-------|
| Hypothesis | One-page checkout increases conversion rate vs. multi-step |
| Primary metric | Conversion rate |
| Secondary metric | Average Order Value (AOV) |
| MDE (minimum detectable effect) | 0.5 percentage points |
| Required sample size | ~8,600 per group (power analysis) |
| Significance level (α) | 0.05 |
| Statistical power | 80% |
| Test type | Two-tailed z-test + Bayesian Beta-Binomial |

---

## Results

| Metric | Control | Treatment | Lift | p-value | Significant? |
|--------|---------|-----------|------|---------|--------------|
| Conversion Rate | 4.00% | 5.20% | +1.20pp (+30%) | < 0.001 | ✅ Yes |
| AOV (converters) | ~$58 | ~$61 | +$3 (+5%) | < 0.05 | ✅ Yes |

**Bayesian result:** P(Treatment > Control) = ~99% — overwhelming evidence the simplified checkout outperforms.

**Annualised revenue uplift** (200k monthly visitors): ~$**[X]** — see script output for exact figure.

---

## Business Problem

A/B testing without statistical rigour is dangerous in both directions:

**False positives (Type I errors):** Shipping a change that "worked" in the test due to random variation — shipping features that hurt revenue because the test wasn't properly powered.

**False negatives (Type II errors):** Abandoning ideas that genuinely work because the test was underpowered — missing real improvements by running tests that were too small to detect meaningful effects.

This analysis demonstrates the full experiment lifecycle a rigorous product or analytics team should follow.

---

## Methodology

### 1. Pre-Test Power Analysis
Before running the experiment, calculate the minimum sample size needed to detect a meaningful effect at acceptable false-positive and false-negative rates. This prevents stopping tests early or running them too long.

```
n ≈ [(z_α × √(2p̄(1-p̄)) + z_β × √(p₁(1-p₁) + p₂(1-p₂)))²] / MDE²
```

### 2. Frequentist Z-Test (Conversion Rate)
A two-proportion z-test with Bonferroni correction for multiple metrics. Reports p-value, z-statistic, and 95% confidence interval on the absolute lift.

### 3. Bayesian A/B Test (Beta-Binomial)
Uses a Beta(1,1) uninformative prior, updating with observed successes and failures. Outputs:
- P(Treatment > Control) — probability treatment is better
- Posterior distribution of the lift
- 95% credible interval

The Bayesian result is more interpretable for business stakeholders ("99% confident treatment is better") than the frequentist p-value.

### 4. CUPED Variance Reduction
CUPED (Controlled-Experiment Using Pre-Experiment Data) removes variance explained by a pre-experiment covariate (past purchase count), improving test sensitivity without increasing sample size.

```
Y_cuped = Y - θ × (X - E[X])
```

---

## Recommendations

1. **Ship the one-page checkout.** Both frequentist (p < 0.001) and Bayesian (P > 99%) analyses confirm the treatment effect is real and material.

2. **Prioritise CUPED in future experiments.** Variance reduction of ~[X]% means we can run experiments with smaller sample sizes — faster iteration cycles.

3. **Monitor post-launch metrics for 2-4 weeks.** Novelty effect can inflate short-term results — confirm the lift sustains in the post-experiment period.

4. **Run follow-up experiments on AOV.** The +$3 AOV lift is promising — a dedicated experiment on checkout upsell placement could further improve revenue per conversion.

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/payaljarviya/ab-test-analysis
cd ab-test-analysis
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### 3. Run the script
```bash
python ab_test_analysis.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `output/ab_test_results.png` | Conversion rates, lift CI, AOV distribution |
| `output/bayesian_posteriors.png` | Posterior distributions + lift probability |
| `output/power_curve.png` | Sample size vs. MDE power curve |

---

## Skills Demonstrated

- Experiment design (power analysis, MDE, sample size calculation)
- Frequentist hypothesis testing (z-test, t-test, multiple comparison correction)
- Bayesian inference (Beta-Binomial, credible intervals)
- Variance reduction techniques (CUPED)
- Business impact quantification
- Python: `scipy.stats`, `pandas`, `numpy`, `matplotlib`

---

*Author: Payal Jarviya | MBA Candidate, SKK GSB Seoul | AI & Business Analytics + Marketing Analytics*
