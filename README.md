# A/B Test Design and Causal Analysis

A clean A/B test on a meaningless metric tells you almost nothing. This project demonstrates the full statistical workflow for a checkout conversion experiment: power analysis before the test, frequentist and Bayesian inference after, CUPED variance reduction throughout, and a business impact calculation at the end.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![SciPy](https://img.shields.io/badge/SciPy-Stats-8CAAE6?style=flat)](https://scipy.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)

## The experiment

One-page checkout vs. multi-step checkout. Conversion rate as primary metric, average order value as secondary.

| Parameter | Value |
|-----------|-------|
| Minimum detectable effect | 0.5 percentage points |
| Required sample size | ~8,600 per group |
| Significance level | 0.05 |
| Statistical power | 80% |
| Test type | Two-tailed z-test + Bayesian Beta-Binomial |

## Results

| Metric | Control | Treatment | Lift | p-value |
|--------|---------|-----------|------|---------|
| Conversion rate | 4.00% | 5.20% | +1.20pp (+30%) | < 0.001 |
| Average order value | ~$58 | ~$61 | +$3 (+5%) | < 0.05 |

Bayesian result: P(Treatment > Control) = ~99%. Both analyses point the same direction.

At 200,000 monthly visitors, that lift translates to a material annualised revenue increase — run the script to see the exact figure.

## Why each step matters

**Pre-test power analysis.** You can't interpret a test result if you didn't size the test correctly beforehand. Too small and real effects go undetected. Too large and you waste time running an experiment that was conclusive weeks earlier. The required sample size calculation here sets a minimum floor before a single visitor is assigned.

**Frequentist z-test.** Two-proportion test with Bonferroni correction for the multiple metric comparison. Reports a p-value and 95% confidence interval on the absolute lift.

**Bayesian Beta-Binomial.** Starts with a flat Beta(1,1) prior and updates it with the observed data. The output — "99% probability that treatment is better" — is the number a business stakeholder can actually act on. The frequentist p-value says something subtly different and most people reading a dashboard misinterpret it.

**CUPED variance reduction.** Using pre-experiment purchase history as a covariate, CUPED adjusts each user's observed outcome to remove variance that has nothing to do with the checkout change. Equivalent statistical power at a smaller sample size — which means faster iteration cycles.

## Running the analysis

```bash
git clone https://github.com/payaljarviya/ab-test-analysis
cd ab-test-analysis
pip install pandas numpy matplotlib seaborn scipy
python ab_test_analysis.py
```

## Output files

| File | Contents |
|------|----------|
| `output/ab_test_results.png` | Conversion rates, lift confidence interval, AOV distribution |
| `output/bayesian_posteriors.png` | Posterior distributions and lift probability |
| `output/power_curve.png` | Sample size vs. MDE at varying power levels |

## Three things to take forward

Ship the one-page checkout. The evidence is strong on both the frequentist and Bayesian sides, and the AOV improvement removes any concern that simplifying the flow was trading conversion for order quality.

Apply CUPED in future experiments. Variance reduction compresses experiment runtime. At the traffic levels in this model, that means weeks off experiment cycles.

Monitor for 2 to 4 weeks post-launch. Novelty effects can inflate short-term conversion lifts. The result should hold — but confirm it before committing to the annualised revenue estimate.

Built with Python using scipy.stats, pandas, numpy, and matplotlib.

---

*Payal Jarviya | MBA Candidate, SKK GSB Seoul | AI and Business Analytics*
