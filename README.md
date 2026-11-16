# A/B Test Design and Causal Analysis

**The data in this project is simulated.** No real users were tested. What is being demonstrated is the statistical workflow: power analysis before the test, frequentist and Bayesian inference after, CUPED variance reduction, and an honest reading of what the result does and does not support.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![SciPy](https://img.shields.io/badge/SciPy-Stats-8CAAE6?style=flat)](https://scipy.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)

## The experiment

A checkout optimisation test. One-page checkout as control, multi-step checkout as treatment. Conversion rate is the primary metric.

| Parameter | Value |
|-----------|-------|
| Baseline conversion rate | 4.0% |
| Minimum detectable effect | 0.5 percentage points, absolute |
| Significance level | 0.05, two-tailed |
| Statistical power | 80% |
| **Required sample size** | **25,551 per group** |
| Random seed | 42 |

The sample size is derived from the four parameters above, not chosen. At a 4% baseline, detecting a 0.5pp absolute lift with 80% power needs roughly 25,500 users per arm. A smaller test would only be powered to detect a larger effect: 8,600 per arm, for instance, detects 0.88pp rather than 0.5pp.

## Results

| Metric | Control | Treatment | Absolute lift | 95% CI | p-value |
|--------|---------|-----------|---------------|--------|---------|
| Conversion rate | 4.07% (1,040 / 25,551) | 5.34% (1,365 / 25,551) | +1.27pp | +0.90 to +1.64pp | < 0.001 |

The confidence interval does not contain zero, so the effect is significant at the stated alpha. The Bayesian analysis points the same way.

**The multi-step checkout (treatment) is the winner on this simulated data.**

## CUPED returned zero, and that is the interesting part

CUPED reduces variance by using a pre-experiment covariate to strip out the part of each user's outcome that was predictable before the test began. Less noise means the same statistical power at a smaller sample size.

It only works when the covariate actually predicts the outcome. **On this data the pre-experiment purchase covariate carried no signal, so variance reduction came out at 0%.** Implementing it and reporting that it did nothing is the honest result. A CUPED implementation that always reports a large reduction is usually measuring its own assumptions.

## Why each step matters

**Pre-test power analysis.** You cannot interpret a test result if you did not size the test beforehand. Too small and real effects go undetected. Too large and you waste weeks on an experiment that was already conclusive. Fixing the sample size in advance is also what stops you peeking at the data and stopping when the numbers look good, which is the most common way A/B tests mislead.

**Frequentist z-test.** Two-proportion test with Bonferroni correction for multiple metric comparisons. Reports a p-value and a 95% confidence interval on the absolute lift. The interval is the more useful output, because it describes the plausible range of the effect rather than just rejecting a null.

**Bayesian Beta-Binomial.** Starts from a flat Beta(1,1) prior and updates with the observed data. The output, the probability that treatment beats control, is the phrasing a business stakeholder can act on directly. The frequentist p-value answers a subtly different question that most dashboard readers misinterpret.

**Sample ratio mismatch check.** Before trusting any result, confirm that assignment actually came out near 50/50. A skewed split usually means a bug in the assignment or logging, and it invalidates everything downstream.

## What this does not show

- No revenue projection is claimed. Extrapolating a simulated lift to an annual figure would be inventing a number.
- A single test on simulated data says nothing about whether multi-step checkout works in any real business.
- Novelty effects cannot appear in simulated data. In a real test, a lift this size would need 2 to 4 weeks of post-launch monitoring before being treated as permanent.

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
| `output/ab_test_results.png` | Conversion rates and lift confidence interval |
| `output/bayesian_posteriors.png` | Posterior distributions and lift probability |
| `output/power_curve.png` | Sample size vs MDE at varying power levels |

Built with Python using scipy.stats, pandas, numpy, and matplotlib.

---

*Payal Jarviya | MBA Candidate, SKK GSB Seoul | AI and Business Analytics*
