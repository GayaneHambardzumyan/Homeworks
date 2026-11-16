# Homework 2: A/B Testing with Multi-Armed Bandits

Comparison of **Epsilon-Greedy** (with 1/t decay) and **Thompson Sampling** (Gaussian rewards, known precision) on four advertisement options.

## Setup

- Bandits (true mean rewards): `[1, 2, 3, 4]`
- Number of trials: `20000` per algorithm
- Rewards are drawn from `Normal(mean, 1/tau)` with known precision `tau = 1.0`
- Epsilon-Greedy: `epsilon_t = 1 / t`; the first 4 trials pull each arm once to initialise the estimates
- Thompson Sampling: Normal prior (mean 0, precision 0.001) with a conjugate Normal posterior update using the known precision
- Regret per trial = best mean (4) minus the mean of the chosen arm

## How to run

```
pip install -r requirements.txt
python Bandit.py
```

Running the script prints (via loguru) the average reward, average regret, cumulative reward and cumulative regret of each algorithm, and creates the CSV and plot files below.

## Files

| File | Description |
|---|---|
| `Bandit.py` | Abstract `Bandit` class, `EpsilonGreedy`, `ThompsonSampling`, `Visualization`, `comparison()` |
| `requirements.txt` | Dependencies (loguru, numpy, matplotlib) |
| `rewards.csv` | All rewards from both algorithms: `Bandit, Reward, Algorithm` |
| `EpsilonGreedy_rewards.csv`, `ThompsonSampling_rewards.csv` | Same data split per algorithm |
| `plot1_EpsilonGreedy.png`, `plot1_ThompsonSampling.png` | Learning process (running average reward, linear and log scale) |
| `plot2_comparison.png` | Cumulative reward and cumulative regret, both algorithms |

## Results (seed 42)

- Both algorithms converge to the best bandit (mean 4) within the first few hundred trials, as shown in the `plot1_*.png` files.
- Thompson Sampling: average reward 3.9971, average regret 0.0016, cumulative reward 79941.44, cumulative regret 33.00.
- Epsilon-Greedy finished with a lower cumulative regret than Thompson Sampling in this run (see `plot2_comparison.png`; the full numbers are printed in the run log).
- The cumulative reward curves of the two algorithms almost overlap, so the difference is easier to see in the cumulative regret plot.
- The gap is small relative to 20,000 trials and comes from a single random seed, so it should not be read as one algorithm being better in general.

## Bonus: better implementation plan

1. **Average over many runs.** Repeat each experiment over many seeds (e.g. 100) and plot mean cumulative regret with confidence bands. A single run cannot reliably rank two algorithms whose results are this close.
2. **Better epsilon schedule.** `1/t` decays very quickly, so exploration nearly stops after the first few hundred trials and a wrong early estimate is hard to correct. A slower decay such as `1/sqrt(t)`, or a decay with a small floor, is more robust. Epsilon could also be tuned by grid search.
3. **Learn the precision.** Thompson Sampling currently assumes known precision. A Normal-Gamma prior would estimate the variance of each ad from the data, which is closer to a real A/B test.
4. **Informative priors.** If historical click or revenue data exist, use them to set the prior mean and precision instead of an uninformative prior, which reduces early exploration cost.
5. **Add stronger baselines.** Compare against UCB1 and a non-stationary variant (discounted updates), since real ad performance drifts over time.
6. **Stopping and reporting.** Add a stopping rule (e.g. stop when the posterior probability that one ad is best exceeds 95%) and report the probability of each ad being best, which is more useful for business decisions than a point estimate.
7. **Code quality.** Move the shared logic (reward sampling, logging, CSV export) fully into the base class, add unit tests, and make parameters configurable through the command line.
