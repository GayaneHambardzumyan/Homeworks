"""
A/B testing with multi-armed bandits: Epsilon-Greedy vs Thompson Sampling.

Run this file:  python Bandit.py
It runs both experiments (20000 trials each), saves CSVs and plots,
and logs average reward / regret and cumulative reward / regret.
"""
############################### LOGGER
import csv
import sys
from abc import ABC, abstractmethod

import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

BANDIT_REWARDS = [1, 2, 3, 4]   # true mean reward of each advertisement
NUM_TRIALS = 20000              # number of trials per algorithm
TAU = 1.0                       # known precision (1/variance) of the rewards
EPSILON_0 = 1.0                 # epsilon at t=1; decays as EPSILON_0 / t
SEED = 42


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

    ## Extra helpers (allowed: "you can add more stuff") ##
    def _record(self, arm, reward):
        """Store the outcome of one trial.

        :param arm: index of the pulled bandit
        :param reward: reward obtained
        """
        self.arms.append(arm)
        self.rewards.append(reward)
        # expected (pseudo) regret: best mean minus mean of the chosen arm
        self.regrets.append(max(self.p) - self.p[arm])

    def _save_csv(self):
        """Write {Bandit, Reward, Algorithm} rows to '<name>_rewards.csv'."""
        filename = f"{self.name}_rewards.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Bandit", "Reward", "Algorithm"])
            for arm, r in zip(self.arms, self.rewards):
                writer.writerow([self.p[arm], r, self.name])
        logger.info(f"Saved rewards to {filename}")

    def _log_summary(self):
        """Log average / cumulative reward and regret."""
        logger.info(f"[{self.name}] average reward: {np.mean(self.rewards):.4f}")
        logger.info(f"[{self.name}] average regret: {np.mean(self.regrets):.4f}")
        logger.info(f"[{self.name}] cumulative reward: {np.sum(self.rewards):.2f}")
        logger.info(f"[{self.name}] cumulative regret: {np.sum(self.regrets):.2f}")

#--------------------------------------#


class Visualization():

    def plot1(self, algos):
        """Visualize the learning process of each algorithm (linear and log x-axis).

        Plots the running average reward over trials. Saves 'plot1_<name>.png'.

        :param algos: list of finished Bandit experiments
        """
        for algo in algos:
            rewards = np.array(algo.rewards)
            running_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, scale in zip(axes, ["linear", "log"]):
                ax.plot(running_avg, label="running average reward")
                ax.axhline(max(algo.p), color="r", linestyle="--", label="optimal mean")
                ax.set_xscale(scale)
                ax.set_title(f"{algo.name} learning process ({scale} scale)")
                ax.set_xlabel("Trial")
                ax.set_ylabel("Average reward")
                ax.legend()
            fig.tight_layout()
            fig.savefig(f"plot1_{algo.name}.png")
            plt.close(fig)
            logger.info(f"Saved plot1_{algo.name}.png")

    def plot2(self, algos):
        """Compare cumulative rewards and cumulative regrets of the algorithms.

        Saves 'plot2_comparison.png'.

        :param algos: list of finished Bandit experiments
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for algo in algos:
            axes[0].plot(np.cumsum(algo.rewards), label=algo.name)
            axes[1].plot(np.cumsum(algo.regrets), label=algo.name)
        axes[0].set_title("Cumulative reward")
        axes[1].set_title("Cumulative regret")
        for ax in axes:
            ax.set_xlabel("Trial")
            ax.legend()
        fig.tight_layout()
        fig.savefig("plot2_comparison.png")
        plt.close(fig)
        logger.info("Saved plot2_comparison.png")

#--------------------------------------#


class EpsilonGreedy(Bandit):
    """Epsilon-Greedy with epsilon decaying as 1/t."""

    def __init__(self, p, tau=TAU, seed=SEED):
        """Create the experiment.

        :param p: list of true mean rewards of the bandits
        :param tau: known precision of the reward noise
        :param seed: random seed
        """
        self.p = list(p)
        self.k = len(p)
        self.tau = tau
        self.name = "EpsilonGreedy"
        self.rng = np.random.default_rng(seed)
        self.estimates = np.zeros(self.k)
        self.counts = np.zeros(self.k)
        self.arms, self.rewards, self.regrets = [], [], []

    def __repr__(self):
        return f"EpsilonGreedy(p={self.p})"

    def pull(self, arm):
        """Sample a reward from Normal(true mean, 1/tau).

        :param arm: index of the bandit
        :return: the reward
        """
        return self.rng.normal(self.p[arm], 1 / np.sqrt(self.tau))

    def update(self, arm, reward):
        """Incrementally update the sample-mean estimate of the arm.

        :param arm: index of the bandit
        :param reward: observed reward
        """
        self.counts[arm] += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]

    def experiment(self, n_trials=NUM_TRIALS):
        """Run the experiment. Epsilon at trial t is EPSILON_0 / t.

        The first k trials pull each arm once so every estimate is initialised.

        :param n_trials: number of trials
        """
        for t in range(1, n_trials + 1):
            epsilon = EPSILON_0 / t
            if t <= self.k:
                arm = t - 1
            elif self.rng.random() < epsilon:
                arm = int(self.rng.integers(self.k))
            else:
                arm = int(np.argmax(self.estimates))
            reward = self.pull(arm)
            self.update(arm, reward)
            self._record(arm, reward)

    def report(self):
        """Save rewards to CSV and log average/cumulative reward and regret."""
        self._save_csv()
        self._log_summary()

#--------------------------------------#


class ThompsonSampling(Bandit):
    """Thompson Sampling with Gaussian rewards of known precision."""

    def __init__(self, p, tau=TAU, prior_mean=0.0, prior_precision=0.001, seed=SEED):
        """Create the experiment.

        :param p: list of true mean rewards of the bandits
        :param tau: known precision of the reward noise
        :param prior_mean: mean of the Normal prior on each arm's mean
        :param prior_precision: precision of that prior (small = uninformative)
        :param seed: random seed
        """
        self.p = list(p)
        self.k = len(p)
        self.tau = tau
        self.name = "ThompsonSampling"
        self.rng = np.random.default_rng(seed)
        self.post_mean = np.full(self.k, float(prior_mean))
        self.post_precision = np.full(self.k, float(prior_precision))
        self.arms, self.rewards, self.regrets = [], [], []

    def __repr__(self):
        return f"ThompsonSampling(p={self.p})"

    def pull(self, arm):
        """Sample a reward from Normal(true mean, 1/tau).

        :param arm: index of the bandit
        :return: the reward
        """
        return self.rng.normal(self.p[arm], 1 / np.sqrt(self.tau))

    def update(self, arm, reward):
        """Bayesian update of the Normal posterior (known precision tau).

        new_precision = old_precision + tau
        new_mean = (old_precision * old_mean + tau * reward) / new_precision

        :param arm: index of the bandit
        :param reward: observed reward
        """
        old_prec = self.post_precision[arm]
        new_prec = old_prec + self.tau
        self.post_mean[arm] = (old_prec * self.post_mean[arm] + self.tau * reward) / new_prec
        self.post_precision[arm] = new_prec

    def experiment(self, n_trials=NUM_TRIALS):
        """Run the experiment: sample from each posterior, pull the best sample.

        :param n_trials: number of trials
        """
        for _ in range(n_trials):
            samples = self.rng.normal(self.post_mean, 1 / np.sqrt(self.post_precision))
            arm = int(np.argmax(samples))
            reward = self.pull(arm)
            self.update(arm, reward)
            self._record(arm, reward)

    def report(self):
        """Save rewards to CSV and log average/cumulative reward and regret."""
        self._save_csv()
        self._log_summary()


def comparison():
    """Run both algorithms, report, plot, and compare them visually and numerically."""
    eg = EpsilonGreedy(BANDIT_REWARDS)
    ts = ThompsonSampling(BANDIT_REWARDS)
    eg.experiment()
    ts.experiment()
    eg.report()
    ts.report()

    viz = Visualization()
    viz.plot1([eg, ts])
    viz.plot2([eg, ts])

    # combined CSV with all rows: {Bandit, Reward, Algorithm}
    with open("rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Bandit", "Reward", "Algorithm"])
        for algo in (eg, ts):
            for arm, r in zip(algo.arms, algo.rewards):
                writer.writerow([algo.p[arm], r, algo.name])
    logger.info("Saved combined rewards.csv")

    better = ts if sum(ts.rewards) > sum(eg.rewards) else eg
    logger.info(f"Higher cumulative reward: {better.name}")


if __name__ == '__main__':
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
    comparison()
