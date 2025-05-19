# Study Guide: Multi-Armed Bandit Algorithms for Real-Time Optimization

## Question 23
**When implementing a multi-armed bandit algorithm for real-time optimization, which approach balances exploration and exploitation most effectively?**

### Correct Answer
**`Thompson Sampling` with prior distribution updates**

#### Explanation
Thompson Sampling provides optimal exploration-exploitation balance by:
1. Maintaining probability distributions over possible rewards
2. Sampling from these distributions to select actions
3. Updating beliefs based on observed rewards
4. Naturally adapting to changing environments through Bayesian updates

```python
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

class ThompsonSamplingBandit:
    def __init__(self, num_arms):
        self.alpha = np.ones(num_arms)  # Beta prior parameters
        self.beta = np.ones(num_arms)
        self.rewards = {arm: [] for arm in range(num_arms)}
    
    def select_arm(self):
        # Sample from each arm's posterior
        samples = [beta.rvs(a, b) for a, b in zip(self.alpha, self.beta)]
        return np.argmax(samples)
    
    def update(self, arm, reward):
        # Update Beta distribution parameters
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
        self.rewards[arm].append(reward)

# Example usage
bandit = ThompsonSamplingBandit(num_arms=3)
true_means = [0.3, 0.5, 0.7]  # Unknown to the algorithm

for _ in range(1000):
    arm = bandit.select_arm()
    reward = np.random.binomial(1, true_means[arm])
    bandit.update(arm, reward)

# Visualize posterior distributions
plt.figure(figsize=(10, 6))
for arm in range(3):
    x = np.linspace(0, 1, 100)
    y = beta.pdf(x, bandit.alpha[arm], bandit.beta[arm])
    plt.plot(x, y, label=f'Arm {arm+1} (N={len(bandit.rewards[arm])})')
plt.title('Posterior Distributions After Learning')
plt.legend()
plt.show()
```

### Alternative Options Analysis

#### Option 1: Epsilon-greedy with annealing schedule
**Pros:**
- Simple to implement
- Explicit control over exploration rate
- Annealing reduces exploration over time

**Cons:**
- Requires careful tuning of schedule
- Doesn't use uncertainty information
- Suboptimal exploration (random vs. directed)

```python
class EpsilonGreedyBandit:
    def __init__(self, num_arms, initial_eps=1.0, min_eps=0.01, decay=0.999):
        self.eps = initial_eps
        self.min_eps = min_eps
        self.decay = decay
        self.means = np.zeros(num_arms)
        self.counts = np.zeros(num_arms)
    
    def select_arm(self):
        if np.random.random() < self.eps:
            return np.random.randint(len(self.means))
        return np.argmax(self.means)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.means[arm] += (reward - self.means[arm]) / self.counts[arm]
        self.eps = max(self.min_eps, self.eps * self.decay)

# Usage with different decay schedules
bandit = EpsilonGreedyBandit(num_arms=3, decay=0.995)
```

#### Option 2: `Upper Confidence Bound` (UCB) algorithm
**Pros:**
- Theoretical guarantees on regret
- Explicit uncertainty quantification
- Automatically balances exploration/exploitation

**Cons:**
- Requires known reward bounds
- Can be too optimistic initially
- Less flexible than Bayesian methods

```python
class UCBBandit:
    def __init__(self, num_arms, c=2):
        self.counts = np.zeros(num_arms)
        self.means = np.zeros(num_arms)
        self.c = c  # Exploration parameter
        self.total_counts = 0
    
    def select_arm(self):
        if self.total_counts < len(self.means):
            return self.total_counts
        ucb = self.means + self.c * np.sqrt(np.log(self.total_counts) / self.counts)
        return np.argmax(ucb)
    
    def update(self, arm, reward):
        self.total_counts += 1
        self.counts[arm] += 1
        self.means[arm] += (reward - self.means[arm]) / self.counts[arm]
```

#### Option 3: Contextual bandits with linear payoffs
**Pros:**
- Incorporates additional context information
- Can handle non-stationary environments
- More personalized decisions

**Cons:**
- More complex implementation
- Requires good context features
- Linear payoff assumption may be limiting

```python
from sklearn.linear_model import SGDRegressor

class ContextualBandit:
    def __init__(self, num_arms, context_dim):
        self.models = [SGDRegressor() for _ in range(num_arms)]
        # Initialize models
        for model in self.models:
            model.partial_fit(np.random.randn(1, context_dim), [0])
    
    def select_arm(self, context):
        predictions = [model.predict([context])[0] for model in self.models]
        return np.argmax(predictions)
    
    def update(self, arm, context, reward):
        self.models[arm].partial_fit([context], [reward])
```

### Why the Correct Answer is Best
1. **Bayesian Optimality**: Mathematically proven to minimize regret
2. **Natural Uncertainty Handling**: Explicit probability distributions over rewards
3. **Adaptability**: Automatically adjusts exploration as knowledge improves
4. **Empirical Performance**: Consistently outperforms other methods in A/B tests

### Key Concepts
- **Regret**: Difference between optimal and actual rewards
- **Bayesian Inference**: Updating beliefs with evidence
- **Beta-Bernoulli Model**: Conjugate prior for binary rewards
- **Non-Stationary Environments**: Where reward distributions change over time

### Advanced Considerations
For non-stationary environments, consider:
```python
# Discounted Thompson Sampling for changing rewards
class DiscountedThompsonSampling(ThompsonSamplingBandit):
    def __init__(self, num_arms, discount=0.99):
        super().__init__(num_arms)
        self.discount = discount

    def update(self, arm, reward):
        # Discount old evidence
        self.alpha *= self.discount
        self.beta *= self.discount
        super().update(arm, reward)
```