# Markov Chain Monte Carlo - Hands on

**Markov chain Monte Carlo (MCMC)** is a **class** of algorithms used to **draw samples from a probability distribution**. In statistics and Bayesian inference, usually we face the problem of evaluating some intractable integrals, like the marginal likelihood or the posterior predictive distribution.

What we can do is leverage methods to **approximate** these integrals. MCMC is one of the most popular methods to do so.

## Monte Carlo Estimate
Monte Carlo methods are a class of **computational algorithms** that rely on **repeated random sampling** to obtain numerical results. The basic idea is to use randomness to solve problems that might be deterministic in principle.

Monte Carlo methods are mainly used in three problem classes: **optimization**, **numerical integration**, and **generating draws from a probability distribution**.

### Simple Monte Carlo
Let's suppose we want to compute the expected value of a population $\mu$, but the formula is intractable. We can use Monte Carlo methods to estimate it

$$
\mu = \mathbb{E}[X] = \int x p(x) dx \approx \frac{1}{N} \sum_{i=1}^{N} x_i.
$$

Where $x_i$ are samples drawn from the distribution $p(x)$. Numerically, the algorithm is as follows:

```
s = 0;
for i in range(N):
    x_i = sample_from_distribution(p)
    s += x_i
mu_hat = s / N
```

## Markov Chains
A **Markov chain** is a stochastic process describing a sequence of possible events in which the probability of each event depends only on the state attained in the previous event. This is called the **Markov property**.

This is the same as saying that the future is independent of the past, given the present. Let's consider an example with a system of 4 states, 'Rain' or 'Car Wash' causing the 'Wet Ground' followed by the 'Slip'. Jumping from one state to the next state depends only on the current state not on the sequence of previous states which lead to this state.

```mermaid
  flowchart TD;
      A-->B;
      A-->C;
      B-->D;
      C-->D;
```
Mathematically, it is expressed as

$$
P(X_{n+1} = n \mid X_n = k_n, X_{n-1}=k_{n-1}, \dots, X_1 = k_1) = P(X_{n+1} = k \mid X_n = k_n).
$$

### Markov Chain Monte Carlo (MCMC)
In Markov Chain Monte Carlo we construct leverage the Markov property. That is, starting from an initial state, we move to a new state guided by **transition rules** that depend only on the current state. The goal is to construct a Markov chain whose stationary distribution is the target distribution we want to sample from.
The transition rules are designed to ensure that the Markov chain spends more time in regions of high probability under the target distribution.






So let's switch to an example where now we have to compute the posterior distribution of our NN parameters $p(\theta \mid D)$ given the data $D$. The posterior is proportional to the likelihood times the prior

$$
p(\theta \mid D) \propto p(D \mid \theta) p(\theta).
$$

To get the exact posterior we need to compute the marginal likelihood $p(D)$ which is intractable

$$
p(D) = \int p(D \mid \theta) p(\theta) d\theta.
$$

We can use MCMC to sample from the posterior distribution $p(\theta \mid D)$ and then use these samples to approximate the posterior predictive distribution

$$
p(y^* \mid x^*, D) = \int p(y^* \mid x^*, \theta) p(\theta \mid D) d\theta \approx \frac{1}{N} \sum_{i=1}^{N} p(y^* \mid x^*, \theta_i)
$$

where $\theta_i$ are samples drawn from the posterior distribution $p(\theta \mid D)$. See, here we have a double integral, one for the marginal likelihood and one for the posterior predictive distribution. MCMC helps us to avoid computing the marginal likelihood. Instead, we can directly sample from the posterior distribution and use these samples to approximate the posterior predictive distribution, by computing only the likelihood $p(y^* \mid x^*, \theta)$ that is usually tractable.

In this case the MCMC algorithm is as follows:
- Start from an initial value $\theta_0$.
- Generate a new candidate value $\theta'$ from a proposal distribution $q(\theta' \mid \theta)$.
- Compute the acceptance ratio, that is the ratio between the posterior of the candidate value and the posterior of the current value.

## Hamiltonian Monte Carlo

## Langevin Dynamics
https://friedmanroy.github.io/blog/2022/Langevin/#mjx-eqn%3Aeq%3Alangevin

## MCMC vs VI

