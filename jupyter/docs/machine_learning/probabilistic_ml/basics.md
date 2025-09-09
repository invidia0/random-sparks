# Probability - Basic concepts

In this spark, we will go through the basics of probability theory concepts. After crusing my head on this topic for a while, I found that everyone is basically giving its own fancy version/notation about everything in this field. In the following I'll try to give you a simple, intuitive and real-world applications-grounded formulation of these concepts from a humble PhD student perspective. A lot of the concepts are drawn from online materials and university courses ([like this one](https://ermongroup.github.io/cs228-notes/preliminaries/probabilityreview/)).

## Concepts of Probability
Throughout this chapter we will consider a **parametric formulation** of probability concepts.

### Independent and identically distributed (IID)
A collection of random variables is IID if each random variable has the same probability distribution as the others and all are mutually independent. It's almost the same as saying "random sampling".
- **Identically distributed** → this means that in the data there are no trends.
- **Independent** → this means that the sample items are all independent and they are not connected to each other in any way.

### Prior probability
A **prior probability distribution** of an uncertain quantity, simply called the **prior**, is the probability distribution of our model's parameters before some evidence is taken into account.

This can be the prior belies that we have about our parameters. Usually, we have some ways-to-go as prior distributions:
- **Gaussian**
- **Uniform**
- **Beta**
- **Bernoulli**

In general the design of a prior distribution is very case-dependent. Sometimes you can hear about "uninformative", flat, or diffuse priors expressing vague or general information about a random variable.

### Posterior probability
In probability theory, the **posterior probability** is a type of **conditional probability** i.e., the probability of an event occurring given that another has occurred. The official formulation is

$$
P(A|B)=\dfrac{P(A\cap B)}{P(B)},
$$

where $P(A \cap B)$ is the probability of $A$ and $B$ happening at the same time. This quantity is usually unknown. $P(B)$ is called the marginal distribution.

In practice, the marginal is almost always intractable to compute in closed form. Typically, what we have instead is a joint distribution $p(x,y)$, describing the probability of pairs $(x,y) \in X \times Y$ falling within certain ranges or discrete values. To obtain the marginal distribution of a variable $X$, we remove the influence of the other variable $Y$—a process known as marginalization.

But we are still missing information about $P(A \cap B)$. At this point, it’s fair to say that in most cases we deal with limited information. The main theorem that allows us to update our knowledge under such conditions is Bayes’ theorem, which is the foundation of Bayesian inference. This will be very useful in the upcoming section on [Bayesian Inference](#bayesian-inference).

#### Bayes' Rule
For events A and B, the Bayes' theorem says

$$
P(A|B) = \dfrac{P(B|A)P(A)}{P(B)},
$$

in Bayesian inference, the event B is fixed in the discussion and we wish to consider the effect of its having been observed on our belief in various possible events A. 
- $P(A|B)$ → the **posterior probability**, our updated belief in $A$ after observing $B$.
- $P(B|A)$ → the **likelihood**, how compatible $B$ is with $A$.
- $P(A)$ → the **prior probability**, our belief in $A$ before seeing $B$.
- $P(B)$ → the **marginal probability**, the overall probability of observing $B$ across all possible $A$.
- 
This theorem has countless applications and underpins many probabilistic methods. For our purposes, we will revisit it intuitively in its parametric form in [Bayesian Inference](#bayesian-inference).

### Likelihood function
**Likelihood** measures how well a statistical model explains observed data by calculating the probability of seeing that data under different parameter values of the model.
- **High likelihood** → the parameters make the observed data very plausible.
- **Low likelihood** → the parameters make the observed data unlikely.

```{admonition} **Hyperparameters vs Parameters**
:class: note, dropdown
This is usually a misunderstood concept in machine learning. Let's clarify this intuitively.\
**Parameters**\
Parameters are learned from data.
- Weights in a neural network.
- Regression coefficients in linear regression.

**Hyperparameters**\
Hyperparameters are **set before** and not learned directly from data.
- Learning rate
- Stopping criterion
- Number of layers in a neural network.

`Hyperparameters govern the model/algorithm setup; parameters are learned from data.`
```

Usually we talk about a **likelihood function**, in fact given a probability density or mass function $f$

$$
x \mapsto f(x | \theta),
$$

where x is from a random variable $X$. Then the likelihood function is defined as

$$
\theta \mapsto f(x|\theta),
$$

usually

$$
\mathcal{L}(\theta | x).
$$

The concept is simple, we can view the function $f$ in two different ways:
- $\theta$ fixed: the function is a probability density function.
- $x$ fixed: the function is a likelihood function.

```{warning}
The likelihood is **NOT** the probability of $\theta$ being the real one (truth), given the realization $X=x$. The likelihood is the probability that a particular outcome $x$ is observed when the true value of the parameter of our model is $\theta$.
```

### Marginalization


---
TBF
---

## Bayesian Inference
**What's the goal?** We want to make predictions when we only have partial information.

### Definitions
- $x$ → a single data point from a random variable $X$.
- $\theta$ → the parameter of the data distribution, i.e. $x \sim p(x \mid \theta)$.
- $\alpha$ → a hyperparameter of the parameter distribution, i.e. $\theta \sim p(\theta \mid \alpha)$.
- $\mathbf{X}$ → the dataset, a collection of $n$ observed data points.
- $\tilde{x}$ → a new, unseen data point we want to predict.

### Bayesian Inference Step 1: The Posterior
The central object we want is the posterior distribution of the parameters
$$
p(\theta\mid\mathbf{X},\alpha).
$$

Using Bayes' rule, we get
$$
\begin{aligned}
p(\theta\mid\mathbf{X},\alpha) &= \dfrac{p(\theta,\mathbf{X},\alpha)}{p(\mathbf{X},\alpha)}, \\
&= \dfrac{p(\mathbf{X}\mid \theta,\alpha)p(\theta,\alpha)}{p(\mathbf{X}\mid\alpha)p(\alpha)}, \\
&= \dfrac{p(\mathbf{X}\mid\theta,\alpha)p(\theta\mid\alpha)\cancel{p(\alpha)}}{p(\mathbf{X}\mid\alpha)\cancel{p(\alpha)}}, \\
&= \dfrac{p(\mathbf{X}\mid\theta,\alpha)p(\theta\mid\alpha)}{p(\mathbf{X}\mid\alpha)}.
\end{aligned}
$$

Here:
- $p(\theta\mid\alpha)$ → the prior distribution of the parameters, i.e., what we believed about the parameters before seeing the data.
- $p(\mathbf{X}\mid\theta,\alpha)$ → the likelihood, i.e., how well a parameter value explains the observed data.
- $p(\mathbf{X}\mid\alpha)$ → the marginal distribution, i.e., the distribution of the observed data marginalized over the parameters.
- $p(\theta\mid\mathbf{X},\alpha)$ → is the posterior distribution, i.e., our updated belief about the parameters after seeing the data.

This is the heart of Bayesian inference: **prior + data → posterior**.

**In practice**: for most realistic models in machine learning, this posterior is intractable (we can’t write it down in closed form). That’s why we need **approximation methods** (MCMC, Variational Inference, etc.).

### Bayesian Inference Step 2: Predictions
The posterior alone only tells us about parameters. What we usually want is to predict new data.

$$
p(\tilde x\mid\mathbf{X},\alpha) = \int p(\tilde x, \mid \theta) p(\theta \mid \mathbf{X},\alpha)\; d\theta.
$$

Here’s what’s happening:
- $p(\tilde{x} \mid \theta)$ → how likely a new point is, given a fixed parameter.
- $p(\theta \mid \mathbf{X}, \alpha)$ → our uncertainty about which parameter is correct.
- The integral is just the expectation of $p(\tilde{x} \mid \theta)$ under the posterior distribution of $\theta$.

Formally,

$$
p(\tilde{x} \mid \mathbf{X}, \alpha) = \mathbb{E}_{\theta \sim p(\theta \mid \mathbf{X}, \alpha)}\big[p(\tilde{x} \mid \theta)\big].
$$

So the posterior predictive distribution says:

> *To predict new data, average the predictions of every possible parameter, weighted by how plausible that parameter is given the data.*


So instead of returning a single prediction, Bayesian inference gives a distribution of possible outcomes, capturing both:
- Randomness in the data.
- Uncertainty in the parameters.

This is why Bayesian inference is powerful, it doesn’t just say “here’s my best guess”, it says “here’s what I think could happen, and how confident I am.”
