# Variational Inference
Variational inference is what we do when math gets too hard. More precisely: it’s a family of methods for approximating integrals that show up in Bayesian inference--integrals that are usually impossible to compute directly.

The goal is to get the **posterior distribution**: the distribution of the hidden stuff we care about ($Z$) after seeing some data ($X$). That’s just Bayes’ theorem:

$$
p(z \mid x) = \frac{p(x \mid z)p(z)}{p(x)}.
$$

Here’s how to think about it:

* $Z$ are the hidden random variables. These could be **the parameters of a neural network** we’re trying to learn, or **latent variables** we want to infer.
* $X$ is the observed data. This could be the **training dataset** if we’re learning parameters, or the **model parameters** if we’re making predictions on new data.

Breaking down Bayes’ theorem:

* $p(z \mid x)$ → posterior (what we want)
* $p(x \mid z)$ → likelihood (how well the hidden stuff explains the data)
* $p(z)$ → prior (our belief before seeing data)
* $p(x)$ → evidence (the troublemaker)


And why is evidence such a problem? Because by definition, it’s the expected likelihood under the prior

$$
p(x) = \mathbb{E}_{z \sim p(z)}[p(x \mid z)] = \int p(x \mid z) \, p(z) \, dz.
$$

That means: *to compute it exactly, we’d have to average over every possible setting of the hidden variables*.

So instead of aiming for the exact posterior, we go for a **simpler approximation** that’s good enough. Variational inference is the toolbox that lets us pick this simpler distribution and nudge it as close as possible to the true posterior.

## Kullback-Leibler (KL) Divergence
So how do we actually measure how “close” our simple guess distribution is to the true posterior? The workhorse here is the KL divergence.

Think of KL divergence as a dissimilarity score between two probability distributions: our tentative approximation $Q$ and the ground truth $P$. It’s also called relative entropy, and it’s defined as

$$
D_{\text{KL}}(P \| Q) = \sum_{x\in\mathcal{X}}P(x) \log \dfrac{P(x)}{Q(x)}.
$$

**Intuition**: the KL divergence of $P$ from $Q$ is the extra “surprise” you’d feel if you used $Q$ to model the world, when in reality the world is following $P$.

:::{admnition} **Example**
:class: tip, dropdown
Imagine the true model of a die is “fair six-sided” ($P$). But you’re using a counterfeit die that rolls six more often ($Q$). Every time you rely on $Q$, you’ll be more “surprised” than necessary when reality follows $P$. That extra surprise is exactly what KL is measuring.

### A numeric dice example

Take a fair six-sided die as the *true* distribution $P$:

$$
P(i)=\tfrac{1}{6}\quad\text{for }i=1,\dots,6.
$$

Now suppose you’re using a **counterfeit** model $Q$ that overestimates 6:

$$
Q(1)=Q(2)=Q(3)=Q(4)=Q(5)=0.1,\quad Q(6)=0.5.
$$

The KL divergence of $P$ from $Q$ (the expected extra surprisal when reality is $P$ but you use $Q$) is

$$
D_{\mathrm{KL}}(P\|Q)=\sum_{i=1}^{6} P(i)\log\frac{P(i)}{Q(i)}.
$$

Plugging numbers in (natural log → [*nats*](https://en.wikipedia.org/wiki/Nat_(unit))):

* For faces $1$–$5$ (each):

  $$
  \tfrac{1}{6}\log\!\left(\frac{1/6}{0.1}\right) \approx 0.0851376
  $$
* For face $6$:

  $$
  \tfrac{1}{6}\log\!\left(\frac{1/6}{0.5}\right) \approx -0.1831020
  $$

Add them up:

$$
D_{\mathrm{KL}}(P\|Q)\approx 0.242586\ \text{nats} \approx 0.3500\ \text{bits}.
$$

Interpretation:

* The positive contributions from faces 1–5 say: when one of those faces occurs, using $Q$ gives you *more* surprisal than the true model would.
* The negative contribution from face 6 means: because $Q$ thinks 6 is much more likely than it really is, when a 6 appears you’re actually *less* surprised under $Q$ than under $P$.
* Overall, on average each roll costs you $\approx 0.2426$ nats (≈0.35 bits) of extra surprisal by using $Q$ instead of $P$. If you roll the die $N$ times, the total extra surprisal grows roughly as $N\times 0.2426$ nats.

A quick extra note: KL is **asymmetric**. If you compute $D_{\mathrm{KL}}(Q\|P)$ (i.e., expectation under $Q$ instead of $P$), you get a different number:

$$
D_{\mathrm{KL}}(Q\|P)\approx 0.293893\ \text{nats}\approx 0.4240\ \text{bits}.
$$

That asymmetry is one reason the direction of KL used in variational inference matters — it changes what kinds of approximations $Q$ will prefer.

:::

Now, in variational inference, we flip things around. We measure:

$$
D_{\text{KL}}(Q\| P ) \triangleq \sum_{\mathbf{Z}}Q(\mathbf{Z}\log\dfrac{Q(\mathbf{Z})}{P(\mathbb{Z}\mid\mathbf{X})})
$$

But wait a second — notice the problem? We still have the posterior hiding inside the formula! That’s the very thing we couldn’t compute in the first place.
This is where the **Evidence Lower Bound (ELBO)** comes in. It’s a clever reformulation that gives us a lower bound on the log evidence and allows us to sidestep the intractable posterior. By maximizing the ELBO, we indirectly minimize the KL divergence between our guess $Q$ and the true posterior $P$.

## Evidence Lower Bound (ELBO)

Start from the KL divergence between our variational distribution $Q(\mathbf{Z})$ and the true posterior $P(\mathbf{Z}\mid \mathbf{X})$

$$
D_{\mathrm{KL}}(Q\|P) \;=\; \mathbb{E}_{\mathbf{Z}\sim Q}\Big[\log\frac{Q(\mathbf{Z})}{P(\mathbf{Z}\mid \mathbf{X})}\Big]
\;=\; \sum_{\mathbf{Z}} Q(\mathbf{Z})\log\frac{Q(\mathbf{Z})}{P(\mathbf{Z}\mid \mathbf{X})}.
$$

Now substitute Bayes’ rule $P(\mathbf{Z}\mid\mathbf{X})=\dfrac{P(\mathbf{X},\mathbf{Z})}{P(\mathbf{X})}$

$$
\begin{aligned}
D_{\mathrm{KL}}(Q\|P)
&= \mathbb{E}_Q\Big[\log Q(\mathbf{Z}) - \log P(\mathbf{Z}\mid\mathbf{X})\Big] \\
&= \mathbb{E}_Q\Big[\log Q(\mathbf{Z}) - \log P(\mathbf{X},\mathbf{Z}) + \log P(\mathbf{X})\Big].
\end{aligned}
$$

Because $\log P(\mathbf{X})$ does not depend on $\mathbf{Z}$, its expectation under $Q$ is itself

$$
D_{\mathrm{KL}}(Q\|P)
= \mathbb{E}_Q\big[\log Q(\mathbf{Z}) - \log P(\mathbf{X},\mathbf{Z})\big] + \log P(\mathbf{X}).
$$

Rearrange to isolate $\log P(\mathbf{X})$

$$
\log P(\mathbf{X}) = D_{\mathrm{KL}}(Q\|P) + \underbrace{\mathbb{E}_Q\big[\log P(\mathbf{X},\mathbf{Z}) - \log Q(\mathbf{Z})\big]}_{\displaystyle \mathcal{L}(Q)}.
$$

Here notice that we changed sign in the expected value. Define the **ELBO** (Evidence Lower Bound) as

$$
\mathcal{L}(Q) \;=\; \mathbb{E}_Q\big[\log P(\mathbf{X},\mathbf{Z})\big] \;-\; \mathbb{E}_Q\big[\log Q(\mathbf{Z})\big].
$$

So the exact identity is

$$
\log P(\mathbf{X}) \;=\; D_{\mathrm{KL}}(Q\|P) \;+\; \mathcal{L}(Q).
$$

### Why is $\mathcal{L}(Q)$ a *lower bound*?

The KL divergence is always non-negative: $D_{\mathrm{KL}}(Q\|P)\ge 0$. Therefore

$$
\log P(\mathbf{X}) \ge \mathcal{L}(Q).
$$

So $\mathcal{L}(Q)$ is literally a lower bound on the log evidence (hence **ELBO**). The gap between the true log-evidence and the ELBO is exactly the KL divergence

$$
\log P(\mathbf{X}) - \mathcal{L}(Q) = D_{\mathrm{KL}}(Q\|P).
$$

Maximizing $\mathcal{L}(Q)$ therefore reduces that gap — i.e., it minimizes the KL divergence between $Q$ and the true posterior.

### Alternate, useful decomposition of the ELBO

Split the joint $\log P(\mathbf{X},\mathbf{Z}) = \log P(\mathbf{X}\mid \mathbf{Z}) + \log P(\mathbf{Z})$. Then

$$
\begin{aligned}
\mathcal{L}(Q)
&= \mathbb{E}_Q\big[\log P(\mathbf{X}\mid\mathbf{Z})\big] + \mathbb{E}_Q\big[\log P(\mathbf{Z})\big] - \mathbb{E}_Q\big[\log Q(\mathbf{Z})\big]\\[6pt]
&= \mathbb{E}_Q\big[\log P(\mathbf{X}\mid\mathbf{Z})\big] \;-\; D_{\mathrm{KL}}\big(Q(\mathbf{Z})\;\big\|\;P(\mathbf{Z})\big).
\end{aligned}
$$

This form is especially intuitive

* $\mathbb{E}_Q[\log P(\mathbf{X}\mid\mathbf{Z})]$ is a *reconstruction* or *data-fit* term (how well samples from $Q$ explain the data).
* $D_{\mathrm{KL}}(Q\|P)$ is a *complexity* term that penalizes $Q$ for straying from the prior $P(\mathbf{Z})$.

So maximizing ELBO trades off fit vs. complexity.

### Why is it called *variational free energy*?

If you flip the sign, define the (variational) **free energy**

$$
\mathcal{F}(Q) \;=\; -\mathcal{L}(Q).
$$

Then

$$
\mathcal{F}(Q) \;=\; \underbrace{\mathbb{E}_Q\big[-\log P(\mathbf{X},\mathbf{Z})\big]}_{\text{expected energy}} \;+\; \underbrace{\mathbb{E}_Q\big[\log Q(\mathbf{Z})\big]}_{-\;H(Q)}.
$$

Writing $U(\mathbf{Z}) = -\log P(\mathbf{X},\mathbf{Z})$ (an energy), $\mathcal{F}$ becomes expected energy minus entropy

$$
\mathcal{F}(Q) = \mathbb{E}_Q[U(\mathbf{Z})] - H(Q).
$$

That is exactly the structure of free energy in statistical physics: energy minus entropy. Minimizing $\mathcal{F}$ (the free energy) is equivalent to maximizing the ELBO, and thus to finding the best variational approximation $Q$.

So people call $\mathcal{L}(Q)$ the ELBO and $-\mathcal{L}(Q)$ the variational free energy (the two names point to the same object with opposite sign conventions).

:::{admonition} The physical analogy
:class: tip, dropdown

In statistical mechanics, the **Helmholtz free energy** is defined as

$$
F = U - TS,
$$

where

* $U$ = internal energy (how much “stuff” the system contains),
* $T$ = temperature,
* $S$ = entropy (how disordered the system is).

It’s called *free* because it measures the energy that is “free to do useful work” once you’ve discounted the part that is *unavailable* due to disorder (entropy).

#### The variational inference version

In VI, the ELBO can be written as

$$
\mathcal{L}(Q) = \underbrace{\mathbb{E}_Q[\log P(\mathbf{X},\mathbf{Z})]}_{\text{negative energy}} + \underbrace{H(Q)}_{\text{entropy of }Q}.
$$

* The first term is like “energy”: it rewards $Q$ for putting weight where the joint $P(\mathbf{X},\mathbf{Z})$ is high.
* The second term is entropy: it rewards $Q$ for being spread out (not collapsing too soon).

Together, this looks just like **energy – entropy** in physics.

#### Why “free”?

It’s “free” in the same sense as in physics:

* The ELBO (or equivalently, the **negative variational free energy**) tells us how much probability mass is *usable* for explaining the data once we account for uncertainty.
* By maximizing ELBO (minimizing free energy), we’re balancing **fit to data** (energy term) and **uncertainty/regularization** (entropy term).

So the “free” part means: *we don’t have to pay for the entropy, it reduces the effective cost of explaining the data*.
:::

## ELBO – Cross-Entropy Connection

Let’s make explicit how the **cross-entropy** and the **Evidence Lower Bound (ELBO)** both arise from the **KL divergence**.

### 1. Start from the KL divergence

For an approximate posterior $q(\theta)$ and the true posterior $p(\theta\mid\mathcal{D})$

$$
D_{\mathrm{KL}}(q(\theta)\,\|\,p(\theta\mid\mathcal{D})) 
= \mathbb{E}_{q(\theta)}\big[\log q(\theta) - \log p(\theta\mid\mathcal{D})\big].
$$

### 2. Introduce cross-entropy

By definition, the cross-entropy between $q$ and $p(\theta\mid\mathcal{D})$ is

$$
H(q, p(\theta\mid\mathcal{D})) = - \mathbb{E}_{q(\theta)}\big[\log p(\theta\mid\mathcal{D})\big].
$$

Thus

$$
D_{\mathrm{KL}}(q \,\|\, p) = H(q, p(\theta\mid\mathcal{D})) - H(q),
$$

where $H(q) = -\mathbb{E}_q[\log q]$ is the entropy of $q$.

**KL = cross-entropy – entropy.**

### 3. Expand with Bayes’ theorem

Using

$$
\log p(\theta\mid\mathcal{D}) = \log p(\mathcal{D}\mid\theta) + \log p(\theta) - \log p(\mathcal{D}),
$$

the cross-entropy becomes

$$
H(q, p(\theta\mid\mathcal{D})) 
= -\mathbb{E}_q[\log p(\mathcal{D}\mid\theta) + \log p(\theta)] + \log p(\mathcal{D}).
$$

### 4. Define the ELBO

The standard decomposition is

$$
\log p(\mathcal{D}) = \text{ELBO}(q) + D_{\mathrm{KL}}(q\,\|\,p(\theta\mid\mathcal{D})).
$$

Rearranging

$$
\text{ELBO}(q) = - H(q, p(\theta\mid\mathcal{D})) + H(q) + \log p(\mathcal{D}).
$$


### Key Takeaway

* **Cross-Entropy** pulls $q$ toward the true posterior.
* **ELBO** does the same, but also rewards $q$ for having higher entropy.
* **KL divergence** ties them together:

$$
\boxed{D_{\mathrm{KL}}(q\|p) = H(q, p) - H(q)}, \quad
\boxed{\text{ELBO} = -H(q,p) + H(q) + \log p(\mathcal{D})}.
$$

### Why Neural Networks Use Cross-Entropy, Not ELBO

In standard deep learning, neural networks are trained by **maximum likelihood estimation (MLE)** rather than full Bayesian inference.

* We assume a **point estimate** of the parameters (weights) instead of a posterior distribution $p(\theta \mid \mathcal{D})$.
* **Thus, there is no approximate distribution $q(\theta)$ to optimize, and no entropy term to keep track of.**
* The training objective reduces to minimizing the **negative log-likelihood** of the data, which in classification problems is exactly the **cross-entropy loss**.

In contrast, when doing **Bayesian deep learning** (e.g. variational inference in Bayesian NNs, VAEs), the ELBO becomes the natural objective because it explicitly balances **data fit (likelihood)** with **regularization (entropy and prior)**.

Summary:

* **Identity:** $\log P(\mathbf{X}) = D_{\mathrm{KL}}(Q\|P) + \mathcal{L}(Q)$.
* **Lower bound:** because $D_{\mathrm{KL}}(Q\|P)\ge0$, we always have $\mathcal{L}(Q)\le\log P(\mathbf{X})$.
* **Optimization:** maximize $\mathcal{L}(Q)$ (ELBO) to make $Q$ as close as possible to the true posterior.
* **Free energy:** $-\mathcal{L}(Q)$ equals expected “energy” minus entropy (so minimizing free energy balances fit and uncertainty).