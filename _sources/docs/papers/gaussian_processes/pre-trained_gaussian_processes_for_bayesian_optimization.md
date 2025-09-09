# Pre-Trained Gaussian Processes for Bayesian Optimization

```{admonition} **Paper information**
:class: note
- Published: 07/2024
- Authors: Zi Wang et. al.
- Link to paper: [Pre-Trained Gaussian Processes for Bayesian Optimization](https://jmlr.org/papers/volume25/23-0269/23-0269.pdf)
- Other links: [Blog](https://research.google/blog/pre-trained-gaussian-processes-for-bayesian-optimization/)
```

Pre-Train a GP to avoid running BO in a super complex scenario where you have a lot of hyperparameters with large domains.

#### The problem
The performance of BO depends on whether the confidence intervals predicted by the surrogate model contain the black-box function. Traditionally, experts use domain knowledge to quantitatively define the mean and kernel parameters (e.g., the range or smoothness of the black-box function) to express their expectations about what the black-box function should look like. However, for many real-world applications like hyperparameter tuning **for deep neural networks**, it is very difficult to understand the landscapes of the tuning objectives. Even for experts with relevant experience, it can be challenging to narrow down appropriate model parameters.

#### The solution
Hyper BayesOpt (HyperBO), a highly customizable interface with an algorithm that removes the need for quantifying model parameters for Gaussian processes in BayesOpt. For new optimization problems, experts can simply select previous tasks that are relevant to the current task they are trying to solve. HyperBO pre-trains a Gaussian process model on data from those selected tasks, and automatically defines the model parameters before running BayesOpt.


During pre-training, we optimize a Gaussian process (GP) such that it can gradually generate functions (illustrated as grey dotted lines) that are similar to the training functions.

**From manual prior quantification -> Abstract identification of training functions**