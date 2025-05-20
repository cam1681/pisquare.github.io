---
layout: post
title: "Deep Dive: Understanding Diffusion Models from a Mathematical View"
date: 2024-03-19
author: "Pipi Hu"
---

*"What I cannot create, I do not understand." - Richard Feynman*

Diffusion models have revolutionized the field of generative AI, producing stunning results in image generation, audio synthesis, and more. But what makes them tick? In this comprehensive guide, we'll break down the mathematical foundations of diffusion models, making complex concepts accessible while maintaining technical accuracy.

## Table of Contents
1. [Introduction to Sample Generation](#introduction-to-sample-generation)
2. [The Score Function: A Key Ingredient](#the-score-function)
3. [Training Approaches: ESM, ISM, and DSM](#training-approaches)
4. [Relations Between Different Models](#relations-between-models)
5. [Diffusion Models: From Theory to Practice](#diffusion-models)
6. [Continuous Formulations and SDEs](#continuous-formulations)
7. [Practical Implementation](#practical-implementation)
8. [Open Questions and Future Directions](#open-questions)

## Introduction to Sample Generation

At its core, generative modeling is about learning to sample from complex probability distributions. Let's start with the basics and gradually build up to more sophisticated methods.

### Simple Distributions
We begin with simple distributions that serve as building blocks for more complex models. The uniform distribution provides the foundation for random number generation. For a uniform distribution $U(a,b)$, we can generate samples using:
$$x = a + (b-a) \cdot \text{rand()}$$
where $\text{rand()}$ generates a random number in $[0,1]$. This simple mechanism forms the basis for sampling from more complex distributions.

Moving to the Gaussian distribution, we can use the elegant Box-Muller transform to generate samples from a standard normal distribution $N(0,1)$:
$$x = \sqrt{-2\ln(u_1)} \cos(2\pi u_2)$$
$$y = \sqrt{-2\ln(u_1)} \sin(2\pi u_2)$$
where $u_1, u_2 \sim U(0,1)$ are independent uniform random variables. This transform converts uniform random variables into normally distributed ones through a clever use of polar coordinates.

For mixture models, we combine these basic building blocks. A mixture of $K$ Gaussians with weights $\pi_k$ has the form:
$$p(x) = \sum_{k=1}^K \pi_k N(x; \mu_k, \Sigma_k)$$
Sampling from this distribution involves two steps: first choosing which Gaussian to sample from using a categorical distribution, then sampling from the chosen Gaussian using the Box-Muller transform. This hierarchical sampling process will become important when we discuss more complex models.

### Complex Distributions
When dealing with complex distributions, we need more sophisticated methods. Let's explore two powerful approaches that form the foundation of modern generative modeling.

The first approach, Langevin Markov Chain Monte Carlo (Langevin MCMC), uses gradient information to guide the sampling process. The discrete-time update rule is:
$$x_{t+1} = x_t + \frac{\epsilon^2}{2}\nabla_x \log p(x_t) + \epsilon z_t$$
where $z_t \sim N(0,I)$ and $\epsilon$ is the step size. As we take smaller steps (as $\epsilon \to 0$), this process converges to the continuous-time Langevin dynamics:
$$dx_t = \frac{1}{2}\nabla_x \log p(x_t) dt + dB_t$$
This continuous formulation will be particularly relevant when we discuss diffusion models later.

The second approach, Stein Variational Gradient Descent (SVGD), takes a different perspective by using kernel methods. It updates particles using:
$$x_i \leftarrow x_i + \epsilon \phi^*(x_i)$$
where $\phi^*(x) = \mathbb{E}_{y \sim q} [k(x,y)\nabla_y \log p(y) + \nabla_y k(x,y)]$ and $k(x,y)$ is a positive definite kernel (typically an RBF kernel). This method combines the advantages of particle-based methods, kernel methods, and gradient-based optimization, making it particularly effective for high-dimensional distributions.

### The Score Function: A Key Ingredient

Now that we understand the basic sampling methods, let's introduce a crucial concept that will be central to our discussion of diffusion models: the score function. Traditional sample generation often involves three key steps:

1. Modeling the probability function $p(x)$ as $p(x;\theta)$
2. Dealing with normalization constants: $p(x;\theta)=\frac{1}{C(\theta)}e^{-E(x;\theta)}$, where $C(\theta)\equiv\int_x e^{-E(x;\theta)}dx$ is often intractable
3. Using sampling methods like Langevin MCMC

The Langevin MCMC update rule we saw earlier:
$$x_t = x_{t-1}+\frac{\Delta t}{2}\nabla_x\log p(x_{t-1};\theta)+\sqrt{\Delta t}\epsilon, \quad \epsilon\sim N(0, 1)$$

reveals something crucial: the term $\nabla_x\log p(x_{t-1};\theta)$ (the **score function**) is free of the intractable normalization constant $C(\theta)$. This is because:
$$\nabla_x\log p(x;\theta) = \nabla_x (\log(e^{-E(x;\theta)}) - \log C(\theta)) = -\nabla_xE(x;\theta)$$

We define the **score function** as:
$$S(x;\theta) \equiv \nabla_x \log p(x;\theta)$$

This function will be our guide through the rest of the discussion, as it plays a central role in both the theory and practice of diffusion models.

### More about Langevin MCMC and the Score Function

Let's dive deeper into the relationship between Langevin dynamics and the score function. The continuous form of Langevin MCMC (as $\Delta t\to 0$) is:
$$dx = \frac{1}{2}\nabla_x\log p(x) dt+ dB_t$$
where $dB_t$ is a Brownian motion. This equation describes how a particle moves through the probability landscape, being pulled by the score function while being jostled by random noise.

If we express our probability distribution as $p(x)=\frac{1}{C}e^{-E}$, we get:
$$dx = -\frac{1}{2}\nabla_x E dt + dB_t$$

This is known as overdamped Langevin dynamics. The full Langevin dynamics (from molecular dynamics) is more complex:
$$\ddot{x} = -\nabla_x E - \gamma \dot{x} + dB_t$$

When the friction term $\gamma \dot{x}$ dominates inertia $\ddot{x}$ (i.e., $\gamma \dot{x} \gg \ddot{x}$), we arrive at the overdamped Langevin equation:
$$dx = -\frac{1}{\gamma}\nabla_x E dt + \frac{1}{\gamma}dB_t$$

This simplified form will be particularly useful when we discuss the continuous-time formulation of diffusion models.

### Properties of the Score Function

The score function has several important properties that make it particularly useful for generative modeling. Let's explore these properties in detail:

1. **Zero Mean**: The score function has zero mean under the data distribution:
   $$\mathbb{E}_{p(x)}[\nabla_x \log p(x)] = \int p(x) \nabla_x \log p(x) dx = \int \nabla_x p(x) dx = 0$$
   This property ensures that the score function provides unbiased gradient information.

2. **Fisher Information**: The covariance of the score function is the Fisher information matrix:
   $$\mathbb{E}_{p(x)}[\nabla_x \log p(x) \nabla_x \log p(x)^T] = I(\theta)$$
   This connection to Fisher information provides a natural way to measure the quality of our score estimates.

3. **Stein's Identity**: For any function $f(x)$ with sufficient regularity:
   $$\mathbb{E}_{p(x)}[f(x)\nabla_x \log p(x) + \nabla_x f(x)] = 0$$
   This identity will be crucial when we discuss different training approaches.

4. **Score Matching Property**: The score function is invariant to the normalization constant:
   $$\nabla_x \log p(x) = \nabla_x \log \frac{p(x)}{Z} = \nabla_x \log p(x) - \nabla_x \log Z = \nabla_x \log p(x)$$
   This property is particularly important as it allows us to work with unnormalized probability distributions.

These properties form the theoretical foundation for the training approaches we'll discuss next.

## Training Approaches

Now that we understand the score function and its properties, let's explore different ways to train models to estimate it. We'll discuss three main approaches, each with its own advantages and challenges.

### 1. Explicit Score Matching (ESM)
The most direct approach is to minimize the difference between our model's score and the true score:
$$J_{ESM} =\frac{1}{2} \mathbb{E}_{p(x)}\left[\|s_\theta(x) - \nabla_x \log p(x)\|^2\right]$$

This approach has several important properties:
- It provides an unbiased estimator of the true score
- It requires access to the true score function
- It can be computationally expensive for complex distributions

The main challenge with ESM is that it requires knowledge of $\nabla_x \log p(x)$, which is often unavailable in practice. This limitation led to the development of alternative approaches.

### 2. Implicit Score Matching (ISM)
To address the limitations of ESM, Hyvärinen (2005) proposed a tractable alternative:
$$J_{ISM} \equiv \mathbb{E}_{p(x)}\left[\frac{1}{2}\|s_\theta(x)\|^2+\text{Tr}(\nabla_x s_\theta(x))\right]$$

Let's prove that this is equivalent to ESM up to a constant. Starting from $J_{ESM}$:
$$
\begin{aligned}
J_{ESM} &= \frac{1}{2} \mathbb{E}_{p(x)}\left[\|s_\theta(x)-\nabla_x \log p(x)\|^2\right] \\
        &= \frac{1}{2} \mathbb{E}_{p(x)}\left[\|s_\theta(x)\|^2\right] - \mathbb{E}_{p(x)}[s_\theta(x) \cdot \nabla_x \log p(x)] + \frac{1}{2}\mathbb{E}_{p(x)}\left[\|\nabla_x \log p(x)\|^2\right]
\end{aligned}
$$

The cross term can be rewritten using integration by parts:
$$
\begin{aligned}
- \mathbb{E}_{p(x)}[s_\theta(x) \cdot \nabla_x \log p(x)] &= -\int_x p(x) s_\theta(x) \cdot \frac{\nabla_x p(x)}{p(x)} dx \\
&= -\int_x s_\theta(x) \cdot \nabla_x p(x) dx \\
&= -\left[ \int_x \nabla_x \cdot (p(x)s_\theta(x)) dx - \int_x p(x) (\nabla_x \cdot s_\theta(x)) dx \right] \\
&= \mathbb{E}_{p(x)}[\nabla_x \cdot s_\theta(x)]
\end{aligned}
$$

Thus, $J_{ESM} = J_{ISM} - C$, where $C = \frac{1}{2}\mathbb{E}_{p(x)}\left[\|\nabla_x \log p(x)\|^2\right]$ is a constant.

ISM has several advantages:
- It doesn't require the true score function
- It can be computed from samples
- It's equivalent to ESM up to a constant
- It provides a more stable training process

### 3. Denoising Score Matching (DSM)
Building on the insights from ISM, Vincent (2011) proposed an even more stable approach:
$$J_{DSM} = \frac{1}{2}\mathbb{E}_{p(x)p(\tilde{x}|x)}\left[\|s_\theta(\tilde{x}) - \nabla_{\tilde{x}}\log p(\tilde{x}|x)\|^2\right]$$

This approach adds noise to the data and matches the score of the noisy distribution. Let's prove its equivalence to ESM:
$$
\begin{aligned}
  J_{DSM} &= \frac{1}{2}\mathbb{E}_{p(\tilde{x})}\left[\|s_\theta(\tilde{x})\|^2\right] - \mathbb{E}_{p(x,\tilde{x})}[s_\theta(\tilde{x})\cdot\nabla_{\tilde{x}}\log p(\tilde{x}|x)] \\
  &+ \frac{1}{2}\mathbb{E}_{p(x,\tilde{x})}[\|\nabla_{\tilde{x}}\log p(\tilde{x}|x)\|^2]
\end{aligned}
$$

The cross term $\mathbb{E}_{p(x,\tilde{x})}[s_\theta(\tilde{x})\cdot\nabla_{\tilde{x}}\log p(\tilde{x}|x)]$ can be shown to be equal to $\mathbb{E}_{p(\tilde{x})}[s_\theta(\tilde{x})\cdot\nabla_{\tilde{x}}\log p(\tilde{x})]$ under expectation over $p(\tilde{x}|x)$.

DSM offers several advantages:
- It's more stable than ISM
- It works well with high-dimensional data
- It can be used with any noise distribution
- It's particularly effective for image generation

These three approaches provide different ways to train score-based models, each with its own strengths and applications. In practice, the choice between them depends on the specific requirements of your task.

## Relations Between Different Models

Now that we understand different training approaches, let's explore how they relate to each other and to other modeling approaches. This will help us understand the broader landscape of generative modeling.

### Tweedie's Formula
A key result from Bayesian statistics, Tweedie's Formula, provides an elegant connection between the score function and posterior expectations. Given random variables $X, Y$, where $Y \sim N(X, \sigma^2 I)$ (i.e., $p(y|x) = N(y; x, \sigma^2 I)$), the posterior expectation of $X$ is:
$$\mathbb{E}[X|y] = y + \sigma^2 \nabla_y \log p(y)$$

Let's prove this important result. For $p(y|x) = N(y; x, \sigma^2 I)$, we have:
$$\nabla_y \log p(y|x) = -\frac{y-x}{\sigma^2}$$

The posterior expectation is:
$$\mathbb{E}[X|y] = \int x p(x|y) dx = \int x \frac{p(y|x)p(x)}{p(y)} dx$$

Taking the gradient with respect to $y$:
$$\nabla_y \log p(y) = \frac{\nabla_y p(y)}{p(y)} = \frac{\int \nabla_y p(y|x) p(x) dx}{p(y)} = \frac{\int (x-y) p(y|x) p(x) dx}{\sigma^2 p(y)} = \frac{\mathbb{E}[X|y] - y}{\sigma^2}$$

Therefore:
$$\mathbb{E}[X|y] = y + \sigma^2 \nabla_y \log p(y)$$

This formula has profound implications for our understanding of diffusion models. Applying it to our noising process:
$$\mathbb{E}[\alpha x_0 | \tilde{x}] = \tilde{x} + \sigma^2 \nabla_{\tilde{x}} \log p(\tilde{x})$$

So, the score of the marginal distribution $p(\tilde{x})$ is:
$$s(\tilde{x}) \equiv \nabla_{\tilde{x}} \log p(\tilde{x}) = \frac{\mathbb{E}[\alpha x_0 | \tilde{x}] - \tilde{x}}{\sigma^2}$$

We can also write this in terms of the expected noise $\mathbb{E}[\epsilon|\tilde{x}]$:
$$s(\tilde{x}) = -\frac{\mathbb{E}[\epsilon|\tilde{x}]}{\sigma}$$

This connection between the score function and posterior expectations will be crucial when we discuss different model parametrizations.

### Alternative Derivation of the Score
Let's explore another way to understand the score function. Starting with $\tilde{x} = \alpha x_0 + \sigma \epsilon$, so $p(\tilde{x}|x_0) = C e^{-\frac{\|\tilde{x}-\alpha x_0\|^2}{2\sigma^2}}$.

The score of the marginal $p(\tilde{x})$ is:
$$s(\tilde{x}) = \nabla_{\tilde{x}}\log p(\tilde{x}) = \frac{\nabla_{\tilde{x}}p(\tilde{x})}{p(\tilde{x})} = \frac{\int_{x_0} \nabla_{\tilde{x}}p(\tilde{x}|x_0) p(x_0)dx_0}{p(\tilde{x})}$$

Since $\nabla_{\tilde{x}}p(\tilde{x}|x_0) = p(\tilde{x}|x_0) \nabla_{\tilde{x}}\log p(\tilde{x}|x_0)$:
$$s(\tilde{x}) = \mathbb{E}_{x_0 \sim p(x_0|\tilde{x})}[\nabla_{\tilde{x}} \log p(\tilde{x}|x_0)]$$

Given $p(\tilde{x}|x_0) = N(\tilde{x}; \alpha x_0, \sigma^2 I)$, $\nabla_{\tilde{x}}\log p(\tilde{x}|x_0) = -\frac{\tilde{x}-\alpha x_0}{\sigma^2}$.

$$s(\tilde{x}) = \mathbb{E}_{x_0}\left[-\frac{\tilde{x}-\alpha x_0}{\sigma^2} \middle| \tilde{x}\right] = -\frac{\tilde{x}-\alpha \mathbb{E}[x_0|\tilde{x}]}{\sigma^2}$$

This derivation shows how the score function naturally emerges from the noise-corrupted data distribution.

### Relating Different Model Parametrizations
We can train neural networks to predict different quantities, each with its own advantages:

1. **Score Model $s_\theta(\tilde{x})$**:
   - Trained to predict $s(\tilde{x}) = \nabla_{\tilde{x}}\log p(\tilde{x})$
   - Using DSM, the target is $\nabla_{\tilde{x}}\log p(\tilde{x}|x_0) = -\frac{\tilde{x}-\alpha x_0}{\sigma^2} = -\frac{\epsilon}{\sigma}$
   - Loss: $Loss_{score} = \mathbb{E}\left[\left\|s_\theta(\tilde{x}) + \frac{\epsilon}{\sigma}\right\|^2\right]$
   - Advantages:
     - Direct modeling of the score function
     - Clear connection to Langevin dynamics
     - Theoretical guarantees

2. **$x_0$-Prediction Model $x_\theta(\tilde{x})$ (Denoising Model)**:
   - Trained to predict $x_0$
   - Optimal $x_{\theta^*}(\tilde{x}) = \mathbb{E}[x_0|\tilde{x}]$
   - Loss: $Loss_{x_0} = \mathbb{E}[\|x_\theta(\tilde{x}) - x_0\|^2]$
   - Connection to score: $s_\theta(\tilde{x}) \approx -\frac{\tilde{x}-\alpha x_\theta(\tilde{x})}{\sigma^2}$
   - Advantages:
     - More intuitive objective
     - Often more stable training
     - Better for high-dimensional data

3. **$\epsilon$-Prediction Model $\epsilon_\theta(\tilde{x})$**:
   - Trained to predict the noise $\epsilon$
   - Optimal $\epsilon_{\theta^*}(\tilde{x}) = \mathbb{E}[\epsilon|\tilde{x}]$
   - Loss: $Loss_{\epsilon} = \mathbb{E}[\|\epsilon_\theta(\tilde{x}) - \epsilon\|^2]$
   - Connection to score: $s_\theta(\tilde{x}) \approx -\frac{\epsilon_\theta(\tilde{x})}{\sigma}$
   - Advantages:
     - Simpler training objective
     - Often more stable
     - Better for high-dimensional data

These relationships show the consistency between modeling the score, the original data, or the noise. In practice, these parametrizations can be interchanged:
$$s_\theta(\tilde{x}) = -\frac{\tilde{x}-\alpha x_\theta(\tilde{x})}{\sigma^2} = -\frac{\epsilon_\theta(\tilde{x})}{\sigma}$$

This flexibility in model parametrization is one of the key strengths of diffusion models.

## Diffusion Models

Now that we understand the theoretical foundations, let's explore the two main types of diffusion models in detail.

### 1. Score Matching with Langevin Dynamics (SMLD)
Proposed by Song & Ermon (2019), SMLD combines score matching with annealed Langevin dynamics:

**Forward Process**:
- Gradually adds noise to data
- Uses a sequence of noise levels: $\sigma_1 > \sigma_2 > \dots > \sigma_T \approx 0$
- Transition kernel: $p(\tilde{x}_i| x) = N(\tilde{x}_i; x, \sigma_i^2)$
- The noise schedule can be geometric: $\sigma_i = \sigma_1 \cdot \gamma^{i-1}$ for some $\gamma \in (0,1)$

**Training**:
$$J_{DSM} = \frac{1}{T}\sum_{i=1}^T \mathbb{E}_{p(x_0),\epsilon}\left[\lambda(\sigma_i)\|s_\theta(x_0+\sigma_i\epsilon, \sigma_i) + \frac{\epsilon}{\sigma_i}\|^2\right]$$

The weighting function $\lambda(\sigma_i)$ can be chosen to:
- Balance different noise levels
- Improve training stability
- Optimize sample quality

**Sampling**:
Annealed Langevin MCMC:
$$x_k^{(i)} = x_{k-1}^{(i)} + \frac{\text{step_size}_i}{2} s_\theta(x_{k-1}^{(i)}, \sigma_i) + \sqrt{\text{step_size}_i} z_k, \quad z_k \sim N(0,I)$$

The step size $\text{step_size}_i$ should be:
- Small enough for stability
- Large enough for efficient sampling
- Typically chosen as $\text{step_size}_i \propto \sigma_i^2$

### 2. Denoising Diffusion Probabilistic Models (DDPM)
Proposed by Ho et al. (2020), DDPM takes a different approach to the forward process:

**Forward Process**:
$$x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon_{t-1}$$
where $\beta_t \in (0,1)$ is a noise schedule.

The noise schedule can be:
- Linear: $\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)$
- Cosine: $\beta_t = \frac{\beta_1}{2}(1 + \cos(\frac{t-1}{T-1}\pi))$
- Quadratic: $\beta_t = \beta_1 + \frac{(t-1)^2}{(T-1)^2}(\beta_T - \beta_1)$

**Direct Sampling**:
$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$
where $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$.

**Training Loss**:
$$\mathbb{E}_{t\sim U[0,1]}\mathbb{E}_{p(x)}\mathbb{E}_{p(x_t|x)}\left[\|\epsilon_\theta(x_t,t)-\epsilon\|^2\right]$$

The loss can be weighted by:
- Uniform weighting
- Signal-to-noise ratio: $\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$
- Custom weighting for better sample quality

**Sampling**:
$$x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}(x_t+\beta_t s_\theta(x_t, t)) +\sqrt{\beta_t}\epsilon_t, \quad t = T, T-1, \cdots, 1$$

The sampling process can be improved by:
- Using multiple denoising steps
- Adding noise during sampling
- Using a predictor-corrector scheme

## Continuous Formulations

The discrete processes we've discussed can be generalized to continuous-time Stochastic Differential Equations (SDEs), providing a more elegant theoretical framework.

### General Form
The general form of the SDE is:
$$dx_t = f(x_t, t)dt + g(t) dB_t$$

The probability density $p(x,t)$ evolves according to the Fokker-Planck equation:
$$\frac{\partial p(x,t)}{\partial t} + \nabla_x \cdot \left(f(x,t) p(x,t)\right) - \frac{1}{2}g^2(t) \Delta_x p(x,t)=0$$

### Two Main Types:
1. **Variance Exploding (VE) SDE**:
   $$dx_t = \sqrt{\frac{d[\sigma^2(t)]}{dt}} dB_t$$
   Transition kernel: $p(x_t|x_0) = N(x_t; x_0, \sigma^2(t)I)$
   - The variance grows without bound
   - Often used with $\sigma^2(t) = t$
   - Simpler to implement

2. **Variance Preserving (VP) SDE**:
   $$dx_t = -\frac{1}{2}\tilde{\beta}(t) x_t dt + \sqrt{\tilde{\beta}(t)} dB_t$$
   Transition kernel: $p(x_t|x_0) = N(x_t; x_0 e^{-\frac{1}{2}\int_0^t \tilde{\beta}(s)ds}, (1-e^{-\int_0^t \tilde{\beta}(s)ds})I)$
   - The variance is bounded
   - Often used with $\tilde{\beta}(t) = \beta_0 + t(\beta_1 - \beta_0)$
   - More stable training

### Reverse Process
The reverse of the forward SDE is:
$$dx_t = [f(x_t, t) - g^2(t) \nabla_x \log p(x_t, t)]dt + g(t) d\bar{B}_t$$

During sampling, we replace $\nabla_x \log p(x_t, t)$ with our learned score model $s_\theta(x_t, t)$.

The reverse process can be solved using:
1. Euler-Maruyama method
2. Runge-Kutta methods
3. Probability flow ODE
4. Predictor-corrector methods

## Practical Implementation

Now that we understand the theory, let's discuss how to implement these models in practice.

### Training Process
The training process involves several steps:
1. Sample data point $x \sim p_{data}$
2. Sample time $t \sim U[0, 1]$
3. Sample noise $\epsilon \sim N(0, 1)$
4. Add noise: $x_t = \tilde{\alpha}_t x + \tilde{\sigma}_t\epsilon$
5. Compute loss: $\|s_\theta(x_t, t) - \epsilon/\tilde{\sigma}_t\|$

**Training Tips**:
- Use gradient clipping to prevent exploding gradients
- Normalize the data to improve training stability
- Use learning rate scheduling to adapt to different training phases
- Monitor the loss carefully to detect potential issues
- Use mixed precision training for better efficiency
- Implement proper logging for debugging and monitoring

### Inference Process
The inference process involves:
1. Generate initial noise $x_T \sim N(0, \sigma^2_{max})$
2. Solve reverse SDE:
   $$dx_t = [f(x_t, t) - g^2(t) s_\theta(x_t, t)]dt + g(t) d\bar{B}_t$$
3. Apply corrector steps using Langevin MCMC:
   $$x_{k+1} = x_k + \frac{\delta}{2}s_\theta(x_k, t) + \sqrt{\delta}z_k, \quad z_k \sim N(0,I)$$

**Inference Tips**:
- Use multiple sampling steps for better quality
- Implement early stopping to save computation
- Use temperature scaling to control sample diversity
- Apply proper noise scheduling for better results
- Monitor sample quality during generation
- Use proper random seeds for reproducibility

## Open Questions

The field of diffusion models is still evolving rapidly, with many interesting questions remaining open for research:

1. **Reverse Process Equivalence**
   - Can we prove that different reverse sampling forms lead to the same marginal distribution?
   - What are the trade-offs between different formulations?
   - How do different noise schedules affect the results?
   - Can we derive optimal noise schedules?

2. **Score Function Evolution**
   - How does the score function evolve over time?
   - Can we derive a PDE for the score function?
   - The score function $s(x,t)$ should satisfy:
     $$\frac{\partial s}{\partial t} = \nabla_x (-\nabla_x \cdot f - \langle f, s\rangle + \frac{1}{2}g^2\|s\|^2 + \frac{1}{2}g^2\langle \nabla, s \rangle)$$
   - Can we use this PDE to improve training?

3. **Training Objectives**
   - Why is ISM loss less commonly used than DSM loss?
   - Are there better training objectives?
   - Can we derive optimal weighting functions?
   - How do different loss functions affect sample quality?

4. **Architecture Design**
   - What are the best architectures for score models?
   - How do different normalization layers affect training?
   - Can we design more efficient architectures?
   - How do different activation functions affect performance?

5. **Sampling Efficiency**
   - Can we reduce the number of sampling steps?
   - Are there better sampling algorithms?
   - Can we parallelize the sampling process?
   - How do different sampling methods affect quality?

## Conclusion

Diffusion models represent a powerful approach to generative modeling, with deep mathematical foundations in probability theory and stochastic processes. Understanding these foundations is crucial for both theoretical insights and practical implementations.

The field continues to evolve rapidly, with new architectures and training methods being developed regularly. By understanding the mathematical principles behind these models, we can better appreciate their capabilities and limitations.

Key takeaways:
1. Score-based models provide a powerful framework for generative modeling
2. Different training objectives (ESM, ISM, DSM) are equivalent up to constants
3. Continuous-time formulations provide elegant theoretical insights
4. Practical implementation requires careful consideration of many details
5. Many interesting questions remain open for research

---
**Any comments or corrections? Please feel free to reach out!**
*(Contact: pisquare@microsoft.com)* 