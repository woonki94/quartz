
## Linear Regression
#linear-regression
### What is Linear Regression?

Linear regression is a **supervised learning** method for **regression tasks**, where the goal is to predict a continuous value based on input features.
#### Example Tasks
| Prediction Target | Features                   |
| ----------------- | -------------------------- |
| House price       | Sqft, lot size, # of rooms |
| Species abundance | Environmental variables    |
| Stock price       | Price history              |

---

### Hypothesis Space: Linear Models

The hypothesis space $\mathcal{H}$ consists of all **linear functions** of the input features.

Let:
- $\mathbf{x} = [1, x_1, x_2, \dots, x_d]^T$ (augmented with bias)
- $\mathbf{w} = [w_0, w_1, \dots, w_d]^T$

Then:
$$
\hat{y}(\mathbf{x}; \mathbf{w}) = \mathbf{w}^T \mathbf{x} = w_0 + w_1 x_1 + \cdots + w_d x_d
$$

Each $\mathbf{w}$ defines a specific **linear predictor**.

---

### Learning as Loss Minimization
#loss 
We define a **loss function** that quantifies prediction error on the training set:
$$
\mathcal{L}(\mathbf{w}) = \sum_{i=1}^N (\hat{y}_i - y_i)^2 = \sum_{i=1}^N (\mathbf{w}^T \mathbf{x}_i - y_i)^2
$$

This is known as the **Sum of Squared Errors (SSE)**.  
We often use the **Mean Squared Error (MSE)** instead:
$$
\mathcal{L}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2
$$
> we Square, to make it differentiable(Gradient descent)

- In short, we aim to find linear function that best matches the given labeled data points ${(\mathbf{x}_i, y_i)}_{i=1}^N$ 
- Here, **$\mathbf{x}_i$ is fixed (known)** — it’s your input feature vector.
- You optimize over $\mathbf{w}$ to minimize some notion of error (typically Mean Squared Error). 
---
### Optimization: Gradient Descent

To minimize MSE, we apply **gradient descent**:

1. Start from a random guess $\mathbf{w}_0$
2. Update iteratively:
$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \gamma \nabla \mathcal{L}(\mathbf{w}_t)
$$

Gradient of MSE:
$$
\nabla \mathcal{L}(\mathbf{w}) = \frac{2}{N} \sum_{i=1}^N (\mathbf{w}^T \mathbf{x}_i - y_i)\mathbf{x}_i
$$

---

### Behavior of Gradient Descent

- If the loss is **convex** (which it is here), gradient descent will **converge to the global minimum**. #convex
- If the step size $\gamma$ is too large: it may **diverge**.
- If too small: **slow convergence**.
- A good practice: use **decaying learning rate** or **adaptive methods**.

---
### Convexity

A function $f$ is **convex** if:
$$
f(\theta x_1 + (1 - \theta)x_2) \leq \theta f(x_1) + (1 - \theta)f(x_2), \quad \forall \theta \in [0, 1]
$$

This means: line between any two points on the graph lies **above the curve**.

MSE is convex → single global minimum.

---

### Gradient Descent Variants
#gd
- **Batch GD:** Uses all $N$ samples per update (Update all)
- **Stochastic GD:** Updates weights after each sample  (Update one by one)
- **Mini-batch GD:** Updates with a small batch of examples (Update by size of batch)

All variants minimize the MSE loss:
$$
\mathcal{L}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (\mathbf{w}^T \mathbf{x}_i - y_i)^2
$$

#### Batch Gradient Descent (BGD)
Updates the weights using **all** training examples at each step.
```pseudo
Input: data (X, Y), learning rate γ, initial weights w₀
Repeat until convergence:
    grad ← (2 / N) * Xᵀ (Xw - Y)
    w ← w - γ * grad
```

#### Stochastic Gradient Descent 
Updates weights after each training example.
```psuedo
Input: data (X, Y), learning rate γ, initial weights w₀
Repeat for multiple epochs:
    Shuffle (X, Y)
    For each (xᵢ, yᵢ) in dataset:
        grad ← 2 * (wᵀ xᵢ - yᵢ) * xᵢ
        w ← w - γ * grad
```

#### Mini-Batch Gradient Descent (MBGD)
A compromise: updates weights using a small **batch** of examples.
```pseudo

Input: data (X, Y), learning rate γ, batch size B, initial weights w₀
Repeat for multiple epochs:
    Shuffle (X, Y)
    Partition data into batches of size B
    For each batch (X_b, Y_b):
        grad ← (2 / B) * X_bᵀ (X_b w - Y_b)
        w ← w - γ * grad


```
> B = 1 : Stochastic
> B = N : Batch 
> Tradeoff: batch = stable but slow; SGD = fast but noisy; mini-batch = sweet spot


### Closed-Form Solution for Linear Regression
#closed-form
Instead of using gradient descent, we can solve for $\mathbf{w}$ **analytically** by setting the gradient to zero.

Given dataset:
- $X \in \mathbb{R}^{N \times d}$ is the **design matrix**
- $Y \in \mathbb{R}^N$ is the **vector of labels**

Then the **MSE loss** can be written with the norm:
$$
\nabla \mathcal{L}(\mathbf{w}) = \frac{2}{N} \sum_{i=1}^N (\mathbf{w}^T \mathbf{x}_i - y_i)\mathbf{x}_i =  \frac{1}{N} \| X\mathbf{w} - Y \|^2
$$

Gradient:
$$
\nabla \mathcal{L}(\mathbf{w}) = \frac{2}{N} X^T (X\mathbf{w} - Y)
$$

Set gradient to zero:
$$
X^T X \mathbf{w} = X^T Y
$$

**Solution**:
$$
\mathbf{w} = (X^T X)^{-1} X^T Y
$$

> This is called the **normal equation**.  
> If $X^T X$ is invertible, we can compute $\mathbf{w}$ exactly.
> This solution is same for both MSE & SSE (Think about it, it's obvious) 
> Note: in gd case, they are similar but have to use different step size.

---

### Interpretation: Least Squares

This closed-form $\mathbf{w}$ minimizes the squared residuals:
$$
\min_{\mathbf{w}} \sum_{i=1}^N (\mathbf{w}^T \mathbf{x}_i - y_i)^2
$$

It's also known as the **least squares solution** to the system $X\mathbf{w} \approx Y$.

When $X^T X$ is not invertible (e.g. $X$ has linearly dependent columns), we use the **Moore-Penrose pseudo-inverse**:
#pseudo-inverse
$$
\mathbf{w} = X^\dagger Y
$$

---

### Background: Maximum Likelihood Estimation (MLE)
#MLE

#### What is Parameter Estimation?

We assume data is generated from some true function with noise:
$$
y_i = f(\mathbf{x}_i) + \varepsilon_i
$$

We want to learn parameters (like $\mathbf{w}$) that **best approximate** this unknown function.

In **linear regression**, we assume:
$$
f(\mathbf{x}) = \mathbf{w}^T \mathbf{x}
\quad \Rightarrow \quad
y_i = \mathbf{w}^T \mathbf{x}_i + \varepsilon_i
$$

#### Maximum Likelihood Principle

Given a probabilistic model $P(\mathbf{x}; \theta)$ (where $\theta$ are unknown parameters), we observe $N$ **i.i.d.** samples:
#IID
> In MLE we assume IID 


$$
\mathcal{D} = \{ \mathbf{x}_1, \dots, \mathbf{x}_N \}
$$

The **likelihood** of this data is:
$$
P(\mathcal{D}; \theta) = \prod_{i=1}^N P(\mathbf{x}_i; \theta)
$$

The maximum likelihood estimator (MLE) is:
$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} P(\mathcal{D}; \theta)
$$

#### Why Log-Likelihood?

Products of small probabilities can underflow. We take log:
$$
\log P(\mathcal{D}; \theta) = \sum_{i=1}^N \log P(\mathbf{x}_i; \theta)
$$

>This also turns products into sums — easier to optimize.

#### Coin Toss Example

Assume we toss a biased coin $N$ times and get:
- $h$ heads
- $t$ tails (so $N = h + t$)
>This is the known distribution. With unknown samples we assume Gaussian or Laplace

Let $\theta$ be the probability of heads. Then:
$$
P(\mathcal{D}; \theta) = \theta^h (1 - \theta)^t
$$

**Log-likelihood**:
$$
\log P(\mathcal{D}; \theta) = h \log \theta + t \log (1 - \theta)
$$

To find $\hat{\theta}_{\text{MLE}}$, take derivative:
$$
\frac{d}{d\theta} \log P(\mathcal{D}; \theta) = \frac{h}{\theta} - \frac{t}{1 - \theta}
$$

Set to 0 and solve:
$$
\hat{\theta}_{\text{MLE}} = \frac{h}{h + t}
$$

> Intuition: MLE returns the **empirical frequency** of heads

#### General MLE Procedure

1. Choose a probabilistic model $P(\text{data} \mid \theta)$
2. Form the **log-likelihood** over dataset $\mathcal{D}$
3. Maximize the log-likelihood:
   $$
   \hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \sum_{i=1}^N \log P(x_i \mid \theta)
   $$


### Apply to Regression

Assume that the target variable $y$ is generated as:
$$
y_i = \mathbf{w}^T \mathbf{x}_i + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

This means:
- $y_i$ is a **random variable**
- Each output is the linear prediction **plus Gaussian noise**

So the conditional distribution of $y_i$ given $\mathbf{x}_i$ is:
$$
P(y_i \mid \mathbf{x}_i; \mathbf{w}) = \mathcal{N}(\mathbf{w}^T \mathbf{x}_i, \sigma^2)
$$

#### Likelihood of Dataset
#iid
For $N$ i.i.d. training examples:
$$
P(\mathbf{Y} \mid \mathbf{X}, \mathbf{w}) = \prod_{i=1}^N P(y_i \mid \mathbf{x}_i; \mathbf{w}) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi} \sigma} \exp\left( -\frac{(y_i - \mathbf{w}^T \mathbf{x}_i)^2}{2\sigma^2} \right)
$$

#### Log-Likelihood

Taking the log (for convenience and convexity):
$$
\log P(\mathbf{Y} \mid \mathbf{X}, \mathbf{w}) = -\frac{N}{2} \log(2\pi \sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2
$$

- The first term is constant w.r.t. $\mathbf{w}$
- The second term involves the **Sum of Squared Errors (SSE)** with negative at front.


Thus, maximizing the log-likelihood is equivalent to minimizing:
$$
\sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2
$$

> So **minimizing MSE is equivalent to maximizing likelihood** under a Gaussian noise assumption.

#### Interpretation

- This connects **linear regression** to **probabilistic modeling**
- MSE arises naturally from **likelihood under Gaussian noise**
- We are learning a **discriminative model**: $P(y \mid \mathbf{x})$

---

### Regularized Linear Regression

#### Why Regularize?

In linear regression, overfitting is often linked to **large and unstable weight values**. Especially with higher-degree polynomial features or small training data, the model fits noise and loses generalization.

####  Regularized Objective

To avoid overfitting, we modify the loss function:

$$
\min_{\mathbf{w}} \frac{1}{N} \sum_{i=1}^N \left(y_i - \mathbf{w}^\top \mathbf{x}_i \right)^2 + \lambda \sum_{j=1}^M |w_j|^q
$$

This is composed of:
- **Data term** (mean squared error)
- **Regularization term** (penalizes large weights)

The parameter $\lambda$ controls the **trade-off**:
- Large $\lambda$: simpler model, smaller weights
- Small $\lambda$: better fit to training data

#### L2-Regularization (Ridge)

If $q = 2$, the regularizer becomes:
$$
\lambda \|\mathbf{w}\|_2^2 = \lambda \sum_j w_j^2
$$

This gives a **closed-form solution**:
$$
\mathbf{w}_{\text{ridge}} = \left(N \lambda I + X^\top X \right)^{-1} X^\top Y
$$

- Adds $\lambda I$ to the matrix $X^\top X$, improving **numerical stability**
- Helps when $X^\top X$ is **not full rank** or **ill-conditioned**

#### Interpretation

L2 regularization corresponds to assuming a **Gaussian prior** over weights with zero mean. This shrinks the weights uniformly and prevents large fluctuations in their values.


#### Equivalence: Constraint vs. Penalty

The penalty form:
$$
\min_{\mathbf{w}} \frac{1}{N} \sum (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_q^q
$$

Is equivalent to the constraint form:
$$
\min_{\mathbf{w}} \frac{1}{N} \sum (y_i - \mathbf{w}^\top \mathbf{x}_i)^2 \quad \text{subject to } \|\mathbf{w}\|_q^q \leq \epsilon
$$

Different $\lambda$ values correspond to different $\epsilon$ values.

 >This equivalence connects **regularized optimization** to **constrained optimization**.
 > Constrain shape impacts solution. (More discussed in non-linear) #constrained_opt 
 

### Implementation

#### Overview
This code implements **linear regression** for house price prediction using both:

- **Closed-form solution** (normal equation)
- **Batch gradient descent** (iterative optimization)

It includes:
- Preprocessing
- Feature engineering
- Training
- Evaluation (MSE)
- Visualization of gradient descent convergence of varying step size.

#### Preprocessing Pipeline
-  `convert_date_to_features(df)`
	- Converts `'date'` from string to datetime.
	- Extracts: `day`, `month`, `year`.
	- Drops original `'date'` column.
	- Adds temporal features useful for modeling.

 - `age_since_renovated(df)`
	- Creates a **cleaner age-based feature** by combining `yr_renovated` and `yr_built`.
	- Handles `yr_renovated = 0` (no renovation) properly.
	- Drops `yr_renovated` to reduce noise.

- `normalize(data)`
	- Applies **z-score normalization**:  
	  $$
	  x_{\text{normalized}} = \frac{x - \mu}{\sigma}
	  $$
	- Helps gradient descent converge faster by making feature scales uniform.


#### Solving W
-  `closed_form_sol(x, y)`
	- Computes:
	  $$
	  w = (X^T X)^{-1} X^T y
		$$
	- ✅ Efficient for small to medium datasets.
	- ❌ Infeasible for huge `X` due to matrix inversion.

- `batch_gradient_descent(...)`
	- Initializes weights to zeros.
	- Iteratively updates weights using:
	  $$
	  w := w - \gamma \cdot \nabla \text{MSE}
	  $$

#### Results

After training the linear regression model using both **closed-form solution** and **batch gradient descent**, the following MSE (Mean Squared Error) values were observed:

| Method                 | Train MSE        | Validation MSE   |
|------------------------|------------------|------------------|
| Closed-form solution   | 0.3047           | 0.3092           |
| Batch gradient descent | 0.3047           | 0.3092           |

Observations:
- Both methods yield **almost identical results**, confirming that gradient descent converged properly to the optimal solution.
- Slight numerical differences (in the order of ~1e-11) are due to **floating-point precision**, not model quality.
- The validation MSE being slightly higher than the training MSE is **expected** and suggests reasonable generalization.

Plot of various LR:
- Below is a plot of **training loss (MSE)** over epochs for various learning rates (step sizes) using **batch gradient descent**:
![[myplot.png]]

Observations:
- **Larger learning rates** (e.g., 0.1) converge faster but may become unstable if too high.
- **Smaller learning rates** (e.g., 0.0001) converge slowly but more smoothly.
- All curves appear to **eventually converge** to a similar final MSE, confirming proper scaling and gradient behavior.
- This visualization is useful to **diagnose learning rate choice** and ensures the model doesn't diverge.

####  Notable Points

1. Why Add Bias with `ones_column` #bias
	- Bias term \( w_0 \) allows predictions **not to be anchored at the origin**.
	- Without it, model would wrongly predict `y=0` when all inputs are zero.

2. Why Normalize Data? #normalization
	- Without normalization, features on different scales (e.g. square feet vs. month) would cause:
	  - Gradient descent to converge slowly
	  - Numerical instability
	- Normalization gives each feature **equal influence** in optimization.

3. What is `inplace=True` in pandas? #inplace
	In pandas, many methods like `.drop()`, `.fillna()`, `.sort_values()`, etc., return a **new DataFrame by default**, and **do not change the original one** unless you explicitly ask them to do so **in-place**.




### Things to Remember for this page

Maximum Likelihood Estimation (MLE) under a Gaussian noise assumption justifies minimizing the Sum of Squared Errors (SSE), or equivalently, the Mean Squared Error (MSE), to find the optimal weights in linear regression.

