#logistic-regression #classification

### What is Logistic Regression?

Problem setup
- **Input**: feature vector $\mathbf{x}$
- **Output**: class label $y \in \{0, 1\}$
- Task: Learn a model that estimates the **probability** of the class label given input

Logistic Regression is a probabilistic model for **binary classification**.  
Given a feature vector $\mathbf{x} \in \mathbb{R}^d$, the label $y \in \{0,1\}$ is predicted using:

$$
P(y = 1 \mid \mathbf{x}; \mathbf{w}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}
$$

This is known as the **sigmoid function**, which maps real values into probabilities in $(0, 1)$.

*My intuition about logistic regression*
- Logistic Regression is a statistical model used for binary classification (i.e spam detector)
- predicting whether an instance belongs to class 1 or class 0.



#### Connection with Linear Regression

- Linear regression models a continuous output: $y = \mathbf{w}^T \mathbf{x}$
- Logistic regression transforms this via sigmoid:
  $$
  P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x})
  $$
- For classification, we care about predicting $y \in \{0,1\}$  
  So, logistic regression turns the linear output into a **probability score**.

### Decision Making with Logistic Regression
>Detailed MAP [[Map decision Rule]]
#### Decision Rule (MAP Prediction)
#maximum_a_posteriori 
We predict:
$$
\hat{y} = \arg\max_{v \in \{0,1\}} P(y = v \mid \mathbf{x})
$$

This simplifies to:
$$
\hat{y} = \begin{cases}
1 & \text{if } \mathbf{w}^T \mathbf{x} \ge 0 \\
0 & \text{otherwise}
\end{cases}
$$

So the decision boundary is **linear**: $\mathbf{w}^T \mathbf{x} = 0$

#### Geometric Interpretation

- $\mathbf{w}$ defines the orientation of the boundary
- The decision boundary is a hyperplane perpendicular to $\mathbf{w}$
- The bias term $w_0$ shifts the boundary up/down

#### Scaling Weights Doesn't Change the Boundary

Compare:
- Classifier A: $P(y=1 | x) = σ(w^T x)$
- Classifier B: $P(y=1 | x) = σ(10·w^T x)$

Both produce the **same decision boundary** under a 0.5 threshold,  
but Classifier B has **sharper** (more confident) probabilities.

#### When Magnitude Matters

- For different decision thresholds (not 0.5), the weight magnitude can affect predictions
    
- Example: A classifier tuned for high recall may use a lower threshold (e.g., 0.2)

---

###  Learning for Logistic Regression

Given data:
- $D={(x_1,y_1),…,(x_N,y_N)}$

Assume:
- Data points are **i.i.d.**
- Each label $y_i$ follows a **Bernoulli distribution**: $P(y_i=1∣x_i;w)=σ(w^Tx_i)$

The goal is to learn $\mathbf{w}$ via **Maximum Likelihood Estimation (MLE)**.

#### Maximum Likelihood Estimation (MLE)

We assume each label $y_i \in \{0,1\}$ is drawn from a **Bernoulli distribution**:

$$
P(y_i \mid \mathbf{x}_i; \mathbf{w}) = \sigma(\mathbf{w}^T \mathbf{x}_i)^{y_i} \cdot \left(1 - \sigma(\mathbf{w}^T \mathbf{x}_i)\right)^{1 - y_i}
$$

To fit the model, we maximize the log-likelihood:

$$
\ell(\mathbf{w}) = \sum_i \left[ y_i \log \sigma(\mathbf{w}^T \mathbf{x}_i) + (1 - y_i) \log (1 - \sigma(\mathbf{w}^T \mathbf{x}_i)) \right]
$$
#### Gradient of Log-likelihood

Using the derivative of sigmoid:  
$\sigma'(t) = \sigma(t)(1 - \sigma(t))$

Gradient becomes:
$$
\nabla \ell(\mathbf{w}) = \sum_i \left( y_i - \sigma(\mathbf{w}^T \mathbf{x}_i) \right) \mathbf{x}_i
$$
### Multinomial Logistic Regression (Softmax)
#softmax

For $K$ classes, we learn $K$ weight vectors $\mathbf{w}\_1, \dots, \mathbf{w}\_K$:

$$
P(y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x})}{\sum_{j=1}^K \exp(\mathbf{w}_j^T \mathbf{x})}
$$

Gradient for each $\mathbf{w}\_k$:
$$

\nabla_{\mathbf{w}_k} \ell = \sum_i \left( \mathbf{1}[y_i = k] - P(y_i = k \mid \mathbf{x}_i) \right) \mathbf{x}_i

$$

---

### MSE vs. Log-likelihood

You can train logistic regression with:

* **MSE**: $\sum\_i (y\_i - \sigma(w^T x\_i))^2$

→ not recommended due to flat gradients

* **Cross-Entropy (Log-likelihood)**:

Stronger gradient signals, better optimization behavior

---

### ❗ Problem with Linearly Separable Data

When data is linearly separable:

* MLE keeps increasing $|\mathbf{w}|$ to sharpen predictions
	$\rightarrow$ **Overfitting**

#### ✅ Solution: Regularization

We assume a **prior** over weights, $p(\mathbf{w})$:

* **Gaussian Prior**:

$$

p(\mathbf{w}) \propto \exp\left(-\frac{1}{2\sigma^2} \|\mathbf{w}\|^2\right)

$$
	→ L2 regularization

* **Laplace Prior**:

$$

p(\mathbf{w}) \propto \exp\left(-\frac{1}{b} \|\mathbf{w}\|_1\right)

$$

	→ L1 regularization

MAP estimation:

$$

\mathbf{w}_{MAP} = \arg\max_{\mathbf{w}} \log P(D \mid \mathbf{w}) + \log P(\mathbf{w})

$$

### Implementation

#### Dataset Context: Insurance Customer Interest

The dataset represents insurance customer data with the goal of predicting the **`Response`** — whether a customer is interested in purchasing a vehicle insurance policy.

**Feature Descriptions:**

- `Dummy`: Bias dummy variable (1 for all rows)
- `Gender`: Gender of the customer
- `Age`: Age of the customer
- `Driving_License`: 0 = No DL, 1 = Has DL
- `Region_Code`: Unique region identifier
- `Previously_Insured`: 1 = Has insurance, 0 = Doesn't
- `Vehicle_Age`: Vehicle's age category
- `Vehicle_Damage`: 1 = Past vehicle damage, 0 = None
- `Annual_Premium`: Yearly premium cost
- `PolicySalesChannel`: Encoded sales channel
- `Vintage`: Number of days associated with company
- `Response`: Target (1 = Interested, 0 = Not Interested)

---
#### Logistic Regression — Theoretical Background

- Data samples $(\mathbf{x}_i, y_i)$ are **i.i.d.**
- Labels $y_i \in \{0, 1\}$ follow a **Bernoulli distribution**:
  $$
  y_i \sim \text{Bernoulli}(p_i), \quad p_i = \sigma(\mathbf{w}^\top \mathbf{x}_i)
  $$
- $\sigma$ is the **sigmoid function**:
  $$
  \sigma(z) = \frac{1}{1 + e^{-z}}
  $$

#### Maximum Likelihood Estimation

The likelihood of the entire dataset:

$$
P(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) = \prod_{i=1}^n \sigma(\mathbf{w}^\top \mathbf{x}_i)^{y_i} (1 - \sigma(\mathbf{w}^\top \mathbf{x}_i))^{1 - y_i}
$$

Log-likelihood:

$$
\ell(\mathbf{w}) = \sum_{i=1}^n \left[ y_i \log \sigma(\mathbf{w}^\top \mathbf{x}_i) + (1 - y_i) \log (1 - \sigma(\mathbf{w}^\top \mathbf{x}_i)) \right]
$$

We minimize the **negative log-likelihood** (cross-entropy loss):

$$
L(\mathbf{w}) = -\ell(\mathbf{w})
$$



#### L2 Regularization (MAP Estimation)

We assume a **Gaussian prior** on the weights:

$$
P(\mathbf{w}) \propto \exp\left(-\frac{\lambda}{2} \|\mathbf{w}\|^2\right)
$$

MAP estimation leads to minimizing:

$$
L_{\text{reg}}(\mathbf{w}) = L(\mathbf{w}) + \frac{\lambda}{2} \|\mathbf{w}\|^2
$$

This is equivalent to **L2 regularization** (a.k.a. weight decay).

---
#### Experiment Setup

- Logistic regression trained via **batch gradient descent**
- Loss monitored across iterations
- Hyperparameters:
  - $\lambda$ values: $[10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1]$
  - Learning rates adjusted per $\lambda$
- Datasets:
  - Clean: `IA2-train.csv`
  - Noisy: `IA2-train-noisy.csv`
  - Validation: `IA2-dev.csv`


#### Observations

1. Clean Data Results

![[1.png]]
![[2.png]]
- Training loss does not converge with higher $\lambda$.
- Both **training and validation accuracy decreased** as $\lambda$ increased.
- Indicates **underfitting** caused by overly aggressive regularization.
- Suggests the clean dataset **does not require much regularization**.

####  Noisy Data Results

![[3.png]]
![[4.png]]


-  Training loss does not converge with higher $\lambda$.
- Moderate values of $\lambda$ **improve validation accuracy** while slightly reducing training accuracy.
- Demonstrates how L2 regularization helps **generalize better** in the presence of noise.
- The effect is especially noticeable when comparing curves for $\lambda = 10^{-2}$ vs. $\lambda = 0$.


#### Convergence Criteria

We used the following convergence check:
- If the **difference in loss** between two iterations falls below a threshold ($10^{-7}$), stop.

This is a **practical heuristic**, but not a guarantee of optimality.

 For Theoretical Optimality:
- The best condition is:
  $$
  \|\nabla L(\mathbf{w})\| < \varepsilon
  $$
  This ensures gradient descent has reached a (global) minimum — especially relevant since the loss is **convex**.


####  Takeaways

- Logistic regression loss is **convex**, and regularization **preserves** convexity.
- Regularization is **most useful when data is noisy or high-dimensional**.
- Large $\lambda$ leads to **underfitting**.
- Well, when the data is not noisy or not overfitting, regularization is not a good choice.
- Use **validation performance** to pick the optimal $\lambda$.
