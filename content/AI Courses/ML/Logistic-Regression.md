#logistic-regression #classification

### What is Logistic Regression?

Logistic Regression is a probabilistic model for **binary classification**.  
Given a feature vector $\mathbf{x} \in \mathbb{R}^d$, the label $y \in \{0,1\}$ is predicted using:

$$
P(y = 1 \mid \mathbf{x}; \mathbf{w}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}
$$

This is known as the **sigmoid function**, which maps real values into probabilities in $(0, 1)$.


#### Connection with Linear Regression

- Linear regression models a continuous output: $y = \mathbf{w}^T \mathbf{x}$
- Logistic regression transforms this via sigmoid:
  $$
  P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x})
  $$
- For classification, we care about predicting $y \in \{0,1\}$  
  So, logistic regression turns the linear output into a **probability score**.


#### Decision Rule (MAP Prediction)

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

####  Geometric Interpretation

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


