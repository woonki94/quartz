
## 📘 Linear Regression

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

We define a **loss function** that quantifies prediction error on the training set:
$$
\mathcal{L}(\mathbf{w}) = \sum_{i=1}^N (\hat{y}_i - y_i)^2 = \sum_{i=1}^N (\mathbf{w}^T \mathbf{x}_i - y_i)^2
$$

This is known as the **Sum of Squared Errors (SSE)**.  
We often use the **Mean Squared Error (MSE)** instead:
$$
\mathcal{L}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2
$$

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

- If the loss is **convex** (which it is here), gradient descent will **converge to the global minimum**.
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

- **Batch GD:** Uses all $N$ samples per update  
- **Stochastic GD:** Updates weights after each sample  
- **Mini-batch GD:** Updates with a small batch of examples

> Tradeoff: batch = stable but slow; SGD = fast but noisy; mini-batch = sweet spot

---

### Choosing the Learning Rate $\gamma$

- **Fixed:** can diverge if too large, or converge slowly if too small
- **Scheduled:** e.g., $\gamma_t = \frac{1}{t}$
- **Adaptive:** increase/decrease based on whether loss improves

---

### Summary So Far

- **Hypothesis**: Linear function $\hat{y} = \mathbf{w}^T \mathbf{x}$
- **Loss**: MSE
- **Optimization**: Gradient descent
- **Convexity**: Guarantees global minimum
- **Variants**: Batch, Stochastic, Mini-batch
