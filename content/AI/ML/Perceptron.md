
### Perceptron Algorithm
#### 🧠 What Does the Perceptron Do?

- Solves **binary classification** problems with labels $y \in \{-1, +1\}$
- Learns a **linear decision boundary**:
  $$
  f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}
  $$
- Predicts:
		- $f(x) = 1$ *if* $\mathbf{w}^\top \mathbf{x} \ge 0$
		- $f(x) = -1$ *Otherwise*
  $$
  \hat{y} = \text{sign}(f(\mathbf{x})) = \text{sign}(\mathbf{w}^\top \mathbf{x})
  $$
	

#### Perceptron vs. Logistic Regression

|                    | **Perceptron**                                            | **Logistic Regression**                           |
| ------------------ | --------------------------------------------------------- | ------------------------------------------------- |
| Output             | Hard label: $\{-1, +1\}$                                  | Probability: $P(y=1 \mid \mathbf{x})$             |
| Decision Rule      | $\text{sign}(\mathbf{w}^\top \mathbf{x})$                 | Threshold on $\sigma(\mathbf{w}^\top \mathbf{x})$ |
| **Probabilistic?** | ❌ No probabilistic interpretation                         | ✅ Models $P(y \mid \mathbf{x})$ using sigmoid     |

Key Idea:  
- **Perceptron** makes **hard predictions** and updates on mistakes  
- **Logistic regression** makes **probabilistic predictions** and learns via likelihood maximization


### 0/1 Loss Function - Very intuitive choice
#### Definition

The **0/1 loss** measures whether the prediction is correct or not:

$$
L_{0/1}(\mathbf{w}; \mathbf{x}_i, y_i) = 
\begin{cases}
1 & \text{if } y_i \cdot \mathbf{w}^\top \mathbf{x}_i \le 0 \quad \text{(mistake)} \\
0 & \text{if } y_i \cdot \mathbf{w}^\top \mathbf{x}_i > 0 \quad \text{(correct)}
\end{cases}
$$

#### Intuition

- Returns **1** if the prediction is **wrong**
- Returns **0** if the prediction is **correct**
- Matches classification accuracy exactly

![[Pasted image 20250627114150.png|400]]

#### Why Not Optimize 0/1 Loss?

- **Non-continuous**: abrupt jump between 0 and 1
- **Non-differentiable**: no gradient
- **Non-convex**: has many local minima

This makes it **impossible to optimize directly** with gradient-based methods.


### 📉 Perceptron Loss

---

#### Motivation

The **0/1 loss** is not optimizable due to being:
- Discrete
- Non-differentiable
- Non-convex

To allow for learning, we replace it with a **convex surrogate** — the **perceptron loss**.

#### Definition

The **perceptron loss** for a single example $(\mathbf{x}, y)$ is:

$$
L_{\text{perceptron}}(\mathbf{w}; \mathbf{x}, y) = \max(0, -y \cdot \mathbf{w}^\top \mathbf{x})
$$

#### Intuition

- If the prediction is **correct and confident**:  
  $y \cdot \mathbf{w}^\top \mathbf{x} > 0$ → loss = 0
- If the prediction is **wrong or too close to 0**:  
  $y \cdot \mathbf{w}^\top \mathbf{x} \le 0$ → loss > 0

 Only penalizes **mistakes** or **uncertain predictions**

![[Pasted image 20250627114028.png|400]]

#### Properties

- **Convex** ✅  
- **Piecewise linear** ✅  
- **Subdifferentiable** ✅  
- **Non-smooth at 0** ❗

### Subgradients and the Perceptron Loss

#### What is a Subgradient?
#subgradient

For a **convex but non-differentiable function** $f(\mathbf{w})$, the **subgradient** at a point $\mathbf{w}_0$ is any vector $g$ that satisfies:

$$
f(\mathbf{w}) \ge f(\mathbf{w}_0) + g^\top (\mathbf{w} - \mathbf{w}_0) \quad \forall \mathbf{w}
$$

- At differentiable points: subgradient = gradient
- At **non-smooth** points: subgradient = set of valid directions


####  Why Not Just Use Gradients?

- The **perceptron loss** is convex but **not differentiable at 0**
- At the point where $y \cdot \mathbf{w}^\top \mathbf{x} = 0$, the loss function has a **kink**

This means:
- **Gradient doesn’t exist**
- But we can still **optimize** using any valid **subgradient**

#### Perceptron Loss

$$
L(\mathbf{w}; \mathbf{x}, y) = \max(0, -y \cdot \mathbf{w}^\top \mathbf{x})
$$

It’s piecewise linear:
- Flat (0) when $y \cdot \mathbf{w}^\top \mathbf{x} > 0$
- Linear slope when $y \cdot \mathbf{w}^\top \mathbf{x} < 0$
- **Kink at 0** → not differentiable

#### Subgradient at Each Region

| Region                          | Subgradient                    |
|---------------------------------|--------------------------------|
| $y \cdot \mathbf{w}^\top \mathbf{x} > 0$ | $\nabla = 0$                        |
| $y \cdot \mathbf{w}^\top \mathbf{x} < 0$ | $\nabla = -y \cdot \mathbf{x}$     |
| $y \cdot \mathbf{w}^\top \mathbf{x} = 0$ | Any $\nabla \in [-y \cdot \mathbf{x}, 0]$ |

In practice, the perceptron update uses:
$$
\nabla L_i(\mathbf{w}) =
\begin{cases}
    0 & \text{if } y_i \mathbf{w}^\top \mathbf{x}_i > 0 \\
    -y_i \mathbf{x}_i & \text{elif } y_i \mathbf{w}^\top \mathbf{x}_i \le 0 
\end{cases}
$$


### Effect on Perceptron Update

####  Update Rule from Subgradient

The perceptron uses **subgradient descent** with fixed step size $\eta = 1$:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \nabla L_i(\mathbf{w})
$$

Plugging in the subgradient:

- **If correct prediction** ($y_i \mathbf{w}^\top \mathbf{x}_i > 0$):
  $$
  \nabla L_i(\mathbf{w}) = 0 \quad \Rightarrow \quad \mathbf{w} \leftarrow \mathbf{w}
  $$
  → No update

- **If mistake** ($y_i \mathbf{w}^\top \mathbf{x}_i \le 0$):
  $$
  \nabla L_i(\mathbf{w}) = -y_i \mathbf{x}_i \quad \Rightarrow \quad \mathbf{w} \leftarrow \mathbf{w} + y_i \mathbf{x}_i
  $$

- The perceptron **only updates on errors**.
- The direction of update **pulls $\mathbf{w}$ closer** to align with the misclassified example:
  - If $y_i = +1$: push $\mathbf{w}$ more in the direction of $\mathbf{x}_i$
  - If $y_i = -1$: push $\mathbf{w}$ away from $\mathbf{x}_i$

 Each mistake adjusts the decision boundary to correct that point.



### Perceptron Convergence Theorem

#### Theorem (Novikoff, 1962)

Assume:
- Data is **linearly separable** with margin $\gamma > 0$
- Each input satisfies $\|\mathbf{x}_i\| \le D$

Then, the perceptron algorithm will:
- **Make at most** $\left( \frac{D}{\gamma} \right)^2$ updates
- **Converge in finite time** (no infinite loops)

#### Margin Definition

The data is linearly separable with margin $\gamma$ if there exists a **unit vector** $\mathbf{u}$ such that:

$$
\forall i,\quad y_i (\mathbf{u}^\top \mathbf{x}_i) \ge \gamma
$$

- $\mathbf{u}$ is the **true separator**
- $\gamma$ is the **minimum signed distance** of any point to the hyperplane

#### Proof Sketch

Let:
- $\mathbf{w}^{(t)}$ = weight vector after $t$ updates
- Let $\mathbf{u}$ be the unit-norm optimal separator

We’ll prove an upper bound on $t$ by analyzing:

##### 1. Growth of alignment with the true separator

Let’s look at the **dot product** $\mathbf{w}^{(t)} \cdot \mathbf{u}$:

- Each update (due to mistake):
  $$
  \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + y_i \mathbf{x}_i
  $$
- So:
  $$
  \mathbf{w}^{(t+1)} \cdot \mathbf{u} = \mathbf{w}^{(t)} \cdot \mathbf{u} + y_i \mathbf{x}_i \cdot \mathbf{u}
  $$

- By the margin assumption:
  $$
  y_i (\mathbf{x}_i \cdot \mathbf{u}) \ge \gamma \quad \Rightarrow \quad \mathbf{w}^{(t)} \cdot \mathbf{u} \ge t \gamma
  $$

So the projection of $\mathbf{w}^{(t)}$ onto $\mathbf{u}$ **grows linearly** with $t$:
$$
\mathbf{w}^{(t)} \cdot \mathbf{u} \ge t \gamma
$$


##### 2. Growth of weight norm

At each mistake, we add $y_i \mathbf{x}_i$:

- Norm growth:
  $$
  \|\mathbf{w}^{(t+1)}\|^2 = \|\mathbf{w}^{(t)}\|^2 + 2 y_i \mathbf{w}^{(t)} \cdot \mathbf{x}_i + \|\mathbf{x}_i\|^2
  $$
- But since we only update on mistakes, $y_i \mathbf{w}^{(t)} \cdot \mathbf{x}_i \le 0$
- So:
  $$
  \|\mathbf{w}^{(t+1)}\|^2 \le \|\mathbf{w}^{(t)}\|^2 + \|\mathbf{x}_i\|^2 \le \|\mathbf{w}^{(t)}\|^2 + D^2
  $$

By induction:
$$
\|\mathbf{w}^{(t)}\|^2 \le t D^2
$$

##### 3. Putting It Together

We have:
1. $\mathbf{w}^{(t)} \cdot \mathbf{u} \ge t \gamma$
2. $\|\mathbf{w}^{(t)}\| \le \sqrt{t} D$

So apply Cauchy–Schwarz:
$$
\mathbf{w}^{(t)} \cdot \mathbf{u} \le \|\mathbf{w}^{(t)}\| \cdot \|\mathbf{u}\| = \|\mathbf{w}^{(t)}\| \le \sqrt{t} D
$$

Combine:
$$
t \gamma \le \sqrt{t} D \quad \Rightarrow \quad \sqrt{t} \ge \frac{t \gamma}{D} \quad \Rightarrow \quad t \le \left( \frac{D}{\gamma} \right)^2
$$

##### Conclusion

If data is linearly separable with margin $\gamma$ and bounded norm $D$:

> The perceptron algorithm will make **at most** $\left(\frac{D}{\gamma}\right)^2$ updates.


### Details about Margin in Perceptron Learning

#### What is Margin?

The **margin** is a geometric measure of **confidence** in a linear classifier’s prediction.

Formally, for a linear classifier $\mathbf{w}$, the **signed margin** for an example $(\mathbf{x}_i, y_i)$ is:

$$
\gamma_i = y_i \cdot \frac{\mathbf{w}^\top \mathbf{x}_i}{\|\mathbf{w}\|}
$$

- If $\gamma_i > 0$, the point is **correctly classified**
- If $\gamma_i < 0$, it's **misclassified**
- The larger $\gamma_i$, the **further** the point is from the decision boundary

#### Margin of a Dataset
#margin
#separating_hyperplane
The **margin of the dataset** under a separating hyperplane is:

$$
\gamma = \min_{i} \left( y_i \cdot \frac{\mathbf{w}^\top \mathbf{x}_i}{\|\mathbf{w}\|} \right)
$$

This is the **smallest margin** among all training points — the **worst-case distance** to the decision boundary.

 #### Margin and Normalization

In the perceptron convergence proof, we fix $\|\mathbf{u}\| = 1$, so margin simplifies to:

$$
\gamma = \min_i \left( y_i \cdot \mathbf{u}^\top \mathbf{x}_i \right)
$$

Here, $\mathbf{u}$ is a unit vector defining the true separating direction.  
Then $\gamma$ is the smallest projection of any sample onto $\mathbf{u}$, weighted by its label.


####  Why Does Margin Matter?

- If the margin $\gamma$ is **larger**, points are **well separated**
- The perceptron algorithm **converges faster**:
  $$
  \text{Max updates} \le \left(\frac{D}{\gamma}\right)^2
  $$
  where $D = \max_i \|\mathbf{x}_i\|$

- Intuitively:
  - Small $\gamma$ → hard-to-separate data → more updates
  - Large $\gamma$ → confident separation → fewer mistakes

#### Perceptron vs. Margin Maximization

- **Perceptron** only seeks *some* separating hyperplane
- **SVM** explicitly finds the one with **maximum margin**
- Perceptron might converge to a bad separator (small margin)

#### Key takeaways
- Margin measures how far points are from the decision boundary
- Larger margins lead to **faster convergence** and better generalization
- The perceptron doesn't maximize margin, but its performance depends on it


