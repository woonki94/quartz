###  Why Do We Need Kernel Methods?

#### Problem: Data Not Linearly Separable
- Linear classifiers (like perceptron, logistic regression, SVM) work well **only when data is linearly separable**.
- But many real-world datasets are **not** linearly separable in their original feature space.

**Example:** You can't separate two concentric circles using a straight line.

#### Idea: Feature Mapping

- Map input $\mathbf{x}$ to a **higher-dimensional space** via some transformation $\phi(\mathbf{x})$.
- In this new space, the data **may become linearly separable**.

This idea is backed by **Cover’s Theorem** (1965):
> “A complex classification problem is more likely to be linearly separable in a higher-dimensional space than in a low-dimensional one.”


### Problem with Explicit Mapping

- Computing $\phi(\mathbf{x})$ explicitly is **very expensive**, especially for high-degree polynomial or infinite-dimensional spaces.
- E.g., mapping to all 2nd-order interactions costs $\mathcal{O}(d^2)$; cubic features: $\mathcal{O}(d^3)$.

---

### Solution: The Kernel Trick

- Instead of computing $\phi(\mathbf{x})$ and $\phi(\mathbf{x'})$, just compute their **dot product**:
  $$
  K(\mathbf{x}, \mathbf{x'}) = \phi(\mathbf{x})^\top \phi(\mathbf{x'})
  $$

- This function $K$ is called a **kernel**.
- You can use kernels **without ever computing $\phi(\mathbf{x})$ explicitly**.

---

### Why This Matters

- You can now apply **linear algorithms (SVM, regression, perceptron)** in a **nonlinear feature space** implicitly.
- This allows learning **nonlinear decision boundaries** with minimal computational cost.

---

### Common Kernel Functions

- **Linear**: $K(\mathbf{x}, \mathbf{x'}) = \mathbf{x}^\top \mathbf{x'}$
- **Polynomial**: $K(\mathbf{x}, \mathbf{x'}) = (\mathbf{x}^\top \mathbf{x'} + 1)^d$
- **RBF (Gaussian)**: $K(\mathbf{x}, \mathbf{x'}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x'}\|^2}{2\sigma^2}\right)$
