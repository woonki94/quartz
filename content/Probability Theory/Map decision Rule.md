#map #maximum_a_posteriori
# Maximum A Posteriori

The **MAP decision rule** is used in Bayesian decision theory to make optimal classification decisions. It selects the class with the **highest posterior probability** given the observed data.

---

## Definition

Given an input observation $x$, the MAP decision rule chooses the class $C_k$ such that:
$$
\hat{C}(x) = \arg\max_k \, P(C_k \mid x)
$$

This means: **choose the class with the highest probability after seeing the data**.

---

## Using Bayes’ Theorem

Bayes’ Rule allows us to express the posterior $P(C_k \mid x)$ as:

$$
P(C_k \mid x) = \frac{P(x \mid C_k) P(C_k)}{P(x)}
$$

Since $P(x)$ is constant for all classes, we simplify:
$$
\hat{C}(x) = \arg\max_k \, P(x \mid C_k) P(C_k)
$$

- $P(x \mid C_k)$: Likelihood (how likely the data is given class $C_k$
- $P(C_k)$: Prior probability of class $C_k$

---

## MAP vs ML (Maximum Likelihood)

| Rule | Formula                              | Uses Prior? |
| ---- | ------------------------------------ | ----------- |
| MAP  | $\arg\max_k \, P(x \mid C_k) P(C_k)$ | ✅ Yes       |
| ML   | $\arg\max_k \, P(x \mid C_k)$        | ❌ No        |

MAP incorporates **prior knowledge** of class probabilities, ML does not.

---

## Example

Classify an email as "Spam" or "Not Spam".

- Priors:
  - $P(\text{Spam}) = 0.2$
  - $P(\text{Not Spam}) = 0.8$
- Likelihoods:
  - $P(x \mid \text{Spam}) = 0.6$
  - $P(x \mid \text{Not Spam}) = 0.5$

Compute:

$$
\begin{align*}
\text{Spam: } & 0.6 \cdot 0.2 = 0.12 \\
\text{Not Spam: } & 0.5 \cdot 0.8 = 0.4 \\
\end{align*}
$$

>MAP chooses: **Not Spam**

---

> 💡 Even though the likelihood for "Spam" is higher, the **prior belief** shifts the decision to "Not Spam".

---


- MAP = Bayesian rule for classification
- Chooses the most probable class **after seeing data**
- Incorporates both likelihood and prior
- Useful in imbalanced or prior-sensitive problems

