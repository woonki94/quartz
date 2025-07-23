#ML 

### What is Machine Learning

Machine Learning (ML) studies algorithms that:
- Improve performance **P**
- On a specific task **T**
- Based on past experience **E**

This is commonly written as:
> **Learning = improving P at T using E**

Example:
- **Task (T):** Facial recognition
- **Experience (E):** Labeled images
- **Performance (P):** Accuracy of identifying faces

---
### Types of Machine Learning

ML methods are often categorized by the type of supervision available in the data:

| Type                    | Description                                                                   |
| ----------------------- | ----------------------------------------------------------------------------- |
| Supervised Learning     | Learn from labeled examples (input-output pairs).                             |
| Unsupervised Learning   | Discover patterns from unlabeled data (e.g., clustering, density estimation). |
| Reinforcement Learning  | Learn through interactions and rewards to make sequential decisions.          |
| Semi-Supervised / Other | Include Active Learning, Transfer Learning, etc.                              |

---

### Supervised Learning

Learn a function mapping input **𝑥** to output **𝑦** based on labeled training data.

#### Example Tasks:
- **Regression:** Predict continuous output  
  `Ex: Predict house price from square footage`
- **Classification:** Predict discrete labels  
  `Ex: Classify loan risk from income and savings`
- **Structured Prediction:** Predict structured outputs  
  `Ex: POS tagging, semantic segmentation`

---

### Unsupervised Learning

Find structure in unlabeled data:
- **Clustering:** Group similar items  
  `Ex: Segment customers by behavior`
- **Density Estimation:** Estimate underlying data distribution  
  `Ex: Anomaly detection`
- **Dimensionality Reduction:** Project high-dim data into lower-dim space  
  `Ex: Visualizing gene expression`

---

### Reinforcement Learning

An **agent** interacts with an environment:
- Takes **actions**
- Observes **rewards**
- Learns a **policy** to maximize long-term reward

No supervised labels are given. Instead, learning is driven by delayed rewards.

---

### Key Components of a Learning System

1. **Data Representation (Features)**  
   - E.g., pixel values, word embeddings, metadata
2. **Hypothesis Space**  
   - Set of all functions the model might learn (e.g., linear models, trees)
3. **Loss Function**  
   - Quantifies prediction error (e.g., MSE, cross-entropy)
4. **Optimization**  
   - Method to minimize loss (e.g., gradient descent)
5. **Evaluation Metric**  
   - Measures generalization (e.g., accuracy, F1, AUC)

---

### When is Supervised Learning Appropriate?

- When humans can do the task but can't explain it
	`EX: Segmentation, Voice recognition`
- When each individual requires a personalized function
	`EX: Recommnedation systems in Netflix`
- When the true function is changing over time
	`EX: Finance Chart`
- When experts lack full domain knowledge  
	`Ex: drug/material discovery using ML models`

---

