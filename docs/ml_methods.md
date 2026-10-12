# Machine Learning Methods & Mathematical Formulations

This document details the mathematical theory, hyperparameters, objective functions, evaluation metrics, and practical trade-offs for all algorithms implemented in RoadSafety_ML.

---

## 1. Linear Regression Models

Predicts continuous accident impact / blockage distance in miles ($y = \text{Distance(mi)}$).

### 1.1 Standard Ordinary Least Squares (OLS)
Minimizes the sum of squared residual errors between observed and predicted targets:
$$\min_{w} \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - x_i^T w \right)^2$$

### 1.2 Ridge Regression ($L_2$ Regularization)
Adds an $L_2$ Euclidean norm weight penalty to prevent coefficient explosion under multicollinearity:
$$\min_{w} \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - x_i^T w \right)^2 + \alpha \sum_{j=1}^{p} w_j^2$$

### 1.3 Lasso Regression ($L_1$ Regularization)
Adds an $L_1$ Manhattan norm penalty driving non-informative feature weights strictly to zero (sparse feature selection):
$$\min_{w} \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - x_i^T w \right)^2 + \alpha \sum_{j=1}^{p} |w_j|$$

### 1.4 ElasticNet Regression
Convex combination of $L_1$ and $L_2$ penalties:
$$\min_{w} \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - x_i^T w \right)^2 + \alpha \left( \rho \sum_{j=1}^{p} |w_j| + \frac{1 - \rho}{2} \sum_{j=1}^{p} w_j^2 \right)$$

---

## 2. Logistic Regression Models

Predicts binary crash severity tier ($y \in \{0, 1\}$, where $1 = \text{Severe}, 0 = \text{Moderate/Minor}$).

The sigmoid activation maps linear inputs to class probabilities:
$$P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = w_0 + \sum_{j=1}^{p} w_j x_j$$

### Regularized Log-Loss Formulations:
- **Standard**: Minimizes binary cross-entropy loss.
- **Ridge ($L_2$)**: Penalizes large log-odds ratios.
- **Lasso ($L_1$)**: Prunes irrelevant weather features from the decision boundary.
- **ElasticNet**: Balanced feature shrinkage and selection.

---

## 3. Decision Trees & Ensemble Methods

### 3.1 Base Decision Tree
Recursively partitions feature space by maximizing Gini Impurity reduction at each internal split:
$$I_G(p) = 1 - \sum_{k=1}^{K} p_k^2$$

### 3.2 Bagging Classifier (Bootstrap Aggregating)
Trains parallel independent decision trees on randomly resampled subsets (with replacement) and aggregates votes to reduce variance.

### 3.3 Random Forest
Decorrelates individual trees by selecting a random subset of $m \approx \sqrt{p}$ features at every split candidate.

### 3.4 AdaBoost (Adaptive Boosting)
Trains weak learners sequentially, up-weighting previously misclassified accident instances:
$$w_i^{(t+1)} = w_i^{(t)} \exp\left( -\alpha_t y_i h_t(x_i) \right)$$

### 3.5 Gradient Boosting (GBM)
Sequentially adds decision trees fitted directly to the negative pseudo-residuals (gradients) of the loss function.

### 3.6 XGBoost (Extreme Gradient Boosting)
Regularized gradient tree boosting utilizing second-order Taylor expansions (gradients and Hessians) and built-in leaf penalty terms.

### 3.7 LightGBM (Histogram-based Boosting)
Bins continuous features into discrete integer buckets and utilizes leaf-wise (best-first) tree expansion for superior speed on large datasets.

---

## 4. Unsupervised Clustering Algorithms

### 4.1 K-Means Clustering
Partitions accident records into $k$ distinct spatial/meteorological clusters by iteratively minimizing Within-Cluster Sum of Squares (Inertia):
$$\text{WCSS} = \sum_{k=1}^{K} \sum_{x \in C_k} \|x - \mu_k\|^2$$
- **Elbow Method**: Evaluates the inflection point on the Inertia vs. $k$ curve.

### 4.2 DBSCAN & Automated Knee Detection
- **Density-Based Spatial Clustering of Applications with Noise**: Discovers arbitrary-shaped spatial clusters without requiring a pre-specified cluster count.
- **Core Point**: Point with at least `min_samples` within distance `eps`.
- **Border Point**: Non-core point within distance `eps` of a core point.
- **Noise Point**: Isolated point outside any dense neighborhood (`label = -1`).
- **Automated Knee Epsilon Detection (`knee_for_dbscan.py`)**:
  1. Computes the sorted distance from every sample to its $k$-th nearest neighbor ($k = \text{min\_samples}$).
  2. Plots the sorted $k$-distance graph.
  3. Detects the maximum perpendicular distance from the curve to the baseline chord (Kneedle algorithm) to recommend the optimal $\epsilon$.

### 4.3 Hierarchical Agglomerative Clustering
Bottom-up clustering iteratively merging the closest pairs of clusters based on linkage criterion:
- **Ward**: Minimizes total within-cluster variance.
- **Complete**: Maximum pairwise distance between cluster members.
- **Average**: Average pairwise distance between cluster members.
- **Single**: Minimum pairwise distance between cluster members.
- Generates reproducible tree dendrograms for visual hierarchical interpretation.

---

## 5. Evaluation Metrics

| Metric | Domain | Description |
| :--- | :--- | :--- |
| **$R^2$ / Adjusted $R^2$** | Regression | Proportion of target variance explained by predictor variables. |
| **RMSE / MAE** | Regression | Root Mean Squared Error / Mean Absolute Error in original target units (miles). |
| **Accuracy / Precision / Recall** | Classification | Overall correctness, true positive rate among positive predictions, and true positive coverage. |
| **F1-Score** | Classification | Harmonic mean of Precision and Recall. |
| **ROC-AUC** | Classification | Area under the Receiver Operating Characteristic curve across all classification thresholds. |
| **Silhouette Score** | Clustering | Measures how similar an object is to its own cluster compared to other clusters ($-1$ to $+1$). |
| **Davies-Bouldin Index** | Clustering | Ratio of within-cluster distance to between-cluster distance (lower is better). |
