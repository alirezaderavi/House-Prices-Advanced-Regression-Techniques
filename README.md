# 🏠 House Price Prediction with Hybrid Ensemble Learning

## 📌 Overview

This project develops a machine learning framework to predict house prices using advanced ensemble techniques. The approach combines multiple regression models and explores the impact of clustering-based segmentation to improve predictive performance.

The goal is to move beyond basic models and evaluate whether hybrid strategies (stacking + clustering) provide measurable improvements.

---

## 🚀 Methodology

### 1. Data Preparation

* Selected relevant features (`final_features`)
* Target variable transformed using:

  ```python
  y = log1p(SalePrice)
  ```
* Train-validation split:

  * 80% training
  * 20% validation

---

### 2. Models Used

#### 🔹 Base Models

* Ridge Regression
* Random Forest Regressor
* XGBoost Regressor

#### 🔹 Ensemble Model

* **Stacking Regressor**

  * Base learners: Ridge, RF, XGBoost
  * Meta-model: Linear Regression

---

### 3. Hybrid Approach

A hybrid prediction was created by combining:

* Global stacking model
* Cluster-based model (experimental)

```python
final_pred = 0.5 * preds_cluster + 0.5 * global_pred
```

---

## 📊 Evaluation Metrics

The following metrics were used:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

---

## 🧪 Results

### 🔹 Validation Performance

| Model            | RMSE       |
| ---------------- | ---------- |
| Baseline (Ridge) | 0.0391     |
| Stacking         | **0.0269** |
| Cluster-only     | 0.0385     |
| Hybrid           | 0.0325     |

---

### 🔬 Statistical Test

A paired t-test was conducted between stacking and hybrid models:

* **T-statistic:** -2.10
* **P-value:** 0.037

✅ The difference is statistically significant (p < 0.05)

---

## 🧠 Key Findings

* Stacking ensemble significantly improves performance over baseline models.
* Clustering-based modeling did **not** improve predictions in this setup.
* The hybrid model performed worse than stacking alone.
* Statistical testing confirms that the degradation is significant.

---

## ⚠️ Conclusion

Although clustering was expected to capture hidden structure in the data, experimental results suggest:

> Clustering does not provide additional predictive value when strong ensemble models are already used.

---

## 🔮 Future Work

* Improve clustering using:

  * Feature selection before clustering
  * Alternative methods (e.g., spectral clustering, GMM)
* Explore soft clustering approaches
* Apply SHAP for model interpretability
* Perform cross-validation and robustness analysis

---

## 🛠️ Requirements

* Python 3.x
* scikit-learn
* xgboost
* numpy
* matplotlib
* scipy

---

## 👤 Author

Alireza Deravi
Data Science | Machine Learning | Health Analytics

---

## ⭐ Notes

This project is designed as a step toward research-level machine learning and can be extended into a publishable paper with further experimentation and validation.
