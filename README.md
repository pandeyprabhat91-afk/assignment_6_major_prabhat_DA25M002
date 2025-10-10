#  DA5401 Assignment 6 – Imputation via Regression for Missing Data  
**Course:** DA5401 – Data Analytics  
**Student:** Major Prabhat Pandey  
**Roll No:** DA25M002  
**Program:** M.Tech in Artificial Intelligence and Data Science  
**Development Period:** October 9–11, 2025  

---

##  Overview

This repository presents the complete implementation of **Assignment 6: Imputation via Regression**, focusing on handling missing data using both **linear** and **non-linear regression** imputation strategies.  

The project explores how the choice of imputation technique affects **downstream classification performance** using the **UCI Credit Card Default Clients Dataset**.  
The workflow emphasizes practical data science practices—ranging from **data preprocessing** to **statistical validation** and **comparative model analysis**.

---

##  Objectives

- Introduce **Missing At Random (MAR)** values in a clean dataset to simulate real-world incompleteness.  
- Apply and compare three imputation strategies:
  1. **Simple Median Imputation** (Baseline)
  2. **Linear Regression Imputation**
  3. **Non-Linear Regression Imputation (KNN)**  
- Evaluate performance using a **Logistic Regression classifier**.  
- Compare results against a **Listwise Deletion** baseline.  
- Analyze trade-offs between accuracy, F1-score, and data retention.  

---

##  Dataset Overview

**Source:** [UCI Credit Card Default Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)  
- **Observations:** 30,000  
- **Features:** 24 (Demographics, payment history, billing amounts)  
- **Target:** `default.payment.next.month` (binary classification)  

Artificial missingness was introduced into:  
- `AGE` → 8% missing  
- `BILL_AMT1` → 7% missing  
- `BILL_AMT2` → 6% missing  

This created realistic incomplete data while maintaining dataset structure.

---

---

##  Implementation Summary

###  Part A – Data Preprocessing & Imputation
| Strategy | Method | Description |
|-----------|---------|-------------|
| **A** | Median Imputation | Replaced missing values with column medians. Robust against outliers. |
| **B** | Linear Regression | Predicted missing `AGE` using linear regression on other numeric features. |
| **C** | Non-Linear Regression (KNN) | Used K-Nearest Neighbors regression for flexible, non-linear prediction. |
| **D** | Listwise Deletion | Removed rows containing any missing values for comparison. |

> **Observation:** KNN handled complex credit patterns better, while linear regression struggled with weak correlations.

---

###  Part B – Model Training & Performance Evaluation
All four datasets (A–D) were standardized using `StandardScaler` and used to train **Logistic Regression** classifiers.  

| Dataset | Method | Accuracy | Precision | Recall | F1-Score |
|----------|---------|-----------|------------|----------|-----------|
| A | Median Imputation | 0.8075 | 0.6870 | 0.2381 | 0.3537 |
| B | Linear Regression | 0.8072 | 0.6848 | 0.2374 | 0.3525 |
| C | KNN Regression | 0.8077 | 0.6860 | 0.2404 | 0.3560 |
| D | Listwise Deletion | **0.8117** | **0.7293** | 0.2393 | **0.3603** |

**Primary Metric:** F1-Score (to account for class imbalance of ~3.5:1)

---

###  Part C – Comparative & Critical Analysis

#### Key Insights:
- **Listwise Deletion** surprisingly outperformed all imputation methods by ~1.8% in F1-score, despite losing 19.5% of data.
- **KNN Imputation** was the best-performing regression-based approach.
- **Linear Regression Imputation** provided stable but less flexible estimates due to linear assumptions.

#### Quantitative Ranking (F1-Score):
1. Complete Case Analysis (Listwise Deletion): **0.3603**
2. KNN Imputation (Non-linear): **0.3560**
3. Median Imputation: **0.3537**
4. Linear Regression Imputation: **0.3525**

#### Why Listwise Deletion May Excel:
- Removes uncertainty from imputed estimates  
- Maintains authentic variable relationships  
- Reduces noise in training data  

#### Why Imputation May Underperform:
- Introduces estimation error and artificial variance  
- Slightly distorts feature-target dependencies  

---

###  Advanced Statistical Validation

To ensure robustness, results were tested under **multiple random seeds** (42, 123, 456, 789, 999).  
The F1-score difference across methods averaged below **2%**, suggesting statistical insignificance.

| Method | Avg F1 ± Std Dev |
|---------|------------------|
| Median Imputation | 0.3583 ± 0.0078 |
| Linear Regression | 0.3577 ± 0.0074 |
| KNN Imputation | 0.3579 ± 0.0074 |
| Listwise Deletion | 0.3665 ± 0.0267 |

**Conclusion:**  
> Listwise deletion’s advantage is not strongly robust—its lead may fall within natural variance bounds.

---

### Key Learnings & Reflections

Regression-based imputation, while sophisticated, can introduce estimation noise.

Simpler strategies (median or deletion) may outperform when missingness is low and not systematically biased.

The F1-score is a better performance indicator than accuracy for imbalanced credit datasets.

Practical trade-offs between data retention and model reliability are more valuable than raw metrics.



