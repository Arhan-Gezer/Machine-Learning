# KNN Gender Classification

This project applies the **K-Nearest Neighbors (KNN)** algorithm to classify gender based on two physical measurements:

* **Head Circumference (Başçevre)**
* **Height (Boy)**

The goal is to evaluate different **K values**, analyze model errors, and improve performance using **rule-based post-processing**.

---

# Dataset

The dataset contains two files:

* `bas-boy.ods` → feature data (head circumference and height)
* `cinsiyet.ods` → gender labels

Each sample contains two features used to predict the gender class.

---

# Project Structure

```
knn-gender-classification
│
├── FindOptimalK.py
├── K1Classifier.py
├── bas-boy.ods
├── cinsiyet.ods
└── README.md
```

---

# Workflow

1. Load dataset
2. Split data into **training and test sets**
3. Test different **K values**
4. Select the best performing model
5. Analyze misclassified samples
6. Apply **rule-based corrections**
7. Compare final model performance

---

# Testing Different K Values

The script `FindOptimalK.py` tests K values from **1 to 15**.

| K  | Test Accuracy |
| -- | ------------- |
| 1  | 0.76          |
| 2  | 0.64          |
| 3  | 0.70          |
| 4  | 0.56          |
| 5  | 0.58          |
| 6  | 0.56          |
| 7  | 0.60          |
| 8  | 0.56          |
| 9  | 0.54          |
| 10 | 0.60          |
| 11 | 0.62          |
| 12 | 0.52          |
| 13 | 0.54          |
| 14 | 0.50          |
| 15 | 0.52          |

The highest accuracy was achieved with:

**K = 1 (Accuracy = 0.76)**

---

# Baseline Model (K = 1)

Baseline KNN model performance:

```
Accuracy = 0.76
```

Several samples were misclassified by the basic KNN model.

---

# Rule-Based Post-Processing

After analyzing the misclassified samples, rule-based corrections were applied using domain-inspired conditions based on:

* height ranges
* head circumference ranges

This additional step attempted to correct systematic classification errors.

---

# Performance Improvement

| Model                       | Accuracy |
| --------------------------- | -------- |
| KNN (K = 1)                 | 0.76     |
| KNN + Rule-Based Correction | **0.84** |

Accuracy improvement:

```
+0.08
```

This demonstrates that combining **machine learning with rule-based logic** can sometimes improve performance.

---

# Error Analysis

### Samples Corrected by Rule-Based Step

Some misclassified samples were corrected after applying rule-based adjustments.

### Regression Errors

A few samples that were previously correct became incorrect after applying the rules, illustrating the trade-off of rule-based post-processing.

---

# Technologies Used

* Python
* Pandas
* NumPy
* scikit-learn
* tabulate
* odfpy


