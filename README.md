# Machine Learning Projects

This repository contains my implementations and experiments with different **machine learning algorithms and models**.
The goal of this repository is to better understand how algorithms work internally by implementing them manually and testing them on small datasets.

---

## Projects

### Decision Tree & Random Forest (From Scratch)

Implementation of a **Decision Tree classifier** and a **Random Forest model** built from scratch using Python and NumPy.

Main features:

* Decision Tree training using **Gini impurity**
* Support for **categorical and numerical features**
* Random Forest using **random feature subsets**
* Tree visualization using **Matplotlib**
* Evaluation metrics: Accuracy, Precision, Recall, F-Score

Location:

```
decision-tree/
```

---

### KNN Gender Classification

This project uses the **K-Nearest Neighbors (KNN)** algorithm to classify gender using physical measurements.

Features:

* Testing multiple **K values** to find the optimal model
* Train/test split evaluation
* Error analysis of misclassified samples
* Rule-based post-processing to improve predictions

Location:

```
knn-gender-classification/
```

---

## Repository Structure

```
machine-learning
│
├── decision-tree-from-scratch
│   ├── decision_tree_random_forest.py
│   ├── X_train.xlsx
│   ├── X_test.xlsx
│   ├── decision_tree.jpg
│   └── README.md
│
├── knn-gender-classification
│   ├── FindOptimalK.py
│   ├── K1Classifier.py
│   ├── bas-boy.ods
│   ├── cinsiyet.ods
│   └── README.md
│
└── README.md
```

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* scikit-learn

---

## Goals of This Repository

* Understand machine learning algorithms **from scratch**
* Practice **model evaluation and error analysis**
* Experiment with different approaches such as **rule-based improvements**
