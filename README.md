# 🧠 Gaussian Naive Bayes from Scratch

This repository presents a **Gaussian Naive Bayes classifier implemented from scratch** using NumPy. The project focuses on understanding the core mathematics and probabilistic reasoning behind Naive Bayes, without relying on high-level machine learning libraries.

---

## ✨ Features
- 🔢 Pure NumPy implementation of Gaussian Naive Bayes  
- 📊 Manual computation of class **priors**, **means**, and **variances**  
- 📐 Gaussian likelihood estimation for continuous features  
- 🧮 Log-posterior probability calculation for numerical stability  
- ⚡ Performance and accuracy comparison with `scikit-learn`  
- 📈 Execution time benchmarking  

---

## 📁 Dataset
A synthetic binary classification dataset is generated using:

- `sklearn.datasets.make_classification`
- **10,000 samples**
- **20 features**
- **2 classes**

The dataset is split into **70% training** and **30% testing** sets.

---

## 🛠️ Implementation Details
The classifier follows these steps:

1. Compute class-wise **mean** and **variance**
2. Estimate **prior probabilities** for each class
3. Apply the **Gaussian Probability Density Function**
4. Use **log-probabilities** to prevent numerical underflow
5. Predict the class with the **maximum posterior probability**

---

## 📊 Evaluation
The custom model is evaluated against **scikit-learn’s `GaussianNB`**, comparing:

- ✅ Classification Accuracy  
- ⏱️ Execution Time  

This highlights the trade-off between **model transparency** and **library-level optimization**.

---

## 📦 Requirements
- Python 3.x  
- NumPy  
- scikit-learn  

---

## ▶️ Usage
Install dependencies and run the script:

```bash
py
