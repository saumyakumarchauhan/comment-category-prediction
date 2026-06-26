<h1 align="center">🧠 Comment Category Prediction Challenge</h1>

<p align="center">
  <strong>
    End-to-End Multi-Class Comment Classification using Natural Language Processing,
    Feature Engineering, and Stacking Ensemble Learning
  </strong>
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-Learn">
  </a>
  <img src="https://img.shields.io/badge/LightGBM-Gradient_Boosting-00C853?style=for-the-badge" alt="LightGBM">
  <img src="https://img.shields.io/badge/XGBoost-Ensemble-EC407A?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/NLP-TF--IDF-6A1B9A?style=for-the-badge" alt="NLP">
  <img src="https://img.shields.io/badge/License-MIT-blueviolet?style=for-the-badge" alt="MIT License">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-90.79%25-success?style=flat-square" />
  <img src="https://img.shields.io/badge/Task-Multi--Class_Classification-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Framework-Scikit--Learn-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Competition-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white" />
</p>

---

## 📌 Overview

This repository presents a complete **end-to-end machine learning solution** for the **Comment Category Prediction Challenge** hosted on Kaggle.

The objective is to automatically classify user comments into one of **four predefined categories** by combining **Natural Language Processing (NLP)** techniques with structured metadata, handcrafted feature engineering, and ensemble learning.

Unlike traditional NLP pipelines that rely solely on text, this project integrates:

- 📝 Word-level & Character-level TF-IDF features
- 📊 Numerical and statistical features
- 👥 Categorical demographic information
- 🕒 Temporal features
- 📈 Post-level aggregated features
- 🤖 Stacking Ensemble (LightGBM + XGBoost + Logistic Regression)

The final model achieves an overall accuracy of **90.79%**, demonstrating the effectiveness of combining multiple feature types with ensemble learning.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [Data Preprocessing Pipeline](#-data-preprocessing-pipeline)
- [Machine Learning Models](#-machine-learning-models)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Stacking Ensemble](#-stacking-ensemble)
- [Model Performance](#-model-performance)
- [Evaluation](#-evaluation)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Future Improvements](#-future-improvements)
- [Roadmap](#-roadmap)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Problem Statement

Modern online platforms generate millions of comments every day. Automatically categorizing these comments helps improve moderation, recommendation systems, search quality, analytics, and user experience.

The challenge is to predict the correct category of each comment using:

- Comment text
- User engagement metrics
- Metadata
- Demographic attributes
- Internal platform-generated features

This is a **multi-class classification problem** with four target classes.

---

## 📂 Dataset

### Files

| File | Description |
|---|---|
| `train.csv` | Training dataset containing labels |
| `test.csv` | Test dataset without labels |
| `Sample.csv` | Submission template |

> **📌 Dataset Notice**
>
> The datasets used in this project are **not included** in this repository due to the Kaggle competition's data-sharing policy.
>
> You can download the datasets directly from the official Kaggle competition page:
>
> **🔗 [https://www.kaggle.com/competitions/comment-category-prediction-challenge](https://www.kaggle.com/competitions/comment-category-prediction-challenge)**
>
> After downloading, place the following files in the project root directory:
>
> ```text
> train.csv
> test.csv
> Sample.csv
> ```

### Features

- Comment text
- Created timestamp
- Upvotes & Downvotes
- Three emoticon features
- Internal numerical features
- Race
- Religion
- Gender
- Disability
- Post ID
- Target Label

---

## 📊 Exploratory Data Analysis

Several exploratory analyses were performed before model development.

### Major Findings

- Highly imbalanced target distribution
- Most comments are relatively short
- Comment length varies across different classes
- Class 2 receives the highest engagement
- Missing values exist primarily in demographic features
- Numerical features have different scales and require normalization

**Visualizations include:**
- Target Distribution
- Comment Length Distribution
- Box Plot
- Violin Plot
- Average Upvotes
- Confusion Matrix
- Prediction Distribution
- Error Analysis

---

## 🛠 Feature Engineering

The project uses extensive handcrafted features.

### Text Features
- Lowercase conversion
- Noise removal
- Word TF-IDF
- Character TF-IDF

### Numerical Features
- Character Count
- Word Count
- Capital Letter Ratio
- Vote Difference
- Interaction Feature

### Temporal Features
- Hour of Comment

### Post-Level Features
- Average Upvotes per Post
- Number of Comments per Post

### Missing Value Handling
Categorical missing values are replaced using the "Missing" category while missing comments are safely imputed.

---

## ⚙ Data Preprocessing Pipeline

The preprocessing workflow is implemented using **ColumnTransformer**.

It combines:
- Word-level TF-IDF
- Character-level TF-IDF
- One-Hot Encoding
- Standard Scaling

into a single reusable pipeline.

---

## 🤖 Machine Learning Models

Three independent models were trained:

| Model | Purpose |
|---|---|
| **Logistic Regression** | Linear baseline & Meta Learner |
| **LightGBM** | Gradient Boosting |
| **XGBoost** | Gradient Boosting |

Each model contributes prediction probabilities that are later combined through stacking.

---

## 🔍 Hyperparameter Tuning

`RandomizedSearchCV` was used to optimize Logistic Regression.

**Optimized Parameter:**
- `C = 1`

Cross-validation ensured robust model selection.

---

## 🏗 Stacking Ensemble

Instead of relying on a single model, the project combines predictions from multiple learners.

```text
                    Dataset
                       │
        ┌──────────────┴──────────────┐
        │                             │
 Feature Engineering           Data Preprocessing
        │                             │
        └──────────────┬──────────────┘
                       │
        Word + Character TF-IDF Features
                       │
       Numerical + Categorical Features
                       │
      ┌────────────┬────────────┬────────────┐
      │            │            │            │
  LightGBM      XGBoost   Logistic Regression
      │            │            │            │
      └────────────┴────────────┴────────────┘
                       │
               Prediction Probabilities
                       │
              Logistic Regression
                 (Meta Learner)
                       │
                Final Prediction
```

This ensemble improves robustness and overall prediction accuracy.

---

## 📈 Model Performance

### Base Models

| Model | Accuracy |
|---|---|
| LightGBM | 89.77% |
| XGBoost | 88.84% |
| Logistic Regression | 87.79% |

### Final Ensemble

| Metric | Score |
|---|---|
| Accuracy | 90.79% |

### Classification Report

| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| 0 | 0.97 | 0.95 | 0.96 |
| 1 | 0.77 | 0.78 | 0.78 |
| 2 | 0.85 | 0.90 | 0.87 |
| 3 | 0.66 | 0.58 | 0.62 |

---

## 📉 Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The stacking ensemble consistently outperformed each individual model.

---

## 📁 Repository Structure

```text
.
├── comment-category-prediction.ipynb
├── README.md
├── LICENSE
├── Sample.csv
├── requirements.txt
└── .gitignore
```

---

## 🚀 Installation

```bash
git clone <repository-url>
cd Comment-Category-Prediction
pip install -r requirements.txt
```

---

## ▶ Usage

Run the notebook sequentially:

1. Load Dataset
2. Perform EDA
3. Engineer Features
4. Train Models
5. Generate Submission

---

## 💡 Future Improvements

- Transformer-based embeddings
- BERT / DeBERTa models
- Optuna optimization
- SHAP Explainability
- Better imbalance handling
- Probability calibration
- Soft voting ensembles

---

## 🛣 Roadmap

- [x] Exploratory Data Analysis
- [x] Feature Engineering
- [x] Multiple ML Models
- [x] Stacking Ensemble
- [x] Submission Generation
- [ ] Transformer Models
- [ ] Model Explainability
- [ ] Deployment

---

## 👨‍💻 Author

**Saumyakumar Chauhan**
B.Tech CSE, IIIT Kota
B.S. in Data Science, IIT Madras

If you found this repository useful, consider giving it a ⭐.

---

## 📜 License

This project is licensed under the **MIT License**.
See the `LICENSE` file included in this repository for the complete license text.

> **Note:** This repository is intended for educational, research, and portfolio purposes. Please ensure compliance with the Kaggle Competition Rules and Terms of Use regarding dataset usage and solution sharing.
