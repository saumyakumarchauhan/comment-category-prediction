<div align="center">

# 🧠 Comment Category Prediction Challenge

### End-to-End Multi-Class Comment Classification using NLP, Feature Engineering & Stacking Ensemble Learning

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-00C853?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-EC407A?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-6A1B9A?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blueviolet?style=for-the-badge)

![Accuracy](https://img.shields.io/badge/Accuracy-90.79%25-success?style=flat-square)
![Task](https://img.shields.io/badge/Task-Multi--Class_Classification-blue?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-Scikit--Learn-orange?style=flat-square)
![Competition](https://img.shields.io/badge/Competition-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)

</div>

<br>

## 📌 Overview

This repository presents a complete **end-to-end machine learning solution** for the **Comment Category Prediction Challenge** hosted on Kaggle.

The objective is to automatically classify user comments into one of **four predefined categories** by combining **Natural Language Processing (NLP)** with structured metadata, handcrafted feature engineering, and ensemble learning.

Unlike pipelines that rely on text alone, this project integrates several complementary signal sources:

| Signal Source | Description |
|:--|:--|
| 📝 **Text** | Word-level & character-level TF-IDF features |
| 📊 **Numerical** | Statistical and engineered numerical features |
| 👥 **Demographic** | Categorical demographic metadata |
| 🕒 **Temporal** | Time-of-day features extracted from timestamps |
| 📈 **Post-level** | Aggregated engagement features per post |
| 🤖 **Ensemble** | Stacked LightGBM + XGBoost + Logistic Regression |

> **Result:** the final stacked model achieves **90.79% accuracy**, outperforming every individual base model.

<br>

## 📑 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Dataset](#-dataset)
3. [Exploratory Data Analysis](#-exploratory-data-analysis)
4. [Feature Engineering](#-feature-engineering)
5. [Preprocessing Pipeline](#-data-preprocessing-pipeline)
6. [Machine Learning Models](#-machine-learning-models)
7. [Hyperparameter Tuning](#-hyperparameter-tuning)
8. [Stacking Ensemble Architecture](#-stacking-ensemble-architecture)
9. [Model Performance](#-model-performance)
10. [Evaluation](#-evaluation)
11. [Repository Structure](#-repository-structure)
12. [Installation](#-installation)
13. [Usage](#-usage)
14. [Future Improvements](#-future-improvements)
15. [Roadmap](#-roadmap)
16. [Author](#-author)
17. [License](#-license)

<br>

## 🎯 Problem Statement

Modern online platforms generate millions of comments every day. Automatically categorizing these comments helps improve **moderation, recommendation systems, search quality, analytics,** and overall **user experience**.

The task is framed as a **multi-class classification problem with four target classes**, predicted using:

- Comment text
- User engagement metrics (upvotes / downvotes)
- Post-level metadata
- Demographic attributes
- Internal platform-generated features

<br>

## 📂 Dataset

### Files

| File | Description |
|:--|:--|
| `train.csv` | Training dataset containing labels |
| `test.csv` | Test dataset without labels |
| `Sample.csv` | Submission template |

> **📌 Dataset Notice**
>
> The datasets used in this project are **not included** in this repository, in compliance with the Kaggle competition's data-sharing policy.
>
> Download them directly from the official competition page:
> **🔗 [kaggle.com/competitions/comment-category-prediction-challenge](https://www.kaggle.com/competitions/comment-category-prediction-challenge)**
>
> After downloading, place the following files in the project root:
>
> ```text
> train.csv
> test.csv
> Sample.csv
> ```

### Feature Columns

| Category | Columns |
|:--|:--|
| Text | Comment text |
| Engagement | Upvotes, Downvotes, 3 emoticon features |
| Demographic | Race, Religion, Gender, Disability |
| Metadata | Post ID, Created timestamp |
| Internal | Platform-generated numerical features |
| Target | Category label (0–3) |

<br>

## 📊 Exploratory Data Analysis

A thorough EDA phase preceded model development to understand class behavior, text patterns, and data quality issues.

**Key findings:**

- 🎯 Highly imbalanced target distribution
- ✏️ Most comments are relatively short in length
- 📏 Comment length varies meaningfully across classes
- 🔥 Class 2 receives the highest engagement
- ❓ Missing values are concentrated in demographic features
- ⚖️ Numerical features span different scales, requiring normalization

**Visualizations produced:**

`Target Distribution` · `Comment Length Distribution` · `Box Plot` · `Violin Plot` · `Average Upvotes` · `Confusion Matrix` · `Prediction Distribution` · `Error Analysis`

<br>

## 🛠 Feature Engineering

An extensive set of handcrafted features was built across five categories:

<table>
<tr><th width="22%">Category</th><th>Features</th></tr>
<tr><td><b>Text</b></td><td>Lowercasing · Noise removal · Word-level TF-IDF · Character-level TF-IDF</td></tr>
<tr><td><b>Numerical</b></td><td>Character count · Word count · Capital letter ratio · Vote difference · Interaction feature</td></tr>
<tr><td><b>Temporal</b></td><td>Hour of comment creation</td></tr>
<tr><td><b>Post-level</b></td><td>Average upvotes per post · Number of comments per post</td></tr>
<tr><td><b>Missing values</b></td><td>Categorical → imputed with <code>"Missing"</code> category · Text → safely imputed with empty string</td></tr>
</table>

<br>

## ⚙️ Data Preprocessing Pipeline

The full preprocessing workflow is implemented as a single, reusable **`ColumnTransformer`** pipeline combining:

- Word-level TF-IDF vectorization
- Character-level TF-IDF vectorization
- One-Hot Encoding for categorical features
- Standard Scaling for numerical features

This guarantees identical transformations are applied consistently across training, validation, and inference.

<br>

## 🤖 Machine Learning Models

Three independent base learners were trained, each contributing prediction probabilities to the final ensemble:

| Model | Role |
|:--|:--|
| **LightGBM** | Gradient-boosted base learner |
| **XGBoost** | Gradient-boosted base learner |
| **Logistic Regression** | Linear baseline **and** meta-learner |

<br>

## 🔍 Hyperparameter Tuning

`RandomizedSearchCV` with stratified cross-validation was used to tune the Logistic Regression model.

| Parameter | Optimized Value |
|:--|:--|
| `C` | `1` |

Cross-validation ensured the selected configuration generalizes well rather than overfitting a single split.

<br>

## 🏗 Stacking Ensemble Architecture

Rather than relying on a single model, predictions from all base learners are combined through a second-stage **meta-learner**:

```text
                         Raw Dataset
                              │
              ┌───────────────┴───────────────┐
              │                                │
      Feature Engineering              Data Preprocessing
              │                                │
              └───────────────┬───────────────┘
                              │
            Word + Character TF-IDF Features
                              │
           Numerical + Categorical Features
                              │
       ┌────────────────┬────────────────┬────────────────┐
       │                │                │                │
   LightGBM          XGBoost      Logistic Regression
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                              │
                Out-of-Fold Prediction Probabilities
                              │
                     Logistic Regression
                        (Meta-Learner)
                              │
                       Final Prediction
```

This two-level architecture lets the meta-learner correct for the individual biases of each base model, improving robustness and overall accuracy.

<br>

## 📈 Model Performance

### Base Model Accuracy

| Model | Accuracy |
|:--|:--:|
| LightGBM | 89.77% |
| XGBoost | 88.84% |
| Logistic Regression | 87.79% |

### Final Stacked Ensemble

| Metric | Score |
|:--|:--:|
| **Accuracy** | **90.79%** |

### Classification Report

| Class | Precision | Recall | F1-Score |
|:--:|:--:|:--:|:--:|
| 0 | 0.97 | 0.95 | 0.96 |
| 1 | 0.77 | 0.78 | 0.78 |
| 2 | 0.85 | 0.90 | 0.87 |
| 3 | 0.66 | 0.58 | 0.62 |

> Class 0 is predicted with near-perfect reliability, while class 3 — the rarest and most ambiguous category — remains the hardest to separate, consistent with the class imbalance observed during EDA.

<br>

## 📉 Evaluation

Model quality was assessed using a standard multi-class evaluation suite:

- ✅ Accuracy
- ✅ Precision (per class)
- ✅ Recall (per class)
- ✅ F1-Score (per class)
- ✅ Confusion Matrix

The stacking ensemble **consistently outperformed every individual base model** across all metrics.

<br>

## 📁 Repository Structure

```text
.
├── comment-category-prediction.ipynb   # Main notebook: EDA → Features → Models → Submission
├── README.md                           # Project documentation
├── LICENSE                             # MIT License
├── Sample.csv                          # Submission template
├── requirements.txt                    # Python dependencies
└── .gitignore
```

<br>

## 🚀 Installation

```bash
# Clone the repository
git clone <repository-url>
cd Comment-Category-Prediction

# Install dependencies
pip install -r requirements.txt
```

<br>

## ▶️ Usage

Run the notebook sequentially:

1. **Load** the dataset (`train.csv`, `test.csv`)
2. **Explore** the data via the EDA section
3. **Engineer** text, numerical, temporal, and post-level features
4. **Train** the base models and the stacking meta-learner
5. **Generate** the final submission file

<br>

## 💡 Future Improvements

- 🔤 Transformer-based embeddings (BERT / DeBERTa)
- 🎛️ Hyperparameter search via Optuna
- 🔍 Model explainability with SHAP
- ⚖️ Stronger class-imbalance handling (SMOTE, focal loss, class weighting)
- 📐 Probability calibration (Platt scaling / isotonic regression)
- 🗳️ Soft-voting ensembles as an alternative to stacking

<br>

## 🛣 Roadmap

- [x] Exploratory Data Analysis
- [x] Feature Engineering
- [x] Multiple Base ML Models
- [x] Stacking Ensemble
- [x] Submission Generation
- [ ] Transformer-based Models
- [ ] Model Explainability (SHAP)
- [ ] Deployment (API / Web App)

<br>

## 👨‍💻 Author

**Saumyakumar Chauhan**
B.Tech CSE, IIIT Kota · B.S. in Data Science, IIT Madras

If you found this repository useful, consider giving it a ⭐ — it helps a lot!

<br>

## 📜 License

This project is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for full details.

> **Note:** This repository is intended for educational, research, and portfolio purposes. Please ensure compliance with the Kaggle Competition Rules and Terms of Use regarding dataset usage and solution sharing.

<br>

<div align="center">

Made with ❤️ and a lot of coffee ☕

</div>
