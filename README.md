# 🧠 Comment Category Prediction Challenge

```{=html}
<p align="center">
```
`<strong>`{=html}End-to-End Multi-Class Comment Classification using
Natural Language Processing, Feature Engineering, and Stacking Ensemble
Learning`</strong>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-green?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

```{=html}
</p>
```

------------------------------------------------------------------------

## 📌 Overview

This repository contains a complete end-to-end machine learning solution
developed for the **Comment Category Prediction Challenge**. The
objective is to automatically classify user comments into one of four
predefined categories by combining textual information with metadata and
user interaction features.

Unlike a traditional NLP project that only relies on text, this solution
integrates **textual, numerical, categorical, temporal, and post-level
features** into a unified preprocessing pipeline. The final prediction
is generated using a **Stacking Ensemble** consisting of LightGBM,
XGBoost, and Logistic Regression.

------------------------------------------------------------------------

# 📑 Table of Contents

-   Overview
-   Problem Statement
-   Dataset
-   Exploratory Data Analysis
-   Feature Engineering
-   Data Preprocessing
-   Models
-   Hyperparameter Tuning
-   Stacking Ensemble
-   Performance
-   Repository Structure
-   Installation
-   Usage
-   Future Improvements
-   Author
-   License

------------------------------------------------------------------------

# 🎯 Problem Statement

Modern online platforms generate millions of comments every day.
Automatically categorizing these comments helps improve moderation,
recommendation systems, search quality, analytics, and user experience.

The challenge is to predict the correct category of each comment using:

-   Comment text
-   User engagement metrics
-   Metadata
-   Demographic attributes
-   Internal platform-generated features

This is a **multi-class classification problem** with four target
classes.

------------------------------------------------------------------------

# 📂 Dataset

## Files

  File         Description
  ------------ ------------------------------------
  train.csv    Training dataset containing labels
  test.csv     Test dataset without labels
  Sample.csv   Submission template

### Features

-   Comment text
-   Created timestamp
-   Upvotes & Downvotes
-   Three emoticon features
-   Internal numerical features
-   Race
-   Religion
-   Gender
-   Disability
-   Post ID
-   Target Label

------------------------------------------------------------------------

# 📊 Exploratory Data Analysis

Several exploratory analyses were performed before model development.

### Major Findings

-   Highly imbalanced target distribution
-   Most comments are relatively short
-   Comment length varies across different classes
-   Class 2 receives the highest engagement
-   Missing values exist primarily in demographic features
-   Numerical features have different scales and require normalization

Visualizations include:

-   Target Distribution
-   Comment Length Distribution
-   Box Plot
-   Violin Plot
-   Average Upvotes
-   Confusion Matrix
-   Prediction Distribution
-   Error Analysis

------------------------------------------------------------------------

# 🛠 Feature Engineering

The project uses extensive handcrafted features.

## Text Features

-   Lowercase conversion
-   Noise removal
-   Word TF-IDF
-   Character TF-IDF

## Numerical Features

-   Character Count
-   Word Count
-   Capital Letter Ratio
-   Vote Difference
-   Interaction Feature

## Temporal Features

-   Hour of Comment

## Post-Level Features

-   Average Upvotes per Post
-   Number of Comments per Post

## Missing Value Handling

Categorical missing values are replaced using the "Missing" category
while missing comments are safely imputed.

------------------------------------------------------------------------

# ⚙ Data Preprocessing Pipeline

The preprocessing workflow is implemented using **ColumnTransformer**.

It combines

-   Word-level TF-IDF
-   Character-level TF-IDF
-   One-Hot Encoding
-   Standard Scaling

into a single reusable pipeline.

------------------------------------------------------------------------

# 🤖 Machine Learning Models

Three independent models were trained.

  Model                 Purpose
  --------------------- --------------------------------
  Logistic Regression   Linear baseline & Meta Learner
  LightGBM              Gradient Boosting
  XGBoost               Gradient Boosting

Each model contributes prediction probabilities that are later combined
through stacking.

------------------------------------------------------------------------

# 🔍 Hyperparameter Tuning

RandomizedSearchCV was used to optimize Logistic Regression.

Optimized Parameter

-   C = 1

Cross-validation ensured robust model selection.

------------------------------------------------------------------------

# 🏗 Stacking Ensemble

Instead of relying on a single model, the project combines predictions
from multiple learners.

``` text
                    Dataset
                       │
        ┌──────────────┴──────────────┐
        │                             │
 Feature Engineering          Data Preprocessing
        │                             │
        └──────────────┬──────────────┘
                       │
        Word + Character TF-IDF Features
                       │
       Numerical + Categorical Features
                       │
      ┌────────────┬────────────┬────────────┐
      │            │            │
 LightGBM      XGBoost   Logistic Regression
      │            │            │
      └────────────┴────────────┘
               Prediction Probabilities
                       │
              Logistic Regression
                 (Meta Learner)
                       │
               Final Prediction
```

This ensemble improves robustness and overall prediction accuracy.

------------------------------------------------------------------------

# 📈 Model Performance

## Base Models

  Model                 Accuracy
  --------------------- ------------
  LightGBM              **89.77%**
  XGBoost               **88.84%**
  Logistic Regression   **87.79%**

## Final Ensemble

  Metric            Score
  ---------- ------------
  Accuracy     **90.79%**

### Classification Report

  Class     Precision   Recall   F1 Score
  ------- ----------- -------- ----------
  0              0.97     0.95       0.96
  1              0.77     0.78       0.78
  2              0.85     0.90       0.87
  3              0.66     0.58       0.62

------------------------------------------------------------------------

# 📉 Evaluation

The model is evaluated using

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   Confusion Matrix

The stacking ensemble consistently outperformed each individual model.

------------------------------------------------------------------------

# 📁 Repository Structure

``` text
.
├── Comment_Category_Prediction.ipynb
├── train.csv
├── test.csv
├── Sample.csv
├── submission.csv
├── README.md
└── requirements.txt
```

------------------------------------------------------------------------

# 🚀 Installation

``` bash
git clone <repository-url>

cd Comment-Category-Prediction

pip install -r requirements.txt
```

------------------------------------------------------------------------

# ▶ Usage

Run the notebook sequentially.

1.  Load Dataset
2.  Perform EDA
3.  Engineer Features
4.  Train Models
5.  Generate Submission

------------------------------------------------------------------------

# 💡 Future Improvements

-   Transformer-based embeddings
-   BERT / DeBERTa models
-   Optuna optimization
-   SHAP Explainability
-   Better imbalance handling
-   Probability calibration
-   Soft voting ensembles

------------------------------------------------------------------------

# 🛣 Roadmap

-   [x] Exploratory Data Analysis
-   [x] Feature Engineering
-   [x] Multiple ML Models
-   [x] Stacking Ensemble
-   [x] Submission Generation
-   [ ] Transformer Models
-   [ ] Model Explainability
-   [ ] Deployment

------------------------------------------------------------------------

# 👨‍💻 Author

**Saumya Kumar**

B.Tech CSE, IIIT Kota\
B.S. in Data Science, IIT Madras

If you found this repository useful, consider giving it a ⭐.

------------------------------------------------------------------------

# 📜 License

MIT License

This project is intended for educational, research, and portfolio
purposes.
