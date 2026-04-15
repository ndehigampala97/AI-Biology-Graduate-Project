# AI-Biology-Graduate-Project
Maize Grain Yield Prediction using Machine Learning

Overview
This project aims to predict maize grain yield (bu/A) using the 2023 Genomes to Fields (G2F) phenotypic dataset. The dataset contains genotype, field, and phenotypic measurements collected across multiple environments.

The goal is to evaluate how well machine learning models can predict yield based on these features and identify which model performs best.

Research Question
Can maize grain yield be accurately predicted using genotype, field, and phenotypic data?

Dataset
File: g2f_2023_phenotypic_clean_data.csv
Source: Genomes to Fields (G2F) Initiative
Rows: ~18,888
Target variable: Grain Yield (bu/A)

Data Summary
Mean yield: ~160 bu/A
Standard deviation: ~55.6 bu/A
Range: ~7 to 300 bu/A

Workflow

1. Data Preprocessing
Removed rows with missing target values
Dropped non-informative columns (IDs, comments, etc.)
Handled missing values:
Numerical → median imputation
Categorical → most frequent value
Encoded categorical variables using One-Hot Encoding
Standardized numerical features for linear models
2. Models Used
Ridge Regression
Linear model with regularization
Helps reduce overfitting
Random Forest Regressor
Ensemble model using decision trees
Captures nonlinear relationships and feature interactions
3. Hyperparameter Tuning
Used GridSearchCV to find optimal parameters:
Ridge:
alpha ∈ {0.1, 1, 10, 50, 100}
Random Forest:
n_estimators ∈ {100, 200}
max_depth ∈ {None, 10, 20}
min_samples_split ∈ {2, 5}