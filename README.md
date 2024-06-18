# European Football
## Overview
The project inspects European Football dataset from Kaggle. 

The primary objectives are to clean the data, perform exploratory data analasys, statistical analasys, and model linear regression on Goal predictions and Winner outcomes. 
Multicollinearity analysis, feature engineering and resampling were explored.

## Dataset
Dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/prajitdatta/ultimate-25k-matches-football-database-european).
Supplementary dataset: [Supplementary Kaggle](https://www.kaggle.com/datasets/jiezi2004/soccer?select=goal_detail.csv).

## Python Libraries

This analysis was conducted using Python 3.11. The following packages were utilized:

- duckdb=0.9.1
- matplotlib=3.7.1
- numpy=1.25.0
- pandas=1.5.3
- scipy=1.11.1
- seaborn=0.12.2
- sqlite3=3.41.2
- sklearn=0.0.post5
- statsmodels=0.14.0
- textblob=0.17.1
- unidecode=1.3.7

## Findings

* Exploratory Data Analysis (EDA): The dataset was a combination of real life events (goal counts) and FIFA game assigned team, player attributes. Not all values were available. Since data is time series, missing player attributes were filled in from previous days. Although, it was noted that player attributes change over time.
* Correlation: Player and team attributes has strong correlation dependent on similar skills. 
* Multicollinearity: Techniques such as variance inflation factor (VIF) analysis and feature aggregation were applied to tackle multicollinearity issues.
* Statistical Testing: Hypothesis testing revealed some notable differences in mean feature values across different player and team attributes.
* Models: Linear and logistic regression has been employed for Goal/Match and Goal winner predictions.

## Suggestions for Football betting

* Pay attention to Leagues - England and Spain has the highest goals per match (up to 9 goals).
* Home advantage must be taken to account.

## Future Work

- Explore the non-linear models: Decision Trees, Random Forests, or Gradient Boosting to improve predictive performance.
- Employing dimensionality reduction techniques like Principal Component Analysis (PCA) or t-Distributed Stochastic Neighbor Embedding (t-SNE) to condense the feature space and enhance interpretability.
- Address class imbalance by utilizing advanced methods, such as Synthetic Minority Over-sampling Technique (SMOTE), Adaptive Synthetic Sampling (ADASYN), or weighted loss functions within models.


## Visualization Dashboard

For a visual overview, you can visit the [Tableau Public overview](https://public.tableau.com/app/profile/gintare6386/viz/EurporeanFootball/Dashboard1?publish=yes).