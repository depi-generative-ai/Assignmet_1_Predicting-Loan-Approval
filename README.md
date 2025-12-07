# Machine Learning Project Report: Loan Eligibility Prediction

## 1. Dataset Description
The dataset used for this project is the **Loan Prediction Dataset**, containing **614 records** and **13 features**. The objective is to predict the `Loan_Status` (Yes/No) based on applicant details.

**Key Features:**
* **Demographic:** Gender, Married, Dependents, Education, Self_Employed, Property_Area.
* **Financial:** ApplicantIncome, CoapplicantIncome (Combined into `TotalIncome` during feature engineering), LoanAmount.
* **Loan Details:** Loan_Amount_Term, Credit_History.
* **Target:** `Loan_Status` (Binary Classification).

**Preprocessing Steps Taken:**
* **Data Cleaning:** The `Loan_ID` column was dropped as it provides no predictive value. An outlier with an extreme income (81,000) was identified via histograms and removed to prevent model skew.
* **Feature Engineering:** A new feature `TotalIncome` was created by summing Applicant and Co-applicant income to capture the total household financial strength.
* **Handling Missing Values:**
    * Categorical variables (e.g., Gender, Married) were imputed with the **mode** (most frequent).
    * Numerical variables (e.g., LoanAmount) were imputed with the **mean**.
* **Encoding & Scaling:** Categorical variables were One-Hot Encoded, and numerical variables were standardized using `StandardScaler` within a Scikit-Learn `Pipeline` to prevent data leakage.

## 2. Methodology
The project implemented a robust machine learning pipeline comparing linear and tree-based models.

### Models Implemented
1.  **Logistic Regression (LR):** Used as a baseline linear model. Both L1 (Lasso) and L2 (Ridge) regularization were tested to handle multicollinearity and feature selection.
2.  **Decision Tree (DT):** A non-linear baseline. Tested with both Gini Impurity and Entropy criteria.
3.  **Random Forest (RF):** An ensemble bagging method used to reduce the variance of decision trees and prevent overfitting.
4.  **AdaBoost:** A boosting ensemble method to combine weak learners (stumps) into a strong classifier.

### Feature Selection
Multiple techniques were applied to identify the most predictive features:
* **Correlation Matrix:** To identify linear relationships.
* **Feature Importance (RF):** Identified `Credit_History`, `TotalIncome`, and `LoanAmount` as the top predictors.
* **RFE (Recursive Feature Elimination) & Chi-Square:** Used to statistically rank features.
* *Decision:* All engineered features were retained as statistical tests did not strongly suggest removing any specific feature would improve performance.

### Optimization
**GridSearchCV** was utilized with 5-fold cross-validation to tune hyperparameters:
* **LR:** Tuned `C`, `penalty`, and `solver`.
* **DT:** Tuned `max_depth`, `min_samples_split`, and `criterion`.
* **RF:** Tuned `n_estimators`, `max_depth`, and `bootstrap`.
* **AdaBoost:** Tuned `n_estimators` and `learning_rate`.

## 3. Results and Analysis
The models were evaluated based on **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **AUC-ROC**.

| Model (Base) | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 76.6% | 0.74 | 0.98 | 0.84 | 0.72 |
| **Decision Tree** | 76.6% | 0.74 | 0.98 | 0.84 | 0.69 |
| **Random Forest** | 77.2% | 0.74 | 0.98 | 0.85 | 0.73 |
| **AdaBoost** | 76.6% | 0.74 | 0.98 | 0.84 | 0.70 |

*Note: Base model performance was similar across the board, likely due to the dominance of the `Credit_History` feature.*

### Hyperparameter Tuning Results (GridSearch)
After tuning, the models showed significant improvement:
* **Logistic Regression:** ~82.7%
* **Decision Tree:** **~83.5%** (Best Params: `max_depth=3`, `criterion='entropy'`)
* **Random Forest:** ~83.2%
* **AdaBoost:** **~83.5%**

## 4. Conclusion
While the base models performed similarly, hyperparameter tuning revealed that the **Decision Tree (pruned to depth 3)** and **AdaBoost** were the best performing models, achieving an accuracy of **83.5%**.

* **Why Decision Tree/AdaBoost?** The dataset contains non-linear boundaries (e.g., thresholds on Income and Credit History) that tree-based models capture effectively.
* **Effect of Regularization:** The jump in Decision Tree performance from ~70% (base) to 83.5% (tuned) highlights the massive impact of **pruning** (limiting `max_depth`). The un-regularized tree likely overfitted the training noise, while the pruned tree captured the general rule (likely heavily weighted on Credit History).
* **Final Recommendation:** Use the **Decision Tree** for deployment due to its high accuracy, interpretability, and low computational cost, or **Random Forest** if variance reduction on future unseen data is a priority.
