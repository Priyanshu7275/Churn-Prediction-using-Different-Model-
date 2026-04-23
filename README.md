# Telco Customer Churn Prediction

A machine learning project that predicts customer churn for a telecom company using the **Telco Customer Churn** dataset from Kaggle. The project focuses heavily on **feature engineering** — creating domain-driven features from raw customer data — and compares three classification models to identify which customers are likely to leave.

---

## Dataset

- **Source:** Telco Customer Churn (Kaggle)
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Target variable:** `Churn` (Yes / No → 1 / 0)
- **Rows:** ~7,043 customers
- **Class distribution:** Imbalanced (~73% No, ~27% Yes)

---

## Project Workflow

1. Data loading and cleaning
2. Exploratory Data Analysis (EDA)
3. Feature engineering (custom domain features)
4. Feature selection (Correlation, Mutual Information, Random Forest Importance)
5. Model training and evaluation

---

## Data Cleaning

- Converted `TotalCharges` to numeric (filled missing values with 0)
- Converted `Churn` column to binary (Yes → 1, No → 0)
- Mapped `Partner`, `Dependents`, and all service-related columns (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) to 0/1

---

## Feature Engineering

Several custom features were created based on business logic and customer behavior:

| Feature | Description |
|---------|-------------|
| **tenure_group** | Tenure bucketed into groups (0-6, 6-12, 12-24, 24-48, 48+ months) |
| **ExpectedTotal** | `MonthlyCharges × tenure` (what the bill *should* be) |
| **Billing_error** | Flag (1/0) if difference between actual and expected total > $5 |
| **ServicesUsed** | Count of active add-on services per customer |
| **costPressure** | `MonthlyCharges / median cost of their InternetService type` |
| **FamilyScore** | `Partner + Dependents` (family-size proxy) |
| **EarlyChurnRisk** | `exp(-tenure / 12)` — exponential decay risk |
| **StreamingUser** | 1 if customer uses streaming TV or movies |
| **Frustration** | 1 if streaming user AND no tech support (pain indicator) |
| **PaymentRisk** | Risk score based on payment method (Electronic check = 2, Mailed check = 1, Auto = 0) |
| **LoyaltyScore** | `tenure + ServicesUsed` |
| **ContractType** | Month-to-month = 0, One year = 1, Two year = 2 |
| **FiberCostRatio** | `MonthlyCharges / average cost of their InternetService type` |

---

## Exploratory Data Analysis

Visualizations created to understand churn drivers:

- Distribution of Churn
- Churn vs Tenure Group
- Billing Error Impact on Churn
- Services Used vs Churn
- Cost Pressure vs Churn
- Family Score vs Churn
- Early Churn Risk vs Churn
- Streaming User vs Churn
- Customer Frustration vs Churn
- Monthly Charges vs Churn
- Payment Method vs Churn
- Correlation Heatmap of all numeric features

---

## Feature Selection

Used **three methods** to rank feature importance:

1. **Correlation Heatmap** — checked linear correlation with `Churn`
2. **Mutual Information Score** — captured non-linear dependencies
3. **Random Forest Feature Importance** — tree-based ranking

**Final selected features:**
```
ContractType, costPressure, EarlyChurnRisk, tenure, LoyaltyScore,
FiberCostRatio, ExpectedTotal, MonthlyCharges, PaymentRisk, TotalCharges
```

---

## Model Training

- **Train-Test Split:** 80/20, stratified on `Churn`
- **Scaling:** `StandardScaler` applied to all features
- **Models Trained:**
  - Logistic Regression
  - Support Vector Machine (SVM with RBF kernel)
  - K-Nearest Neighbors (KNN, k=5)

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **0.8006** | **0.6609** | **0.5107** | **0.5762** |
| SVM (RBF) | 0.7928 | 0.6541 | 0.4652 | 0.5438 |
| KNN (k=5) | 0.7608 | 0.5556 | 0.4947 | 0.5233 |

**Best Model: Logistic Regression** — highest accuracy and F1 score, and also the most interpretable model for business use.

---

## Tech Stack

- **Python 3**
- **pandas, numpy** — data manipulation
- **matplotlib, seaborn** — visualization
- **scikit-learn** — ML models, preprocessing, metrics

---

## How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project directory
4. Open and run `project.ipynb` cell by cell

---

## Key Takeaways

- **Feature engineering matters more than model choice** — domain-driven features like `costPressure`, `EarlyChurnRisk`, and `LoyaltyScore` carried most of the predictive signal.
- **Contract type** and **tenure-based features** are the strongest churn indicators.
- Logistic Regression outperformed SVM and KNN, showing that a simpler linear model can win when features are well-engineered.

---

## Future Improvements

- Handle class imbalance with SMOTE or class weights
- Try ensemble models (Random Forest, XGBoost, LightGBM)
- Perform hyperparameter tuning (GridSearchCV)
- Deploy as a Flask/Streamlit app for real-time predictions
