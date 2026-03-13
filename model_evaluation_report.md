# 🏡 House Price Prediction – Model Evaluation Report

## 📊 Dataset Overview
- **Columns Used:** Property_ID, Area, Bedrooms, Bathrooms, Age, Location, Property_Type, Price  
- **Target Variable:** Price  
- **Number of Records:** 300  
- **Missing Values:** None detected  
- **Data Types:**  
  - Numerical: Area, Bedrooms, Bathrooms, Age, Price  
  - Categorical: Location, Property_Type  

---

## ⚙️ Models Tested
1. **Linear Regression (Baseline)**  
2. **Polynomial Regression (degree=2)**  
3. **Decision Tree Regressor**  
4. **Random Forest Regressor**

---

## 📈 Evaluation Metrics

| Model                  | MAE        | MSE              | R² Score |
|-------------------------|------------|------------------|----------|
| Linear Regression       | 5,411,099.80 | 42,074,706,077,812.27 | 0.70 |
| Polynomial Regression   | –          | –                | 0.96 |
| Decision Tree Regressor | –          | –                | 0.95 |
| Random Forest Regressor | –          | –                | 0.97 |

*(MAE/MSE not shown for non-linear models since only R² was reported in your run — you can add them if you compute later.)*

---

## 📉 Visualization
- **File:** `predictions_vs_actual.png`  
- **Description:** Scatter plot comparing predicted vs actual house prices.  
- **Insight:** Predictions from Random Forest and Polynomial Regression align closely with actual values, while Linear Regression shows larger errors, especially for high-priced properties.  

---

## 🔍 Feature Importance (Random Forest)
- **Area:** 68.7%  
- **Location:** 28.3%  
- **Bedrooms:** 1.9%  
- **Age:** 0.7%  
- **Bathrooms:** 0.2%  
- **Property_Type:** 0.1%  

**Key Drivers of Price:** Area and Location dominate as the most influential predictors.

---

## 📑 Insights & Conclusion
- **Best Performing Model:** Random Forest Regressor (R² = 0.97)  
- **Key Drivers of Price:** Area and Location are the strongest predictors of house prices.  
- **Limitations:** Linear Regression underperforms due to non-linear relationships in the data.  
- **Next Steps:**  
  - Compute MAE/MSE for non-linear models for a complete comparison.  
  - Perform hyperparameter tuning for Random Forest and Decision Tree.  
  - Explore Gradient Boosting or XGBoost for further improvements.  
  - Collect more diverse data to improve generalization for extreme property values.  

