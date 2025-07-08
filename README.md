
#  California House Price Prediction using XGBoost

This project builds a regression model using the **California Housing Dataset** to predict **median house prices** based on features like income, house age, rooms, and location. It uses **XGBoost**, a powerful and efficient gradient boosting algorithm.

---

##  Project Overview

- This is a supervised machine learning project for **regression analysis**.
- The model is trained using the **XGBoost Regressor**.
- Dataset: `fetch_california_housing()` from `sklearn.datasets`
- Evaluation metrics: **R² Score**, **Mean Absolute Error (MAE)**
- Visual comparison between **actual** and **predicted** house prices included.

---

##  Dataset Description

The California Housing dataset contains 20,640 instances and 8 features:

| Feature      | Description                                      |
|--------------|--------------------------------------------------|
| MedInc       | Median income in block group                    |
| HouseAge     | Median house age in block group                 |
| AveRooms     | Average number of rooms per household           |
| AveBedrms    | Average number of bedrooms per household        |
| Population   | Block group population                          |
| AveOccup     | Average number of household members             |
| Latitude     | Block group latitude                            |
| Longitude    | Block group longitude                           |

**Target variable**: `MedHouseVal` (Median house value in $100,000s)

---

##  Tech Stack & Libraries Used

- **Python**
- **Pandas**, **NumPy** – Data manipulation
- **Matplotlib**, **Seaborn** – Visualization
- **Scikit-learn** – Dataset loading, model evaluation
- **XGBoost** – Model training

---

##  Model Training

- The dataset was split into training (80%) and test (20%) sets.
- The model used: `XGBRegressor()`
- Evaluation results:
  - **Training R² Score**: `0.9436`
  - **Training MAE**: `0.193`
  - **Test R² Score**: `0.8338`
  - **Test MAE**: `0.310`

---



##  Visual Output

**Actual Prices vs Predicted Prices (Training Set)**  
[View the plot here](https://raw.githubusercontent.com/pramoduppoor07/House_Price_Prediction/main/actual_vs_predicted.png)

---

##  How to Run This Project

1. Clone the repository:
   ```bash
   git clone https://github.com/pramoduppoor07/House_Price_Prediction.git  
   cd House_Price_Prediction

2. Run the Jupyter Notebook:
   ````bash
   jupyter notebook

---

##  Future Improvements

-  Add hyperparameter tuning using [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html ) or [Optuna](https://optuna.org/ )
-  Deploy the model using [Flask](https://flask.palletsprojects.com/ ) or [Streamlit](https://streamlit.io/ ) as a web app
-  Visualize feature importance for model interpretability using tools like SHAP or built-in XGBoost functions
-  Add cross-validation to improve model robustness using `cross_val_score` or `KFold`
  
---

##  Acknowledgements

- [Scikit-learn California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html )
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/ )
