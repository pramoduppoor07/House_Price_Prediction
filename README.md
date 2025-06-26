
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

## 🔧 Tech Stack & Libraries Used

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
![Plot](https://via.placeholder.com/600x300.png?text=Actual+vs+Predicted+Scatter+Plot)

---

##  How to Run This Project

1. Clone the repository:
   ```bash
   git clonehttps://github.com/yourusername/california-house-price-prediction.git  
   cd california-house-price-prediction
2. Install required packages:
   ```bash
   pip install -r requirements.txt
3. Run the Jupyter Notebook:
   ````bash
   jupyter notebook
##  Project Structure

 📄 california_house_price_prediction.ipynb # Jupyter Notebook with full ML workflow
 📄 README.md                               # Project documentation
 📄 requirements.txt                        # Python dependencies


## 📚 Future Improvements

- 🔧 Add hyperparameter tuning using [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html ) or [Optuna](https://optuna.org/ )
- 🌐 Deploy the model using [Flask](https://flask.palletsprojects.com/ ) or [Streamlit](https://streamlit.io/ ) as a web app
- 📊 Visualize feature importance for model interpretability using tools like SHAP or built-in XGBoost functions
- 🧪 Add cross-validation to improve model robustness using `cross_val_score` or `KFold`

## 🙌 Acknowledgements

- [Scikit-learn California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html )
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/ )
