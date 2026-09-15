# Wine Quality Predictive Model 🍷

This repository contains a machine learning project that performs Exploratory Data Analysis (EDA) and builds a predictive regression model to estimate wine ratings using the **Wine Reviews Dataset**.

## 📊 Project Overview
The main goal of this project is to analyze wine review data (over 130,000 records) and train a machine learning model that can predict a wine's quality score (`Points.1`) based on its price and origin traits.

## 🛠️ Machine Learning Pipeline
The project utilizes a structured `scikit-learn` pipeline that automatically handles:
* **Data Cleaning:** Dropping records with missing critical values.
* **Numerical Preprocessing:** Imputing any missing wine prices with the median.
* **Categorical Encoding:** Utilizing `OneHotEncoder` to process string features dynamically.
* **Model Engine:** Implementing a **Random Forest Regressor** to find complex mathematical patterns between a wine's profile and its final rating.

## 📈 Model Performance
After split-validating on unseen testing data (80/20 split), the predictive model achieved the following baseline diagnostics:
* **Mean Squared Error (MSE):** 5.7204 (Average prediction error is around ~2.39 points)
* **R-squared (R²) Accuracy:** 0.4480 

This means our model successfully explains **44.8%** of the mathematical variance in wine scores purely by looking at its **Price**, **Country**, and **Variety** inputs.

## 🚀 How to Run the Project
1. Clone this repository to your local machine.
2. Ensure you have the required Python libraries installed:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```
3. Replace reviews_df = pd.read_csv('C:\\...') with reviews_df = pd.read_csv(url, index_col=0). Open the Jupyter Notebook file (`.ipynb`) and run the cells. The notebook will automatically download the dataset dynamically via URL, so you don't need to manually upload or store a large 50MB CSV file.

---
*Project engineered by Stephen Adofo Kissi.*
