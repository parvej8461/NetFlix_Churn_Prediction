# NetFlix_Churn_Prediction


This project focuses on predicting customer churn for a telecommunications company using various machine learning models. The goal is to identify customers who are likely to churn, allowing the company to take proactive measures to retain them.

## Dataset

The project utilizes the "Telco Customer Churn" dataset, which includes information about customers' demographics, services, usage, and churn status.

## Project Steps

1. **Data Cleaning and Preprocessing:**
   - Handled missing values in the `TotalCharges` column.
   - Converted categorical variables into numerical format using one-hot encoding.

2. **Exploratory Data Analysis (EDA):**
   - Explored distributions of key features like `tenure`, `MonthlyCharges`, and `TotalCharges`.
   - Visualized churn patterns based on these features.

3. **Model Building and Evaluation:**
   - Built and evaluated multiple classification models, including:
     - Logistic Regression
     - Support Vector Machines (SVC)
     - K-Nearest Neighbors (KNN)
     - Decision Tree Classifier
     - Random Forest Classifier
     - XGBoost
   - Used cross-validation to assess model performance.
   - Fine-tuned hyperparameters using GridSearchCV for SVC, Random Forest, and XGBoost.

4. **Model Selection:**
   - Compared models based on accuracy and other relevant metrics.
   - Selected the best-performing model (XGBoost after hyperparameter tuning).

5. **Feature Importance:**
   - Identified the most important features contributing to churn prediction using feature importance analysis for LightGBM and Random Forest models.

## Key Findings

- The most important features influencing churn included `MonthlyCharges`, `TotalCharges`, `tenure`, `Contract_Month-to-month`, and `OnlineSecurity_No`.
- The XGBoost model, after hyperparameter tuning, achieved the best performance with an accuracy of approximately 80%.

## How to Use

1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook `Telco_Customer_Churn_Prediction.ipynb` to explore the analysis and model building process.

## Future Improvements

- Experiment with additional feature engineering techniques.
- Explore more advanced models like neural networks.
- Deploy the model into a production environment for real-time churn prediction.

## Acknowledgments

- The dataset used in this project is available on Kaggle: [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Feel free to contribute to this project or provide feedback!
