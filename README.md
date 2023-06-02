# Credit Card Approval Prediction

This project involves predicting credit card approval using machine learning techniques. The goal is to build a model that can accurately predict whether a credit card application should be approved or not. The project follows several steps, including data preprocessing, feature engineering, model selection, and evaluation.

## Installation

To run the code in this project, you need to have the following libraries installed:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- CatBoost
- LightGBM

You can install these libraries using the following command:

```shell
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost catboost lightgbm
```

## Dataset

The project uses a credit card approval dataset, which consists of information about credit card applicants. The dataset contains various features such as gender, income, education, family status, housing type, and more. The target variable is whether the credit card application was approved or not.

The dataset is loaded and explored to understand its structure and characteristics. Data preprocessing steps are performed to handle missing values and encode categorical variables. Feature engineering techniques are applied to create new features and scale the existing ones.

## Model Selection and Evaluation

Several machine learning models are used for credit card approval prediction. The selected models include:

- Logistic Regression
- Decision Tree Classifier
- XGBoost Classifier
- CatBoost Classifier
- Random Forest Classifier
- SGD Classifier
- LGBM Classifier

The models are trained on the preprocessed data using the training set. Model performance is evaluated using accuracy score and confusion matrix. The accuracy score measures the overall accuracy of the model's predictions, while the confusion matrix provides insights into the true positive, true negative, false positive, and false negative predictions.

## Usage

To run the credit card approval prediction code, follow these steps:

1. Install the required libraries mentioned in the "Installation" section.
2. Download the credit card approval dataset and place it in the appropriate directory.
3. Run the code in a Python environment, such as Jupyter Notebook or Python IDE.

Make sure to adjust the file paths in the code to match the location of the dataset on your system.

## Contributing

Contributions to this project are welcome. If you find any issues or want to add new features, feel free to open a pull request. Please ensure that your contributions align with the coding style and conventions used in the project.

