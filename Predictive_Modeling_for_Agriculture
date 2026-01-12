# Necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# Check for missing values
crops.isna().sum()

# Confirm if "crop" is a binary or multi-label feature
crops["crop"].unique()

# Assign X to features and y to the target variable
X = crops.drop("crop", axis=1)
y = crops["crop"]

# Split data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a dictionary to store feature predictive performance
feature_performance = {}

# Train a logistic regression model for each feature
for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression(multi_class="multinomial")
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
    # Calculate f1 score
    score = metrics.f1_score(y_test, y_pred, average="weighted")
    feature_performance[feature] = score
    print(f"F1-score for {feature}: {score}")

# Assign K to represent the feature that produced the best model performance
best_predictive_feature = {"K": feature_performance["K"]}
