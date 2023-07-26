import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle
import streamlit as st

# Load the dataset
data = pd.read_csv('copper_data.csv')

# Remove rows with '00000' in 'Material_Reference'
data['Material_Reference'] = data['Material_Reference'].replace('00000', np.nan)

# Drop irrelevant columns
data = data.drop(columns=['INDEX'])

# Encode categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=['Material_Reference', 'Sales_Person'])

# Handling missing values
data.fillna(data.mean(), inplace=True)

# Split the data into features (X) and target (y) for Regression
X_reg = data.drop(columns=['Selling_Price'])
y_reg = data['Selling_Price']

# Split the data into features (X) and target (y) for Classification
X_cls = data.drop(columns=['Status'])
y_cls = data['Status']

# Feature Scaling
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

scaler_cls = StandardScaler()
X_cls_scaled = scaler_cls.fit_transform(X_cls)

# Split data into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls_scaled, y_cls, test_size=0.2, random_state=42)

# Exploratory Data Analysis
# Visualize skewness before and after transformation using seaborn's distplot
sns.distplot(y_reg, label='Before Transformation')
sns.distplot(np.log1p(y_reg), label='After Transformation')
plt.legend()
plt.xlabel('Selling Price')
plt.title('Skewness of Selling Price')
plt.show()

# Visualize outliers using seaborn's boxplot
sns.boxplot(y_reg)
plt.xlabel('Selling Price')
plt.title('Outliers in Selling Price')
plt.show()

# Model Building - Regression
regressor = ExtraTreesRegressor(random_state=42)
regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = regressor.predict(X_test_reg)
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Mean Squared Error (Regression): {mse_reg}")

# Model Building - Classification
classifier = ExtraTreesClassifier(random_state=42)
classifier.fit(X_train_cls, y_train_cls)
y_pred_cls = classifier.predict(X_test_cls)
accuracy_cls = accuracy_score(y_test_cls, y_pred_cls)
print(f"Accuracy (Classification): {accuracy_cls}")

# Model Evaluation
# You can evaluate the model further using other metrics and techniques.

# Save the models and preprocessors
with open('regressor.pkl', 'wb') as f:
    pickle.dump(regressor, f)

with open('classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

with open('scaler_reg.pkl', 'wb') as f:
    pickle.dump(scaler_reg, f)

with open('scaler_cls.pkl', 'wb') as f:
    pickle.dump(scaler_cls, f)

# Streamlit Web Application
def main():
    st.title("Copper Industry Predictions")
    task = st.sidebar.selectbox("Choose a task", ["Regression", "Classification"])

    if task == "Regression":
        st.subheader("Predict Selling Price")
        st.write("Enter the values for each column except 'Selling_Price' to predict the Selling Price.")
        # Input fields for each column except 'Selling_Price'
        # You can add input fields for each feature in X_reg and obtain predictions using the regressor model.
        # Use the loaded scaler_reg to preprocess the input data before prediction.

    elif task == "Classification":
        st.subheader("Lead Classification")
        st.write("Enter the values for each column except 'Status' to predict the lead status (Won/Lost).")
        # Input fields for each column except 'Status'
        # You can add input fields for each feature in X_cls and obtain predictions using the classifier model.
        # Use the loaded scaler_cls to preprocess the input data before prediction.

if __name__ == '__main__':
    main()
