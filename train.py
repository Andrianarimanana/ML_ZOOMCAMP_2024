#!/usr/bin/env python
# coding: utf-8

# #### Project Aretim-po (Heart Disease Prediction)
# This repository designed a project to predict heart disease risk comparing  ML models, featuring data preprocessing, model training, evaluation, and deployment-ready insights.

# #### Data importation

import pandas as pd
import numpy as np
import os
import sys
import pickle
import joblib
import seaborn as sns
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.colors import ListedColormap

# 3. To preprocess the data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# 4. import Iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 5. Machine Learning
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score

# 6. For Classification task.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBClassifier
# from sklearn.ensemble import ExtraTreesClassifier,AdaBoostClassifier

# 7. Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 8. Ignore warnings
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('./Data/heart_disease_uci.csv')

imputer1 = IterativeImputer(max_iter=10, random_state=42)

# Fit the imputer on trestbps column
imputer1.fit(train_df[['trestbps']])

# Transform the data
train_df['trestbps'] = imputer1.transform(train_df[['trestbps']])

# Check the missing values in trestbps column
print(f"Missing values in trestbps column: {train_df['trestbps'].isnull().sum()}")


# let's see which columns has missing values
(train_df.isnull().sum()/ len(train_df)* 100).sort_values(ascending=False)


# create an object of iterative imputer 
imputer2 = IterativeImputer(max_iter=10, random_state=42)

# fit transform on ca,oldpeak, thal,chol and thalch columns
train_df['ca'] = imputer2.fit_transform(train_df[['ca']])
train_df['oldpeak']= imputer2.fit_transform(train_df[['oldpeak']])
train_df['chol'] = imputer2.fit_transform(train_df[['chol']])
train_df['thalch'] = imputer2.fit_transform(train_df[['thalch']])
# let's check again for missing values
(train_df.isnull().sum()/ len(train_df)* 100).sort_values(ascending=False)


# find missing values.
train_df.isnull().sum()[train_df.isnull().sum()>0].sort_values(ascending=False)


missing_data_cols = train_df.isnull().sum()[train_df.isnull().sum()>0].index.tolist()

missing_data_cols

# find categorical Columns
cat_cols = train_df.select_dtypes(include='object').columns.tolist()
cat_cols

# find Numerical Columns
Num_cols = train_df.select_dtypes(exclude='object').columns.tolist()
Num_cols

print(f'categorical Columns: {cat_cols}')
print(f'numerical Columns: {Num_cols}')


# FInd columns 
categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg','fbs', 'cp', 'sex', 'num']
bool_cols = ['fbs', 'exang']
numerical_cols = ['oldpeak', 'thalch', 'chol', 'trestbps', 'age']


# ##### Dealing missing Values with Machine learning model
print("Dealing missing Values with Machine learning model")
passed_col = categorical_cols
def impute_categorical_missing_data(passed_col):
    
    df_null = train_df[train_df[passed_col].isnull()]
    df_not_null = train_df[train_df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]
    
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]
    
    label_encoder = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    if passed_col in bool_cols:
        y = label_encoder.fit_transform(y)
        
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
    
    X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier()

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(x_val)

    acc_score = accuracy_score(y_val, y_pred)

    print("The feature '"+ passed_col+ "' has been imputed with", round((acc_score * 100), 2), "accuracy\n")

    X = df_null.drop(passed_col, axis=1)

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
                
    if len(df_null) > 0: 
        df_null[passed_col] = rf_classifier.predict(X)
        if passed_col in bool_cols:
            df_null[passed_col] = df_null[passed_col].map({0: False, 1: True})
        else:
            pass
    else:
        pass

    df_combined = pd.concat([df_not_null, df_null])
    
    return df_combined[passed_col]

def impute_continuous_missing_data(passed_col):
    
    df_null = train_df[train_df[passed_col].isnull()]
    df_not_null = train_df[train_df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]
    
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]
    
    label_encoder = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])
    
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
    
    X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_regressor = RandomForestRegressor()

    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(x_val)

    print("MAE =", mean_absolute_error(y_val, y_pred), "\n")
    print("RMSE =", mean_squared_error(y_val, y_pred, squared=False), "\n")
    print("R2 =", r2_score(y_val, y_pred), "\n")

    X = df_null.drop(passed_col, axis=1)

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass
                
    if len(df_null) > 0: 
        df_null[passed_col] = rf_regressor.predict(X)
    else:
        pass

    df_combined = pd.concat([df_not_null, df_null])
    
    return df_combined[passed_col]


train_df.isnull().sum().sort_values(ascending=False)

# impute missing values using our functions
for col in missing_data_cols:
    print("Missing Values", col, ":", str(round((train_df[col].isnull().sum() / len(train_df)) * 100, 2))+"%")
    if col in categorical_cols:
        train_df[col] = impute_categorical_missing_data(col)
    elif col in numeric_cols:
        train_df[col] = impute_continuous_missing_data(col)
    else:
        pass


train_df.isnull().sum().sort_values(ascending=False)


# Now, all columns are complete without any missing data.
print("Now, all columns are complete without any missing data.")

# Remove the column because it is an outlier because trestbps cannot be zero.
train_df=train_df[train_df['trestbps']!=0]
train_df=train_df[train_df['oldpeak'] >=-1]

# #### Prepare Training data
print("Prepare Training data")
# Prepare Training data
feature_cols = ['thal',
'slope',
'fbs',
'exang',
'restecg',
'id',
'age',
'sex',
'dataset',
'cp',
'trestbps',
'chol',
'thalch',
'oldpeak',
'ca'
]
X = train_df[feature_cols]
y = train_df['num']

# Encode the categorical columns

Label_Encoder = LabelEncoder()

for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'category':
        X[col] = Label_Encoder.fit_transform(X[col])
    else:
        pass


# #### Split training and validation sets
print("Split training and validation sets")
x_full_train, x_test,y_full_train, y_test= train_test_split(X, y, test_size=0.2, random_state=11)
x_train, x_val, y_train, y_val= train_test_split(x_full_train,y_full_train, test_size=0.25, random_state=11)

# #### Train the model
print("Selecting the final model ")

models = {
    'XGBoost': XGBClassifier()
}

param_grids = {
    'XGBoost': {
        'learning_rate': 0.075,
        'max_depth': 5,
        'n_estimators':50
    }

}
x_train=x_full_train
y_train=y_full_train
x_val=x_test

y_val=y_test
param_grids = {
    'learning_rate': 0.075,
    'max_depth': 4,
    'n_estimators': 50
}

model = XGBClassifier(**param_grids)

# Entraînement du modèle
model.fit(x_train, y_train)

# Évaluation du modèle
y_pred = model.predict(x_val)
# test_accuracy = accuracy_score(y_val, y_pred)
# print(f"Accuracy: {test_accuracy:.2f}")
model_name="XGBClassifier"
test_accuracy = accuracy_score(y_val, y_pred)
print(f"Test Accuracy for {model_name}: {test_accuracy}\n")
# Classification Report
print(f"Classification Report for {model_name}:\n")
print(classification_report(y_val, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_pred)
print(f"Confusion Matrix for {model_name}:\n{conf_matrix}")
# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
class_labels = sorted(set(y_val)) 
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'Confusion Matrix for {model_name}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# uncomment the line below if you want to save the plot
# plt.show()


# ##### Save the model
print("Save the model") 
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "model", "model_XGBClassifier.pkl")
# Saving the model with pickle
with open('./model/model_XGBClassifier.pkl', 'wb') as f:
    joblib.dump(model, f)
    print('Best model saved!')
# Saving the model with pickle
with open('model_XGBClassifier.bin', 'wb') as file:
    pickle.dump((model), file)


print("Modele saved in model_XGBClassifier.bin .... done")