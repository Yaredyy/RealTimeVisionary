import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from joblib import dump

# Load the dataset
df = pd.read_csv('ufc-master.csv')

# Check for problematic string values in numeric columns
# Look for any non-numeric data in the numeric columns
for col in df.columns:
    if df[col].dtype == 'object':
        unique_vals = df[col].unique()
        print(f"Column {col} has unique values: {unique_vals}")

# Drop columns that are not relevant or might be causing issues
df['Winner'] = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0)
X = df.drop(columns=['Winner', 'RedFighter', 'BlueFighter', 'Date', 'Location', 'Country', 'Finish', 'FinishDetails'])

# Handle categorical columns properly
categorical_features = ['RedStance', 'BlueStance', 'Gender', 'WeightClass', 'TitleBout']

# Identify which columns are numeric
numeric_features = [col for col in X.columns if col not in categorical_features and pd.api.types.is_numeric_dtype(df[col])]

# Preprocessing pipeline: OneHotEncoding for categorical, pass-through for numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# XGBoost model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['Winner'], test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model to a file
dump(model, 'ufc_predictor_model.joblib')
print("Model saved as 'ufc_predictor_model.joblib'")