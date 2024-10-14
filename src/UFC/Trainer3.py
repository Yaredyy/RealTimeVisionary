import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# Load the dataset
df = pd.read_csv('ufc-master.csv')

# Convert Winner column to binary
df['Winner'] = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0)

# Drop non-relevant columns
X = df.drop(columns=['Winner', 'RedFighter', 'BlueFighter', 'Date', 'Location', 'Country', 'Finish', 'FinishDetails'])

# Identify categorical columns
categorical_cols = ['RedStance', 'BlueStance', 'Gender', 'WeightClass', 'BetterRank', 'FinishRoundTime']
# Convert categorical columns to category type
for col in categorical_cols:
    X[col] = X[col].astype('category')

# Preprocessing pipeline: OneHotEncoding for categorical, pass-through for numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# XGBoost model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# Define hyperparameter grid
param_grid = {
    'classifier__n_estimators': [1000, 2000],
    'classifier__learning_rate': [0.01, 0.05],
    'classifier__max_depth': [6, 8],
    'classifier__subsample': [0.8, 0.9],
    'classifier__colsample_bytree': [0.8, 0.9]
}

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, df['Winner'], test_size=0.2, random_state=42)

# Perform Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best model from Grid Search
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the best model
dump(best_model, 'ufc_predictor_model.joblib')
print("Model saved as 'ufc_predictor_model.joblib'")