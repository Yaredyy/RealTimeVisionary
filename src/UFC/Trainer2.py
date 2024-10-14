import pandas as pd
from sklearn.model_selection import train_test_split
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
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # Handle unknown categories
    ]
)

# XGBoost model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, df['Winner'], test_size=0.2, random_state=42)

# Fit the pipeline
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the model
dump(model, 'ufc_predictor_model.joblib')
print("Model saved as 'ufc_predictor_model.joblib'")