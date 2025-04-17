import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib


file_path = 'C:/Users/bhavy/Desktop/Minor Project Sem 5/combined_manual.xlsx' 
df = pd.read_excel(file_path, sheet_name='Sheet3')

df['precipitation'] = df.groupby('Area')['precipitation'].transform(
    lambda x: x.replace(-999, x.mean())
)

categorical_features = ['Water_Table', 'urbanization']
numerical_features = ['Elevation', 'precipitation', 'runoff_coefficient', 'drainage']
target = 'output'

label_encoders = {col: LabelEncoder() for col in categorical_features}
for col in categorical_features:
    df[col] = label_encoders[col].fit_transform(df[col])

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

X = df[categorical_features + numerical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200, 300],              # Number of trees
    'max_depth': [None, 10, 20, 30, 50],             # Maximum depth of trees
    'min_samples_split': [2, 5, 10],                 # Minimum samples to split
    'min_samples_leaf': [1, 2, 4],                   # Minimum samples at leaf node
    'max_features': ['auto', 'sqrt', 'log2'],        # Number of features to consider
    'bootstrap': [True, False]                       # Whether to use bootstrap samples
}
rf_classifier = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf_classifier,
    param_distributions=param_grid,
    n_iter=50,              # Number of random parameter combinations to try
    scoring='accuracy',     # Scoring metric for evaluation
    cv=5,                   # 5-fold cross-validation
    verbose=2,              # Print progress to console
    random_state=42,
    n_jobs=-1               # Use all available cores for parallel processing
)

random_search.fit(X_train, y_train)

print(f"Best Parameters: {random_search.best_params_}")
best_rf_model = random_search.best_estimator_

joblib.dump(best_rf_model, 'best_rf_model.pkl')
print("Best Random Forest model saved as 'best_rf_model.pkl'")

rf_predictions_tuned = best_rf_model.predict(X_test)
rf_probabilities_tuned = best_rf_model.predict_proba(X_test)

print("Performance Metrics for Tuned Random Forest Classifier:")
print(f"Accuracy: {accuracy_score(y_test, rf_predictions_tuned):.2f}")
print(f"Precision: {precision_score(y_test, rf_predictions_tuned):.2f}")
print(f"Recall: {recall_score(y_test, rf_predictions_tuned):.2f}")
print(f"F1-Score: {f1_score(y_test, rf_predictions_tuned):.2f}")
print("Classification Report:")
print(classification_report(y_test, rf_predictions_tuned))

test_probabilities_df = X_test.copy()
test_probabilities_df['Actual'] = y_test.values
test_probabilities_df['Probability_Class_0'] = rf_probabilities_tuned[:, 0]
test_probabilities_df['Probability_Class_1'] = rf_probabilities_tuned[:, 1]

output_file_path = 'test_probabilities_rf_tuned.xlsx'
test_probabilities_df.to_excel(output_file_path, index=False)
print(f"Test set probabilities saved to {output_file_path}")

def preprocess_query(query_vector, label_encoders, scaler, categorical_features, numerical_features):
    
    query_df = pd.DataFrame([query_vector], columns=categorical_features + numerical_features)
    
    for col in categorical_features:
        query_df[col] = label_encoders[col].transform(query_df[col])
    
    query_df[numerical_features] = scaler.transform(query_df[numerical_features])
    return query_df.values[0]


query_vector = {
    'Water_Table': 'High',       
    'urbanization': 'Good',     
    'Elevation': 5,           
    'precipitation': 30.0,       
    'runoff_coefficient': 0.8,  
    'drainage': 25              
}

processed_query = preprocess_query(query_vector, label_encoders, scaler, categorical_features, numerical_features)

query_prediction = best_rf_model.predict([processed_query])
query_probabilities = best_rf_model.predict_proba([processed_query])

print(f"Query Prediction: {query_prediction[0]}")
print(f"Query Probabilities: Class 0 = {query_probabilities[0][0]:.2f}, Class 1 = {query_probabilities[0][1]:.2f}")
