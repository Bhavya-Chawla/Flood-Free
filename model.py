import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
# Load the dataset
file_path = 'combined_manual (5).xlsx' 
df = pd.read_excel(file_path, sheet_name='Sheet1')


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

rf_classifier = RandomForestClassifier(random_state=42, n_estimators=200, min_samples_split=5, min_samples_leaf=4,max_depth=10)
rf_classifier.fit(X_train, y_train)


rf_predictions = rf_classifier.predict(X_test)
rf_probabilities = rf_classifier.predict_proba(X_test)

print("Performance Metrics for Random Forest Classifier:")
print(f"Accuracy: {accuracy_score(y_test, rf_predictions):.2f}")
print(f"Precision: {precision_score(y_test, rf_predictions):.2f}")
print(f"Recall: {recall_score(y_test, rf_predictions):.2f}")
print(f"F1-Score: {f1_score(y_test, rf_predictions):.2f}")
print("Classification Report:")
print(classification_report(y_test, rf_predictions))

joblib.dump(rf_classifier, 'sid.pkl')
print("Random Forest model saved as 'sid.pkl'")

for col, le in label_encoders.items():
    joblib.dump(le, f'{col}_label_encoder.pkl')
    print(f"Label encoder for {col} saved as '{col}_label_encoder.pkl'")

joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")

test_probabilities_df = X_test.copy()
test_probabilities_df['Actual'] = y_test.values
test_probabilities_df['Probability_Class_0'] = rf_probabilities[:, 0]
test_probabilities_df['Probability_Class_1'] = rf_probabilities[:, 1]

output_file_path = 'test_probabilities_rf.xlsx'  
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
    'precipitation': 30,    
    'runoff_coefficient': 0.6,
    'drainage': 25              
}

processed_query = preprocess_query(query_vector, label_encoders, scaler, categorical_features, numerical_features)

query_prediction = rf_classifier.predict([processed_query])
query_probabilities = rf_classifier.predict_proba([processed_query])

print(f"Query Prediction: {query_prediction[0]}")
print(f"Query Probabilities: Class 0 = {query_probabilities[0][0]:.2f}, Class 1 = {query_probabilities[0][1]:.2f}")


importances = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print(feature_importance_df)