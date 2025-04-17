import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

file_path = 'C:/Users/bhavy/Desktop/Minor Project Sem 5/combined_manual.xlsx'  
data = pd.read_excel(file_path, sheet_name='Sheet3')

data['precipitation'] = data.groupby('Area')['precipitation'].transform(
    lambda x: x.replace(-999, np.nan).fillna(x.mean())
)

label_encoders = {}
for column in ['Water_Table', 'urbanization']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

scaler = StandardScaler()
numerical_features = ['Elevation', 'precipitation', 'runoff_coefficient', 'drainage']
data[numerical_features] = scaler.fit_transform(data[numerical_features])


X = data[['Elevation', 'precipitation', 'runoff_coefficient', 'drainage', 'Water_Table', 'urbanization']]
y = data['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1],
}
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.5f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Classification Report:\n", classification_rep)

prob_df = pd.DataFrame(y_prob, columns=['Probability_0', 'Probability_1'])
prob_df['Actual'] = y_test.values
prob_df['Predicted'] = y_pred
prob_df.to_excel('test_probabilities_tuned.xlsx', index=False)
print("Test probabilities saved to 'test_probabilities_tuned.xlsx'.")

query_vector = {
    'Elevation': 5,
    'precipitation': 30,
    'runoff_coefficient': 0.8,
    'drainage': 25,
    'Water_Table': 'High',  
    'urbanization': 'Good'  
}

query_vector['Water_Table'] = label_encoders['Water_Table'].transform([query_vector['Water_Table']])[0]
query_vector['urbanization'] = label_encoders['urbanization'].transform([query_vector['urbanization']])[0]

query_df = pd.DataFrame([query_vector])
query_df[numerical_features] = scaler.transform(query_df[numerical_features])

query_pred = best_model.predict(query_df)
query_prob = best_model.predict_proba(query_df)

print(f"Prediction for query vector: {query_pred[0]}")
print(f"Probability of 0: {query_prob[0][0]:.5f}, Probability of 1: {query_prob[0][1]:.5f}")
