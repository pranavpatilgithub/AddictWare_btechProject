import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset 
df = pd.read_csv("dataset.csv")  # Ensure correct path




# Select features and target column
features = ["FBhours", "FBA", "FPS", "FRS", "FUS", "FSS", "FSTS", "INAD", "FAD"]
X = df[features]
y = df["BFAD_CAT"] 

# # Check feature correlation with target
# print("Feature correlation with target:")
# print(df[features].corrwith(df["BFAD_CAT"]))


# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Make sure train and test are different
train_ids = set(X_train.index)
test_ids = set(X_test.index)
print(f"Overlap between train and test: {len(train_ids.intersection(test_ids))}")

# Train models
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)

dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Save models
pickle.dump(dt_model, open("decision_tree.pkl", "wb"))
pickle.dump(rf_model, open("random_forest.pkl", "wb"))

# Make predictions
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)



# Verify prediction distributions 
print("\nTrue distribution:")
print(y_test.value_counts(normalize=True))
print("\nPredicted distribution (RF):")
print(pd.Series(rf_predictions).value_counts(normalize=True))


from sklearn.model_selection import cross_val_score

print("\nPerforming cross-validation...")
dt_cv_scores = cross_val_score(dt_model, X, y, cv=5)
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)

print(f"Decision Tree CV accuracy: {dt_cv_scores.mean():.3f} ± {dt_cv_scores.std():.3f}")
print(f"Random Forest CV accuracy: {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")


# Function to calculate and display metrics
def evaluate_model(y_true, y_pred, model_name):
    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{model_name} Model Performance:")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall: {recall:.1%}")
    print(f"F1-Score: {f1:.1%}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get class-specific metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Return metrics for table creation
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'report': report,
        'confusion_matrix': cm
    }

# Evaluate both models
dt_metrics = evaluate_model(y_test, dt_predictions, "Decision Tree")
rf_metrics = evaluate_model(y_test, rf_predictions, "Random Forest")

# Create dataframes for tables
# Table 4.1: Dataset Overview
dataset_info = {
    'Parameter': ['Total Users', 'Training Data Split', 'Testing Data Split', 
                 'Features Used', 'Target Labels'],
    'Value': [len(df), f"{len(X_train)} ({len(X_train)/len(df):.1%})", 
             f"{len(X_test)} ({len(X_test)/len(df):.1%})", 
             len(features), ', '.join(sorted(y.unique()))]
}
dataset_df = pd.DataFrame(dataset_info)
print("\nTable 4.1: Dataset Overview")
print(dataset_df.to_string(index=False))

# Table 4.2: Model Performance
performance_info = {
    'Model Used': ['Decision Tree', 'Random Forest'],
    'Accuracy': [f"{dt_metrics['accuracy']:.1f}%", f"{rf_metrics['accuracy']:.1f}%"],
    'Precision': [f"{dt_metrics['precision']:.1f}%", f"{rf_metrics['precision']:.1f}%"],
    'Recall': [f"{dt_metrics['recall']:.1f}%", f"{rf_metrics['recall']:.1f}%"],
    'F1-Score': [f"{dt_metrics['f1']:.1f}%", f"{rf_metrics['f1']:.1f}%"]
}
performance_df = pd.DataFrame(performance_info)
print("\nTable 4.2: Model Performance")
print(performance_df.to_string(index=False))

# Table 4.3: Severity-level Classification Results (using Random Forest as it's better)
severity_data = []
labels = sorted(y.unique())

# Extract metrics for each severity level from the classification report
for label in labels:
    if label in rf_metrics['report']:
        severity_data.append({
            'Severity Level': label,
            'Precision': f"{rf_metrics['report'][label]['precision'] * 100:.1f}%",
            'Recall': f"{rf_metrics['report'][label]['recall'] * 100:.1f}%",
            'F1-Score': f"{rf_metrics['report'][label]['f1-score'] * 100:.1f}%"
        })

severity_df = pd.DataFrame(severity_data)
print("\nTable 4.3: Severity-level Classification Results")
print(severity_df.to_string(index=False))

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(rf_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix for Random Forest Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

# Save the results to CSV files for inclusion in reports
dataset_df.to_csv('dataset_overview.csv', index=False)
performance_df.to_csv('model_performance.csv', index=False)
severity_df.to_csv('severity_classification.csv', index=False)

print("\nEvaluation complete! Results saved to CSV files.")
