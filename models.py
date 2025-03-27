import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("dataset.csv")  # Ensure the correct dataset path

# Select features and target column
features = ["FBhours", "FBA", "FPS", "FRS", "FUS", "FSS", "FSTS", "INAD", "FAD"]  # Update as needed
X = df[features]
y = df["BFAD_CAT"]  # Change to the correct target column

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()

dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Save models
pickle.dump(dt_model, open("decision_tree.pkl", "wb"))
pickle.dump(rf_model, open("random_forest.pkl", "wb"))

print("Models trained and saved successfully!")

