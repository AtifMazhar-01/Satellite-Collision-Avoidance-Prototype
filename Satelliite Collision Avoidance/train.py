import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load preprocessed data
data = pd.read_csv("satellite_data.csv")

# 2. Features & target
X = data[["Altitude_km", "Inclination", "Eccentricity"]]
y = data["Collision_risk"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Save model
with open("collision_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("🚀 Model saved as collision_model.pkl")

