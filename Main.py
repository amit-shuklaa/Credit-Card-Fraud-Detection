import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("creditcard.csv")

# Features & Target
X = df.drop(columns=["Class"])
y = df["Class"]

# Handle Imbalance using ADASYN
adasyn = ADASYN(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model using XGBoost
model = XGBClassifier(scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]), random_state=42)
model.fit(X_train, y_train)

# Save Model & Scaler
joblib.dump(model, "fraud_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Evaluate Model with Lower Threshold
y_prob = model.predict_proba(X_test)[:, 1]  # Get fraud probability
threshold = 0.4  # Reduce from default 0.5
y_pred = (y_prob >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

print("Model trained and saved successfully!")
