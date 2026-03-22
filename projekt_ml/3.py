import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

df = pd.read_csv('cancer.csv.csv')

drop_cols = ['diagnosis', 'id'] + [c for c in df.columns if 'Unnamed' in c]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['diagnosis'].map({'M': 1, 'B': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

print(f"RF accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"RF AUC:      {roc_auc_score(y_test, rf_prob):.4f}")
print("\n", classification_report(y_test, rf_pred))

importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 najważniejszych cech:")
print(importances.head(10))