import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

df = pd.read_csv('cancer.csv.csv')

drop_cols = ['diagnosis', 'id'] + [c for c in df.columns if 'Unnamed' in c]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['diagnosis'].map({'M': 1, 'B': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train_scaled, y_train)
baseline_acc = dummy.score(X_test_scaled, y_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

print(f"Baseline Accuracy: {baseline_acc:.4f}")
print(f"Logistic Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(f"Logistic AUC:      {roc_auc_score(y_test, lr_prob):.4f}")
print("\n", classification_report(y_test, lr_pred))