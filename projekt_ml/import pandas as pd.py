import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

base_path = os.path.dirname(os.path.abspath(__file__))
file_name = 'cancer.csv.csv'
full_path = os.path.join(base_path, file_name)

print(f"Szukam pliku w: {full_path}")

try:
    df = pd.read_csv(full_path)
    print("✅ Plik wczytany pomyślnie!")
except FileNotFoundError:
    print("❌ Błąd: Nadal nie widzę pliku. Upewnij się, że skrypt i plik csv są w tym samym folderze!")
    exit()

drop_cols = ['diagnosis', 'id'] 
drop_cols += [col for col in df.columns if 'Unnamed' in col]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("-" * 40)
print(f"Liczba cech (kolumn X): {X.shape[1]}")
print(f"Najczęstsza klasa: {y_train.value_counts().idxmax()}")
print(f"ACCURACY BASELINE: {acc:.4f}")
print("-" * 40)