import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1️⃣ Veri setini yükleme
df = pd.read_csv("../data/anemi.csv")

print("Veri seti yüklendi")
print(df.head())


# 2️⃣ Eksik veri kontrolü
print("\nEksik veriler:")
print(df.isnull().sum())

df = df.dropna()


# 3️⃣ Özellikler ve hedef değişken
X = df.drop("Is_Anemic", axis=1)
y = df["Is_Anemic"]


# 4️⃣ Train Test bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# 5️⃣ Model oluşturma
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 6️⃣ Model eğitme
model.fit(X_train, y_train)

print("\nModel eğitildi")


# 7️⃣ Tahmin
y_pred = model.predict(X_test)


# 8️⃣ Doğruluk
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Doğruluğu:", accuracy)


# 9️⃣ Model kaydetme
joblib.dump(model, "anemi_model.pkl")

print("\nModel kaydedildi -> anemi_model.pkl")