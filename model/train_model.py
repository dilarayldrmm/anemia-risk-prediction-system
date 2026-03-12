import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1️⃣ Veri setini yükleme
df = pd.read_csv("../data/anemi.csv")

print("Veri seti yüklendi")
print(df.head())


# 2️⃣ Eksik veri kontrolü
print("\nEksik veriler:")
print(df.isnull().sum())

df = df.dropna()


# 3️⃣ Data leakage sütunlarını kaldır
drop_columns = [
    "All_Class",
    "HGB_Anemia_Class",
    "Iron_anemia_Class",
    "Folate_anemia_class",
    "B12_Anemia_class"
]

df = df.drop(columns=drop_columns)


# 4️⃣ Özellikler ve hedef değişken
X = df.drop("Is_Anemic", axis=1)
y = df["Is_Anemic"]


# 5️⃣ Train Test bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 6️⃣ Model oluşturma
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 7️⃣ Model eğitme
model.fit(X_train, y_train)

print("\nModel eğitildi")


# 8️⃣ Tahmin
y_pred = model.predict(X_test)


# 9️⃣ Doğruluk
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Doğruluğu:", accuracy)


# 10️⃣ Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 11️⃣ Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 12️⃣ Model kaydetme
joblib.dump(model, "anemi_model.pkl")

print("\nModel kaydedildi -> model/anemi_model.pkl")


import matplotlib.pyplot as plt
import seaborn as sns

# Feature importance
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importances)

plt.figure(figsize=(10, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importances)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()