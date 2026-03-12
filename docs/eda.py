import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Grafik stili
sns.set(style="whitegrid")

# Klasör oluştur
os.makedirs("figures", exist_ok=True)

# Veri setini oku
df = pd.read_csv("../data/anemi.csv")

# ----------------------------
# 1) Genel bilgiler
# ----------------------------
print("=== VERI SETI GENEL BILGILERI ===")
print("Veri seti boyutu:", df.shape)
print("\nSütun isimleri:")
print(df.columns.tolist())

print("\nİlk 5 satır:")
print(df.head())

print("\nVeri tipi bilgileri:")
print(df.info())

print("\nİstatistiksel özet:")
print(df.describe())

print("\nEksik veri sayıları:")
print(df.isnull().sum())

print("\nHedef değişken dağılımı:")
print(df["Is_Anemic"].value_counts())

# Genel bilgi tablosu oluştur
summary_table = pd.DataFrame({
    "Özellik": [
        "Veri seti adı",
        "Gözlem sayısı",
        "Değişken sayısı",
        "Eksik veri",
        "Problem türü"
    ],
    "Değer": [
        "Anemia Dataset",
        df.shape[0],
        df.shape[1],
        "Yok" if df.isnull().sum().sum() == 0 else "Var",
        "İkili sınıflandırma"
    ]
})

print("\n=== VERI SETI OZET TABLOSU ===")
print(summary_table)

summary_table.to_csv("figures/veri_seti_ozet_tablo.csv", index=False)

# ----------------------------
# 2) Sınıf dağılımı
# ----------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Is_Anemic", data=df)
plt.title("Anemi Sınıf Dağılımı")
plt.xlabel("Is_Anemic")
plt.ylabel("Kayıt Sayısı")
plt.tight_layout()
plt.savefig("figures/sekil1_sinif_dagilimi.png")
plt.show()

# ----------------------------
# 3) HGB dağılımı
# ----------------------------
plt.figure(figsize=(8, 4))
sns.histplot(df["HGB"], kde=True)
plt.title("Hemoglobin (HGB) Dağılımı")
plt.xlabel("HGB")
plt.ylabel("Frekans")
plt.tight_layout()
plt.savefig("figures/sekil2_hgb_dagilimi.png")
plt.show()

# ----------------------------
# 4) Anemi durumuna göre HGB
# ----------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(x="Is_Anemic", y="HGB", data=df)
plt.title("Anemi Durumuna Göre HGB")
plt.xlabel("Is_Anemic")
plt.ylabel("HGB")
plt.tight_layout()
plt.savefig("figures/sekil3_anemiye_gore_hgb.png")
plt.show()

# ----------------------------
# 5) RBC dağılımı
# ----------------------------
plt.figure(figsize=(8, 4))
sns.histplot(df["RBC"], kde=True)
plt.title("RBC Dağılımı")
plt.xlabel("RBC")
plt.ylabel("Frekans")
plt.tight_layout()
plt.savefig("figures/sekil4_rbc_dagilimi.png")
plt.show()

# ----------------------------
# 6) Anemi durumuna göre RBC
# ----------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(x="Is_Anemic", y="RBC", data=df)
plt.title("Anemi Durumuna Göre RBC")
plt.xlabel("Is_Anemic")
plt.ylabel("RBC")
plt.tight_layout()
plt.savefig("figures/sekil5_anemiye_gore_rbc.png")
plt.show()

# ----------------------------
# 7) Korelasyon matrisi
# ----------------------------
plt.figure(figsize=(18, 12))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Korelasyon Heatmap")
plt.tight_layout()
plt.savefig("figures/sekil6_korelasyon_heatmap.png")
plt.show()

print("\nTüm grafikler docs/figures klasörüne kaydedildi.")