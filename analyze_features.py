import joblib
import pandas as pd

# Modeli yükle
model = joblib.load('random_forest_model.pkl')

# Özellik isimleri (notebook'tan)
features = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','Hour','Day']

# Feature importances al
importances = model.feature_importances_

print("Modelin Ana Özellikleri ve Önem Dereceleri (Sahtekarlık Tespiti İçin):")
print("=" * 60)

# Sıralayarak yazdır
for f, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
    print(f"{f}: {imp:.4f}")

print("\nEn Önemli Özellikler (Sahtekarlık Olasılığını Artıran):")
print("- Yüksek Amount değerleri")
print("- Anormal V1-V28 değerleri (PCA dönüştürülmüş)")
print("- Belirli saatler (Hour)")
print("- İşlem günü (Day)")

# Veri setini yükleyip örnek sahte işlemlerin özelliklerini göster
df = pd.read_csv('creditcard.csv')
df['Hour'] = (df['Time'] // 3600) % 24
df['Day'] = (df['Time'] // (3600 * 24))

fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]

print(f"\nSahte İşlemler Ortalama Amount: {fraud['Amount'].mean():.2f}")
print(f"Normal İşlemler Ortalama Amount: {normal['Amount'].mean():.2f}")

print(f"\nSahte İşlemler Ortalama Hour: {fraud['Hour'].mean():.2f}")
print(f"Normal İşlemler Ortalama Hour: {normal['Hour'].mean():.2f}")