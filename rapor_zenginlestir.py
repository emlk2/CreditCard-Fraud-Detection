import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Modeli yükle
model = joblib.load('random_forest_model.pkl')

# Özellik isimleri
features = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','Hour','Day']

# Feature importances
importances = model.feature_importances_

# Veri setini yükle
df = pd.read_csv('creditcard.csv')
df['Hour'] = (df['Time'] // 3600) % 24
df['Day'] = (df['Time'] // (3600 * 24))

# Sahte ve normal işlemler
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]

# Tablo için ortalamalar
averages = pd.DataFrame({
    'Özellik': ['Amount', 'Hour', 'Day', 'V1', 'V10', 'V12', 'V14'],
    'Normal İşlemler Ortalama': [
        normal['Amount'].mean(),
        normal['Hour'].mean(),
        normal['Day'].mean(),
        normal['V1'].mean(),
        normal['V10'].mean(),
        normal['V12'].mean(),
        normal['V14'].mean()
    ],
    'Sahte İşlemler Ortalama': [
        fraud['Amount'].mean(),
        fraud['Hour'].mean(),
        fraud['Day'].mean(),
        fraud['V1'].mean(),
        fraud['V10'].mean(),
        fraud['V12'].mean(),
        fraud['V14'].mean()
    ]
})

# Tabloyu CSV'ye kaydet
averages.to_csv('ortalamalar_tablosu.csv', index=False, encoding='utf-8-sig')

# Feature Importance Grafiği (Top 10)
top_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:10]
feat_names, feat_imps = zip(*top_features)

plt.figure(figsize=(10, 6))
sns.barplot(x=list(feat_imps), y=list(feat_names))
plt.title('En Önemli Özellikler (Feature Importance) - Sahtekarlık Tespiti')
plt.xlabel('Önem Derecesi')
plt.ylabel('Özellik')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Senaryo: Ahmet Bey örneği
# Normal kahve alış: Amount=5, Hour=8 (sabah), diğer V'ler normal dağılımdan
scenario_input = {
    'V1': normal['V1'].mean(),
    'V2': normal['V2'].mean(),
    'V3': normal['V3'].mean(),
    'V4': normal['V4'].mean(),
    'V5': normal['V5'].mean(),
    'V6': normal['V6'].mean(),
    'V7': normal['V7'].mean(),
    'V8': normal['V8'].mean(),
    'V9': normal['V9'].mean(),
    'V10': normal['V10'].mean(),
    'V11': normal['V11'].mean(),
    'V12': normal['V12'].mean(),
    'V13': normal['V13'].mean(),
    'V14': normal['V14'].mean(),
    'V15': normal['V15'].mean(),
    'V16': normal['V16'].mean(),
    'V17': normal['V17'].mean(),
    'V18': normal['V18'].mean(),
    'V19': normal['V19'].mean(),
    'V20': normal['V20'].mean(),
    'V21': normal['V21'].mean(),
    'V22': normal['V22'].mean(),
    'V23': normal['V23'].mean(),
    'V24': normal['V24'].mean(),
    'V25': normal['V25'].mean(),
    'V26': normal['V26'].mean(),
    'V27': normal['V27'].mean(),
    'V28': normal['V28'].mean(),
    'Amount': 5,  # Kahve
    'Hour': 8,    # Sabah
    'Day': 0      # Varsayalım
}

# Şüpheli işlem: Amount=500, Hour=2 (gece), V14 anormal (fraud ortalaması)
suspicious_input = scenario_input.copy()
suspicious_input['Amount'] = 500
suspicious_input['Hour'] = 2
suspicious_input['V14'] = fraud['V14'].mean()  # Anormal V14

# Tahminler
normal_pred = model.predict_proba([list(scenario_input.values())])
suspicious_pred = model.predict_proba([list(suspicious_input.values())])

# Senaryo sonuçlarını yaz
scenario_results = pd.DataFrame({
    'Senaryo': ['Normal Kahve Alışı', 'Şüpheli Yurt Dışı Harcama'],
    'Amount': [5, 500],
    'Hour': [8, 2],
    'V14': [scenario_input['V14'], suspicious_input['V14']],
    'Sahtekarlık Olasılığı (%)': [normal_pred[0][1]*100, suspicious_pred[0][1]*100]
})

scenario_results.to_csv('senaryo_ornekleri.csv', index=False, encoding='utf-8-sig')

print("Rapor için dosyalar oluşturuldu:")
print("- ortalamalar_tablosu.csv")
print("- feature_importance.png")
print("- senaryo_ornekleri.csv")