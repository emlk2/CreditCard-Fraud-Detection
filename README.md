# Kredi Kartı Sahtekarlık Tespiti ve Kişiye Özel Risk Skorlama Sistemi

Bu proje, kredi kartı işlemlerinde sahtekarlık ihtimalini tespit etmek amacıyla geliştirilmiş bir makine öğrenmesi ve API tabanlı risk analiz sistemidir.

Projenin ilk aşamasında Kaggle üzerindeki **Credit Card Fraud Detection** veri seti kullanılarak Random Forest algoritması ile sahtekarlık tahmin modeli eğitilmiştir. Daha sonra bu model FastAPI ile servis haline getirilmiş ve kullanıcıdan gelen işlem verilerine göre anlık tahmin yapabilecek hale getirilmiştir.

Projenin geliştirilmiş versiyonunda ise sadece genel model tahminiyle yetinilmemiş, kullanıcıya özel risk skorlama motoru eklenmiştir. Böylece aynı işlem, farklı kullanıcı profillerine göre farklı risk seviyelerinde değerlendirilebilmektedir.

---

## Projenin Amacı

Finansal işlemlerdeki olağandışı hareketleri tespit ederek sahtekarlık riskini azaltmak amaçlanmıştır.

Bu sistem:

- Kredi kartı işlemlerini makine öğrenmesi modeliyle analiz eder.
- İşlemin sahtekarlık olasılığını hesaplar.
- Kullanıcının kişisel harcama alışkanlıklarını dikkate alır.
- Risk seviyesine göre işlem onayı, ek doğrulama veya geçici durdurma önerisi üretir.

---

## Kullanılan Veri Seti

Projede Kaggle üzerinde yer alan **Credit Card Fraud Detection** veri seti kullanılmıştır.

Veri setinde:

- `Time`: İşlemin zaman bilgisi
- `Amount`: İşlem tutarı
- `V1` - `V28`: PCA dönüşümü uygulanmış anonim özellikler
- `Class`: Hedef değişken

bulunmaktadır.

`Class` sütununda:

- `0`: Normal işlem
- `1`: Sahte işlem

anlamına gelmektedir.

---

## V1 - V28 Sütunları Nedir?

Veri setindeki `V1`, `V2`, ..., `V28` sütunları gerçek ham değişkenler değildir. Bu sütunlar, gizlilik nedeniyle PCA yöntemiyle dönüştürülmüş sayısal bileşenlerdir.

Bu nedenle `V14 kesin olarak konumdur` veya `V10 IP adresidir` gibi bir yorum yapılamaz. Bu değişkenler, orijinal işlem özelliklerinin matematiksel dönüşüm sonucu elde edilmiş anonim halleridir.

Modelleme açısından bu değişkenlerin sahtekarlık tespitinde ne kadar etkili olduğu feature importance yöntemiyle analiz edilebilir.

---

## Kullanılan Teknolojiler

- Python
- Pandas
- Scikit-learn
- Imbalanced-learn / SMOTE
- Random Forest Classifier
- FastAPI
- Pydantic
- Joblib
- Swagger UI
- Git / GitHub

---

## Proje Akışı

```text
Veri setinin yüklenmesi
        ↓
Sınıf dağılımının incelenmesi
        ↓
X ve y olarak ayrılması
        ↓
Train-test split işlemi
        ↓
SMOTE ile eğitim verisinin dengelenmesi
        ↓
Random Forest modelinin eğitilmesi
        ↓
Model performansının değerlendirilmesi
        ↓
Modelin joblib ile kaydedilmesi
        ↓
FastAPI ile API haline getirilmesi
        ↓
Kişiye özel risk motorunun eklenmesi