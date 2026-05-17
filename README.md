# Kredi Kartı Sahtekarlık Tespiti (Credit Card Fraud Detection) 🚀

Bu proje, finansal işlemlerdeki sahtekarlıkları (fraud) tespit etmek amacıyla geliştirilmiş uçtan uca (end-to-end) bir Makine Öğrenimi ve API projesidir.

Gerçek dünya veri setlerindeki en büyük problemlerden biri olan **Veri Dengesizliği (Data Imbalance)** sorunu, bu projede sentetik veri üretme yöntemleriyle çözülmüş ve eğitilen model dış dünyanın kullanımına bir REST API olarak sunulmuştur.

## 🛠️ Kullanılan Teknolojiler ve Yöntemler
* **Dil:** Python
* **Makine Öğrenimi:** Scikit-Learn, Random Forest Classifier
* **Veri Ön İşleme & Dengeleme:** Pandas, Numpy, **SMOTE** (Synthetic Minority Over-sampling Technique)
* **Backend & API:** FastAPI, Uvicorn
* **Loglama:** Otomatik Excel/CSV Performans Raporlaması
* **Güvenlik:** API Key Authentication, Rate Limiting, CORS
* **Test:** Pytest
* **CI/CD:** GitHub Actions

## 🧠 Makine Öğrenimi Yaklaşımı
Kredi kartı veri setlerinde "Sahte" işlemler, tüm işlemlerin %1'inden bile azdır. Modelin tembelleşip sürekli "Normal" tahmini yapmasını engellemek için **sadece eğitim veri setine** SMOTE uygulanarak azınlık sınıfı çoğaltılmıştır.

Modelin başarısı değerlendirilirken Accuracy (Doğruluk) yerine, sahtekarları kaçırmama oranını ifade eden **Recall (Duyarlılık)** metriğine odaklanılmıştır. Random Forest modeli ile test verisinde yüksek bir Recall skoru elde edilmiştir.

## 🚀 API Nasıl Çalıştırılır?
Projedeki yapay zeka modeli eğitildikten sonra `.pkl` formatında paketlenmiş ve FastAPI ile canlıya alınmıştır.

1. Gerekli kütüphaneleri kurun:

```bash
pip install -r requirements.txt
```

2. Uygulamayı çalıştırın:

```bash
uvicorn src.main:app --reload
```

API, http://127.0.0.1:8000 adresinde çalışacaktır.

## 📚 API Dokümantasyonu

API, FastAPI ile geliştirilmiştir ve otomatik Swagger UI dokümantasyonu http://127.0.0.1:8000/docs adresinde mevcuttur.

### Endpoint'ler

#### GET /
Ana sayfa endpoint'i. API'nin aktif olduğunu doğrular.

**Headers:**
- `X-API-Key`: API anahtarı (varsayılan: "your-secret-api-key")

**Response:**
```json
{
  "message": "Fraud Detection API is Active!"
}
```

#### POST /predict
Tek bir işlem için sahtekarlık tahmini yapar.

**Headers:**
- `X-API-Key`: API anahtarı

**Request Body:**
```json
{
  "Time": 1.0,
  "V1": -1.0,
  "V2": 2.0,
  "V3": -3.0,
  "V4": 4.0,
  "V5": -5.0,
  "V6": 6.0,
  "V7": -7.0,
  "V8": 8.0,
  "V9": -9.0,
  "V10": 10.0,
  "V11": -11.0,
  "V12": 12.0,
  "V13": -13.0,
  "V14": 14.0,
  "V15": -15.0,
  "V16": 16.0,
  "V17": -17.0,
  "V18": 18.0,
  "V19": -19.0,
  "V20": 20.0,
  "V21": -21.0,
  "V22": 22.0,
  "V23": -23.0,
  "V24": 24.0,
  "V25": -25.0,
  "V26": 26.0,
  "V27": -27.0,
  "V28": 28.0,
  "Amount": 100.0
}
```

**Response:**
```json
{
  "result": "Normal",
  "confidence": 0.95
}
```

### Güvenlik
- API Key gereklidir. Ortam değişkeni `API_KEY` ile ayarlayın.
- Rate limiting: Home 10/dakika, Predict 100/dakika.
- CORS etkin.

## 🧪 Testler
Testleri çalıştırmak için:

```bash
pytest tests/
```

## 📁 Proje Yapısı
```
.
├── src/
│   ├── __init__.py
│   ├── main.py          # FastAPI app
│   ├── models.py        # Pydantic models
│   ├── routes.py        # API endpoints
│   └── utils.py         # Utility functions
├── tests/
│   └── test_api.py      # API tests
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions CI
├── veri_analizi.ipynb   # Data analysis notebook
├── creditcard.csv       # Dataset
├── Model_Performans_Raporu.csv  # Performance report
├── requirements.txt     # Dependencies
└── README.md
```

## 🔧 Kurulum ve Çalıştırma
1. Repository'yi klonlayın.
2. Virtual environment oluşturun: `python -m venv .venv`
3. Aktifleştirin: `.venv\Scripts\activate` (Windows)
4. Bağımlılıkları yükleyin: `pip install -r requirements.txt`
5. Modeli eğitmek için notebook'u çalıştırın.
6. API'yi başlatın: `uvicorn src.main:app --reload`
