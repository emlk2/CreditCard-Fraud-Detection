#Jupyter Notebook (.ipynb) analiz içindir; gerçek uygulamalar .py dosyalarında çalışır.
# Bu dosya bizim sunucumuzun kalbi olacak.
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. API Uygulamasını Başlat
app = FastAPI(title="Kredi Kartı Sahtekarlık Tespiti")

# 2. Kaydettiğimiz Modeli Yükle
model = joblib.load('random_forest_model.pkl')

# 3. Gelen Verinin Formatını Belirle (Data Validation)
# Modelimizin beklediği tüm kolonları buraya tanımlıyoruz
class IslemVerisi(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def ana_sayfa():
    return {"mesaj": "Sahtekarlık Tespiti API'si Aktif!"}

@app.post("/tahmin")
def tahmin_et(islem: IslemVerisi):
    # Gelen veriyi modelin anlayacağı bir tablo (DataFrame) haline getiriyoruz
    input_df = pd.DataFrame([islem.dict()])
    
    # Modelimize soruyoruz: Bu işlem sahte mi?
    tahmin = model.predict(input_df)[0]
    olasilik = model.predict_proba(input_df)[0][1] # Sahte olma ihtimali
    
    sonuc = "Sahtekarlık Şüphesi Var!" if tahmin == 1 else "Normal İşlem"
    
    return {
        "sonuc": sonuc,
        "sahtekarlik_skoru": float(olasilik),
        "tahmin_kodu": int(tahmin)
    }