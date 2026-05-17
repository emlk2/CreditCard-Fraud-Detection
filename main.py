from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd


app = FastAPI(
    title="Kredi Kartı Sahtekarlık Tespiti API",
    description="Random Forest modeli ile kredi kartı işlemlerinde sahtekarlık riskini tahmin eder.",
    version="1.0.0"
)


# Modelin eğitimde gördüğü sütun sırası
FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


try:
    model = joblib.load("random_forest_model.pkl")
except FileNotFoundError:
    model = None


class IslemVerisi(BaseModel):
    Time: float = Field(..., description="İşlemin veri setindeki zaman değeri, saniye cinsinden")
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
    Amount: float = Field(..., ge=0, description="İşlem tutarı")


@app.get("/")
def ana_sayfa():
    return {
        "mesaj": "Kredi Kartı Sahtekarlık Tespiti API çalışıyor.",
        "endpoint": "/tahmin",
        "dokumantasyon": "/docs"
    }


@app.post("/tahmin")
def tahmin_et(islem: IslemVerisi):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model dosyası bulunamadı. random_forest_model.pkl dosyasını kontrol edin."
        )

    try:
        # 1. Gelen veriyi sözlüğe çevir
        veri = islem.model_dump()

        # 2. İşlem saatini sadece analiz amaçlı hesapla
        zaman_saniye = veri["Time"]
        islem_saati = int((zaman_saniye // 3600) % 24)

        # 3. Modelin beklediği sütun sırasına göre DataFrame oluştur
        input_df = pd.DataFrame([veri])
        input_df = input_df[FEATURE_COLUMNS]

        # 4. Modelden olasılık al
        olasilik = float(model.predict_proba(input_df)[0][1])
        tahmin = int(model.predict(input_df)[0])

        # 5. Risk seviyesi belirle
        if olasilik < 0.30:
            risk_seviyesi = "Düşük Risk"
        elif olasilik < 0.70:
            risk_seviyesi = "Orta Risk"
        else:
            risk_seviyesi = "Yüksek Risk"

        sonuc_mesaji = "Sahtekarlık Şüphesi Var!" if tahmin == 1 else "Normal İşlem"

        return {
            "sonuc": sonuc_mesaji,
            "tahmin_sinifi": tahmin,
            "risk_kategorisi": risk_seviyesi,
            "sahtekarlik_olasiligi": round(olasilik, 4),
            "sahtekarlik_skoru": f"%{olasilik * 100:.2f}",
            "islem_saati": islem_saati,
            "islem_saati_analizi": f"{islem_saati}:00 sularında yapılmış işlem."
        }

    except Exception as hata:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin sırasında hata oluştu: {str(hata)}"
        )