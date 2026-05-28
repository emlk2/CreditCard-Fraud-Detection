from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
def pydantic_to_dict(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj.dict()


app = FastAPI(
    title="Kredi Kartı Sahtekarlık Tespiti API",
    description="Random Forest modeli ile kredi kartı işlemlerinde sahtekarlık riskini tahmin eder.",
    version="1.1.0"
)


# Kaydedilen model paketini yüklüyoruz.
try:
    model_paketi = joblib.load("random_forest_model.pkl")
    model = model_paketi["model"]
    FEATURE_COLUMNS = model_paketi["feature_columns"]
except FileNotFoundError:
    model = None
    FEATURE_COLUMNS = []


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


class KullaniciProfili(BaseModel):
    user_id: str = Field(..., description="Kullanıcı kimliği")
    ortalama_tutar: float = Field(..., ge=0, description="Kullanıcının ortalama işlem tutarı")
    maksimum_normal_tutar: float = Field(..., ge=0, description="Kullanıcı için normal kabul edilen maksimum işlem tutarı")
    normal_baslangic_saati: int = Field(..., ge=0, le=23, description="Kullanıcının genelde işlem yapmaya başladığı saat")
    normal_bitis_saati: int = Field(..., ge=0, le=23, description="Kullanıcının genelde işlem yaptığı son saat")
    ulke_uygun_mu: bool = Field(..., description="İşlem kullanıcının alışılmış ülke/konum bilgisinden mi geliyor?")
    guvenilir_cihaz_mi: bool = Field(..., description="İşlem daha önce kullanılan güvenilir cihazdan mı yapıldı?")


class GelismisTahminVerisi(BaseModel):
    islem: IslemVerisi
    kullanici_profili: KullaniciProfili


@app.get("/")
def ana_sayfa():
    return {
        "mesaj": "Kredi Kartı Sahtekarlık Tespiti API çalışıyor.",
        "endpointler": {
            "genel_tahmin": "/tahmin",
            "gelismis_tahmin": "/tahmin-gelismis",
            "dokumantasyon": "/docs"
        },
        "kullanilan_ozellikler": FEATURE_COLUMNS,
        "kullanilan_ozellik_sayisi": len(FEATURE_COLUMNS)
    }


def kural_tabanli_risk_hesapla(islem: IslemVerisi, profil: KullaniciProfili):
    risk_skoru = 0
    risk_nedenleri = []

    veri = islem.dict()
    islem_saati = int((veri["Time"] // 3600) % 24)
    tutar = veri["Amount"]

    # Tutar kullanıcının ortalamasından çok yüksekse risk artırılır.
    if tutar > profil.ortalama_tutar * 5:
        risk_skoru += 25
        risk_nedenleri.append("İşlem tutarı kullanıcının ortalama harcamasının 5 katından fazla.")

    # Tutar kullanıcının normal maksimum tutarını aşıyorsa risk artırılır.
    if tutar > profil.maksimum_normal_tutar:
        risk_skoru += 20
        risk_nedenleri.append("İşlem tutarı kullanıcının normal kabul edilen maksimum tutarını aşıyor.")

    # İşlem kullanıcının alışılmış saat aralığı dışında yapıldıysa risk artırılır.
    if not (profil.normal_baslangic_saati <= islem_saati <= profil.normal_bitis_saati):
        risk_skoru += 15
        risk_nedenleri.append("İşlem kullanıcının alışılmış saat aralığı dışında yapıldı.")

    # Konum/ülke alışılmış değilse risk artırılır.
    if not profil.ulke_uygun_mu:
        risk_skoru += 25
        risk_nedenleri.append("İşlem kullanıcının alışılmış ülke veya konum bilgisi dışında görünüyor.")

    # Cihaz güvenilir değilse risk artırılır.
    if not profil.guvenilir_cihaz_mi:
        risk_skoru += 15
        risk_nedenleri.append("İşlem güvenilir cihaz listesindeki bir cihazdan yapılmadı.")

    risk_skoru = min(risk_skoru, 100)

    return risk_skoru, risk_nedenleri, islem_saati


@app.post("/tahmin")
def tahmin_et(islem: IslemVerisi):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model dosyası bulunamadı. random_forest_model.pkl dosyasını kontrol edin."
        )

    try:
        # Gelen işlem verisini sözlüğe çeviriyoruz.
        veri = islem.dict()

        # Time değerinden işlem saatini sadece açıklama amaçlı hesaplıyoruz.
        zaman_saniye = veri["Time"]
        islem_saati = int((zaman_saniye // 3600) % 24)

        # Modelin eğitimde gördüğü sütun sırasına göre DataFrame oluşturuyoruz.
        input_df = pd.DataFrame([veri])
        input_df = input_df[FEATURE_COLUMNS]

        # Modelden tahmin sınıfı ve sahtekarlık olasılığı alıyoruz.
        tahmin = int(model.predict(input_df)[0])
        olasilik = float(model.predict_proba(input_df)[0][1])

        # Olasılığa göre risk kategorisi belirliyoruz.
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
            "islem_saati_analizi": f"{islem_saati}:00 sularında yapılmış işlem.",
            "kullanilan_ozellik_sayisi": len(FEATURE_COLUMNS)
        }

    except KeyError as hata:
        raise HTTPException(
            status_code=400,
            detail=f"Eksik veya hatalı sütun bilgisi: {str(hata)}"
        )

    except Exception as hata:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin sırasında hata oluştu: {str(hata)}"
        )


@app.post("/tahmin-gelismis")
def gelismis_tahmin_et(veri: GelismisTahminVerisi):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model dosyası bulunamadı. random_forest_model.pkl dosyasını kontrol edin."
        )

    try:
        islem = veri.islem
        profil = veri.kullanici_profili

        # İşlem verisini modelin beklediği formata çeviriyoruz.
        islem_sozluk = islem.dict()
        input_df = pd.DataFrame([islem_sozluk])
        input_df = input_df[FEATURE_COLUMNS]

        # Random Forest modelinden tahmin ve olasılık alıyoruz.
        tahmin = int(model.predict(input_df)[0])
        model_olasiligi = float(model.predict_proba(input_df)[0][1])
        model_skoru = model_olasiligi * 100

        # Kullanıcı profiline göre kural tabanlı risk hesaplıyoruz.
        kural_skoru, risk_nedenleri, islem_saati = kural_tabanli_risk_hesapla(islem, profil)

        # Model skoru ve kural skoru birleştirilir.
        nihai_risk_skoru = (model_skoru * 0.60) + (kural_skoru * 0.40)
        nihai_risk_skoru = min(nihai_risk_skoru, 100)

        # Nihai risk kategorisi belirlenir.
        if nihai_risk_skoru < 30:
            risk_kategorisi = "Düşük Risk"
            sonuc = "Normal İşlem"
            onerilen_aksiyon = "İşlem onaylanabilir."
        elif nihai_risk_skoru < 70:
            risk_kategorisi = "Orta Risk"
            sonuc = "Ek Doğrulama Gerekli"
            onerilen_aksiyon = "SMS veya mobil uygulama onayı istenebilir."
        else:
            risk_kategorisi = "Yüksek Risk"
            sonuc = "Sahtekarlık Şüphesi Var"
            onerilen_aksiyon = "İşlem geçici olarak durdurulmalı ve kullanıcıdan onay alınmalıdır."

        return {
            "user_id": profil.user_id,
            "sonuc": sonuc,
            "risk_kategorisi": risk_kategorisi,
            "model_tahmini": tahmin,
            "model_skoru": round(model_skoru, 2),
            "kural_tabanli_skor": round(kural_skoru, 2),
            "nihai_risk_skoru": round(nihai_risk_skoru, 2),
            "islem_saati": islem_saati,
            "risk_nedenleri": risk_nedenleri if risk_nedenleri else [
                "Kullanıcı profiline göre belirgin bir risk nedeni bulunmadı."
            ],
            "onerilen_aksiyon": onerilen_aksiyon
        }

    except KeyError as hata:
        raise HTTPException(
            status_code=400,
            detail=f"Eksik veya hatalı sütun bilgisi: {str(hata)}"
        )

    except Exception as hata:
        raise HTTPException(
            status_code=500,
            detail=f"Gelişmiş tahmin sırasında hata oluştu: {str(hata)}"
        )