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

from fastapi.responses import HTMLResponse
from fastapi.responses import HTMLResponse

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def arayuz_simulasyonu():
    html_icerik = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Anti-Fraud Karar Motoru | Canlı Test Paneli</title>
        <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); min-height: 100vh; margin: 0; display: flex; justify-content: center; align-items: center; color: #333; padding: 20px; box-sizing: border-box;}
            .dashboard-card { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); max-width: 800px; width: 100%; text-align: center; }
            .header-icon { font-size: 40px; color: #2a5298; margin-bottom: 10px; }
            h2 { margin-top: 0; color: #1e3c72; font-size: 24px; }
            p { color: #666; font-size: 14px; margin-bottom: 20px; }
            .btn-group { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-bottom: 20px; }
            .btn-load { border: 1px solid #ccc; background: #f8f9fa; padding: 8px 15px; border-radius: 5px; cursor: pointer; font-size: 13px; font-weight: bold; transition: 0.2s; color: #333; }
            .btn-load:hover { background: #e2e6ea; }
            .code-editor { width: 100%; height: 350px; background: #282c34; color: #abb2bf; font-family: 'Courier New', Courier, monospace; font-size: 14px; padding: 15px; border-radius: 8px; border: none; outline: none; resize: vertical; box-sizing: border-box; text-align: left; }
            .btn-submit { background-color: #28a745; color: white; border: none; padding: 15px 30px; border-radius: 8px; font-size: 18px; font-weight: bold; cursor: pointer; transition: all 0.3s ease; margin-top: 20px; width: 100%; box-shadow: 0 4px 15px rgba(40,167,69,0.3); }
            .btn-submit:hover { background-color: #218838; transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <div class="dashboard-card">
            <i class="fa-solid fa-shield-halved header-icon"></i>
            <h2>Anti-Fraud Manuel Analiz Paneli</h2>
            <p>Aşağıdaki alana kendi JSON verinizi girebilir veya üstteki butonlarla örnek senaryo yükleyerek sonuçları test edebilirsiniz.</p>
            
            <div class="btn-group">
                <button class="btn-load" onclick="ornekYukle('dusuk')">🟢 Düşük Risk Örneği Yükle</button>
                <button class="btn-load" onclick="ornekYukle('orta')">🟡 Orta Risk Örneği Yükle</button>
                <button class="btn-load" onclick="ornekYukle('yuksek')">🔴 Yüksek Risk Örneği Yükle</button>
            </div>

            <textarea id="jsonInput" class="code-editor" spellcheck="false"></textarea>

            <button class="btn-submit" onclick="veriyiAnalizEt()">
                <i class="fa-solid fa-microchip"></i> SİSTEME GÖNDER VE ANALİZ ET
            </button>
        </div>

        <script>
            const senaryolar = {
                'dusuk': {
                    "islem": { "Time": 45000, "V1": -0.42, "V2": 0.15, "V3": 1.25, "V4": -0.30, "V5": 0.50, "V6": -0.15, "V7": 0.30, "V8": 0.05, "V9": -0.20, "V10": 0.10, "V11": -0.60, "V12": 0.40, "V13": -0.10, "V14": 0.50, "V15": -0.15, "V16": 0.20, "V17": -0.20, "V18": 0.10, "V19": -0.05, "V20": 0.05, "V21": -0.10, "V22": -0.25, "V23": 0.05, "V24": 0.10, "V25": -0.20, "V26": 0.05, "V27": -0.05, "V28": 0.02, "Amount": 55.00 },
                    "kullanici_profili": { "user_id": "Mehmet-1001", "ortalama_tutar": 120.00, "maksimum_normal_tutar": 500.00, "normal_baslangic_saati": 8, "normal_bitis_saati": 22, "ulke_uygun_mu": true, "guvenilir_cihaz_mi": true }
                },
                'orta': {
                    "islem": { "Time": 14400, "V1": -1.50, "V2": 1.20, "V3": -0.80, "V4": 1.50, "V5": -0.90, "V6": 0.60, "V7": -0.70, "V8": 0.40, "V9": -0.90, "V10": -1.10, "V11": 1.20, "V12": -1.30, "V13": 0.30, "V14": -1.20, "V15": 0.50, "V16": -0.90, "V17": -1.10, "V18": -0.40, "V19": 0.60, "V20": 0.25, "V21": 0.20, "V22": 0.45, "V23": -0.15, "V24": -0.30, "V25": 0.30, "V26": -0.20, "V27": 0.10, "V28": -0.05, "Amount": 3250.50 },
                    "kullanici_profili": { "user_id": "Mehmet-1002", "ortalama_tutar": 300.00, "maksimum_normal_tutar": 2500.00, "normal_baslangic_saati": 9, "normal_bitis_saati": 23, "ulke_uygun_mu": true, "guvenilir_cihaz_mi": false }
                },
                'yuksek': {
                    "islem": { "Time": 10800, "V1": -5.20, "V2": 4.50, "V3": -7.80, "V4": 6.10, "V5": -3.50, "V6": -2.10, "V7": -6.50, "V8": 2.80, "V9": -4.20, "V10": -7.50, "V11": 5.80, "V12": -8.90, "V13": -1.20, "V14": -9.50, "V15": 1.10, "V16": -5.60, "V17": -10.50, "V18": -4.20, "V19": 1.80, "V20": 0.90, "V21": 1.10, "V22": -0.60, "V23": -0.35, "V24": -0.20, "V25": 0.60, "V26": 0.45, "V27": 1.20, "V28": -0.30, "Amount": 9500.00 },
                    "kullanici_profili": { "user_id": "Mehmet-1003", "ortalama_tutar": 85.00, "maksimum_normal_tutar": 800.00, "normal_baslangic_saati": 8, "normal_bitis_saati": 21, "ulke_uygun_mu": false, "guvenilir_cihaz_mi": false }
                }
            };

            // Sayfa açıldığında varsayılan olarak Düşük Riski yükle
            window.onload = function() {
                ornekYukle('dusuk');
            };

            // Kutuya örnek veriyi basan fonksiyon
            function ornekYukle(tip) {
                const kutu = document.getElementById('jsonInput');
                // JSON'u şık bir şekilde (4 boşluklu) kutuya yazdır
                kutu.value = JSON.stringify(senaryolar[tip], null, 4);
            }

            // Ana işlem fonksiyonu
            async function veriyiAnalizEt() {
                const kutuIcerigi = document.getElementById('jsonInput').value;
                let gonderilecekVeri;

                // 1. Önce kullanıcının girdiği verinin doğru bir JSON olup olmadığını kontrol ediyoruz
                try {
                    gonderilecekVeri = JSON.parse(kutuIcerigi);
                } catch (error) {
                    Swal.fire({ icon: 'error', title: 'Format Hatası!', text: 'Girdiğiniz veri geçerli bir JSON formatında değil. Lütfen parantezleri ve virgülleri kontrol edin.' });
                    return;
                }

                // 2. Yükleme ekranı çıkarıyoruz
                Swal.fire({ title: 'Yapay Zeka Analiz Ediyor...', html: 'Verileriniz Random Forest ve Kural Motorundan geçiriliyor.', allowOutsideClick: false, didOpen: () => { Swal.showLoading() } });
                
                // 3. API'ye gönderiyoruz
                try {
                    const response = await fetch('/tahmin-gelismis', { 
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(gonderilecekVeri)
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        Swal.fire({ icon: 'error', title: 'API Hatası (' + response.status + ')', text: JSON.stringify(errorData) });
                        return;
                    }

                    const data = await response.json();
                    
                    // 4. Sonuca göre o şık pencereleri patlatıyoruz
                    if (data.risk_kategorisi === "Yüksek Risk" || data.tahmin_sinifi === 1) {
                        Swal.fire({ icon: 'error', title: '🚨 İŞLEM REDDEDİLDİ!', html: `<div style="text-align: left; background: #f8d7da; padding: 15px; border-radius: 8px; margin-top: 10px; color:#721c24;"><b>Sistem Kararı:</b> ${data.sonuc || 'Sahtekarlık Şüphesi'}<br><b>Risk Seviyesi:</b> Yüksek Risk<br><b>Aksiyon:</b> İşlem bloke edildi ve karta geçici kısıtlama konuldu.</div>`, confirmButtonText: 'Kapat', confirmButtonColor: '#dc3545' });
                    } else if (data.risk_kategorisi === "Orta Risk") {
                        Swal.fire({ icon: 'warning', title: '⚠️ ŞÜPHELİ İŞLEM TESPİTİ', html: `<div style="text-align: left; background: #fff3cd; padding: 15px; border-radius: 8px; margin-top: 10px; color:#856404;"><b>Sistem Kararı:</b> Cihaz veya Saat Anormalliği<br><b>Risk Seviyesi:</b> Orta Risk<br><b>Aksiyon:</b> Müşteriye 3D Secure SMS doğrulama kodu gönderildi.</div>`, confirmButtonText: 'Kapat', confirmButtonColor: '#ffc107' });
                    } else {
                        Swal.fire({ icon: 'success', title: '✅ İŞLEM ONAYLANDI', html: `<div style="text-align: left; background: #d4edda; padding: 15px; border-radius: 8px; margin-top: 10px; color:#155724;"><b>Sistem Kararı:</b> Anomali Bulunamadı<br><b>Risk Seviyesi:</b> Düşük Risk<br><b>Aksiyon:</b> Tutar hesaptan düşüldü.</div>`, confirmButtonText: 'Kapat', confirmButtonColor: '#28a745' });
                    }
                } catch (error) {
                    Swal.fire({ icon: 'question', title: 'Bağlantı Hatası', text: 'API sunucusuna ulaşılamıyor.' });
                }
            }
        </script>
    </body>
    </html>
    """
    return html_icerik

 

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
    