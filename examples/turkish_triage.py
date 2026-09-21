"""Check Turkish comprehension and structured instructions."""

from _client import post, show

payload = {
    "model": "qwen3.5-0.8b-systemone",
    "state": {
        "müşteri_mesajı": "Son güncellemeden beri ödeme yapamıyorum. Kartımdan iki kez para çekildi ve siparişim oluşmadı.",
        "müşteri_türü": "kurumsal",
        "ortam": "canlı",
    },
    "questions": {
        "ekip": {
            "type": "choice",
            "instructions": {
                "soru": "Bu talebi öncelikle hangi ekip ele almalı?",
                "kural": "Mesajdaki en acil ve somut sorunu esas al.",
            },
            "criteria": {
                "teknik": "Uygulama hataları, kesintiler ve başarısız işlemler",
                "finans": "Çift çekim, iade, fatura ve ücret sorunları",
                "satış": "Fiyatlandırma, paketler ve yeni satın alımlar",
            },
        },
        "müşteri_memnuniyetsizliği": {
            "type": "score",
            "instructions": "Müşterinin memnuniyetsizlik düzeyi nedir?",
            "criteria": ["Sakin", "Rahatsız", "Çok kızgın"],
        },
        "acil": {
            "type": "noul",
            "instructions": "Bu talep acil müdahale gerektiriyor mu?",
        },
    },
}

show(post(payload))
