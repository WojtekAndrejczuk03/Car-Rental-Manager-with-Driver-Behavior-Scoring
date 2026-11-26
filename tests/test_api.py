"""Endpointy API na tymczasowej kopii bazy."""
from pathlib import Path

NAGLOWEK = Path("data/wynajmy.csv").read_text(encoding="utf-8").splitlines()[0]
NOWY_WYNAJEM = "9001,9001,Fiat 500,miejski,A,70,5.3,6.4,12,26.0,58.0,miasto,40.0,3,nie"


def test_podsumowanie(klient):
    dane = klient.get("/api/podsumowanie").json()
    assert dane["liczba_wynajmow"] == 3000
    assert dane["liczba_kierowcow"] == 598
    assert 0 <= dane["srednia_ocena"] <= 100


def test_lista_kierowcow(klient):
    kierowcy = klient.get("/api/kierowcy").json()
    assert len(kierowcy) == 598
    k = kierowcy[0]
    assert {"ocena", "styl", "mnoznik", "ryzyko", "ryzyko_stylu", "liczba_szkod"} <= k.keys()
    assert k["styl"] in ("spokojny", "normalny", "agresywny")


def test_profil_kierowcy(klient):
    profil = klient.get("/api/kierowcy/1").json()
    assert profil["id_kierowcy"] == 1
    assert len(profil["wynajmy"]) == profil["liczba_wynajmow"]


def test_brak_kierowcy(klient):
    assert klient.get("/api/kierowcy/999999").status_code == 404


def test_wycena_zgodna_z_ocena(klient):
    profil = klient.get("/api/kierowcy/1").json()
    w = klient.post("/api/wycena", json={"id_kierowcy": 1, "stawka_bazowa": 200}).json()
    assert w["ocena"] == profil["ocena"]
    assert w["cena"] == round(200 * profil["mnoznik"], 2)


def test_wycena_wymaga_dodatniej_stawki(klient):
    r = klient.post("/api/wycena", json={"id_kierowcy": 1, "stawka_bazowa": 0})
    assert r.status_code == 422


def test_flota(klient):
    samochody = klient.get("/api/samochody").json()
    assert len(samochody) == 26
    assert sum(s["liczba_wynajmow"] for s in samochody) == 3000
    fiat = next(s for s in samochody if s["marka_i_model"] == "Fiat 500")
    assert fiat["liczba_koni_mechanicznych"] == 70
    assert 0 <= fiat["srednia_ocena"] <= 100


def test_import_dopisuje_wynajem(klient):
    csv = f"{NAGLOWEK}\n{NOWY_WYNAJEM}\n"
    r = klient.post("/api/import", content=csv.encode(), headers={"Content-Type": "text/csv"})
    assert r.status_code == 200, r.text
    assert r.json() == {"zaimportowano": 1, "liczba_wynajmow": 3001}
    # Nowy kierowca od razu ma ocenę
    assert klient.get("/api/kierowcy/9001").json()["liczba_wynajmow"] == 1


def test_bledny_plik_nie_zmienia_bazy(klient):
    zly = NOWY_WYNAJEM.replace(",miasto,", ",kosmos,")
    csv = f"{NAGLOWEK}\n{NOWY_WYNAJEM}\n{zly.replace('9001,9001', '9002,9002')}\n"
    r = klient.post("/api/import", content=csv.encode(), headers={"Content-Type": "text/csv"})
    assert r.status_code == 400
    assert "obszar_jazdy" in r.json()["detail"]
    assert klient.get("/api/podsumowanie").json()["liczba_wynajmow"] == 3000


def test_import_zlych_kolumn(klient):
    r = klient.post("/api/import", content=b"a,b\n1,2\n", headers={"Content-Type": "text/csv"})
    assert r.status_code == 400
    assert "Brakujące" in r.json()["detail"]


def test_import_pustego_pliku(klient):
    r = klient.post("/api/import", content=f"{NAGLOWEK}\n".encode(), headers={"Content-Type": "text/csv"})
    assert r.status_code == 400
