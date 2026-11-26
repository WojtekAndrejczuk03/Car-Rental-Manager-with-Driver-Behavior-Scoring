"""API dla panelu zarządcy: kierowcy, ich oceny, historia wynajmów i wycena.

Dane pochodzą z bazy SQLite (scripts/import_csv.py), a ryzyko szkody z modelu
(python -m src.model.train). Oceny liczymy przy każdym zapytaniu, więc po
imporcie nowych wynajmów panel od razu pokazuje aktualne wyniki.

Uruchomienie (z katalogu głównego repozytorium):
    uvicorn src.api:app --reload
Dokumentacja endpointów: http://127.0.0.1:8000/docs
"""
import io
import json
import sqlite3
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scripts.import_csv import import_csv_file
from src.db import DEFAULT_DB_PATH, connect_database
from src.model.scoring import (KATALOG_GLOWNY, PLIK_METRYK, MNOZNIKI, ocen_kierowcow,
                               ocen_wynajmy, wczytaj_model, wycena)

KATALOG_PANELU = KATALOG_GLOWNY / "frontend"
MAKS_ROZMIAR_CSV = 5 * 1024 * 1024  # 5 MB, ok. 50 tys. wynajmów

# Wynajmy z bazy w tych samych kolumnach co data/wynajmy.csv, bo tego oczekuje model
ZAPYTANIE_WYNAJMY = """
SELECT w.id_wynajmu, w.id_kierowcy, w.id_samochodu,
       s.marka_i_model, s.typ_samochodu, s.klasa_pojazdu,
       s.liczba_koni_mechanicznych, s.spalanie_producenta_l_100km,
       w.spalanie_kierowcy_l_100km, w.liczba_gwaltownych_przyspieszen,
       w.srednia_predkosc_kmh, w.najwyzsza_predkosc_kmh, w.obszar_jazdy,
       w.km_na_dzien, w.liczba_dni_wynajmu, w.uszkodzenie_samochodu
FROM wynajmy w JOIN samochody s USING (id_samochodu)
ORDER BY w.id_wynajmu
"""

ZAPYTANIE_SAMOCHODY = """
SELECT id_samochodu, marka_i_model, typ_samochodu, klasa_pojazdu,
       liczba_koni_mechanicznych, spalanie_producenta_l_100km
FROM samochody
ORDER BY id_samochodu
"""

stan = {}


@asynccontextmanager
async def przy_starcie(app: FastAPI):
    """Wczytuje model i metryki raz, przy starcie serwera."""
    if not DEFAULT_DB_PATH.is_file():
        raise RuntimeError(f"Brak bazy {DEFAULT_DB_PATH}. Najpierw uruchom: python scripts/import_csv.py")
    stan["model"] = wczytaj_model()
    stan["metryki"] = json.loads(PLIK_METRYK.read_text(encoding="utf-8"))
    yield


app = FastAPI(title="Car Rental Manager — ocena stylu jazdy", lifespan=przy_starcie)


def zapytanie(sql: str) -> pd.DataFrame:
    polaczenie = connect_database(DEFAULT_DB_PATH)
    try:
        return pd.read_sql_query(sql, polaczenie)
    finally:
        polaczenie.close()


def wczytaj_wynajmy() -> pd.DataFrame:
    return zapytanie(ZAPYTANIE_WYNAJMY)


def ocen_kierowcow_ze_szkodami(wynajmy: pd.DataFrame) -> pd.DataFrame:
    """Oceny kierowców (scoring.py) uzupełnione o liczbę szkód z historii."""
    kierowcy = ocen_kierowcow(stan["model"], wynajmy)
    szkody = (wynajmy["uszkodzenie_samochodu"] == "tak").groupby(wynajmy["id_kierowcy"]).sum()
    return kierowcy.join(szkody.rename("liczba_szkod"), on="id_kierowcy")


def kierowca_do_json(k) -> dict:
    return {"id_kierowcy": int(k["id_kierowcy"]), "liczba_wynajmow": int(k["liczba_wynajmow"]),
            "liczba_szkod": int(k["liczba_szkod"]), "ryzyko": round(float(k["ryzyko"]), 4),
            "ryzyko_stylu": round(float(k["ryzyko_stylu"]), 4), "ocena": int(k["ocena"]), "styl": k["styl"], "mnoznik": float(k["mnoznik"])}


@app.get("/api/metryki")
def metryki():
    """Wynik modelu obok punktu odniesienia (z models/metryki.json)."""
    return stan["metryki"]


@app.get("/api/podsumowanie")
def podsumowanie():
    """Liczby do kafelków na stronie głównej panelu."""
    wynajmy = wczytaj_wynajmy()
    kierowcy = ocen_kierowcow(stan["model"], wynajmy)
    return {
        "liczba_kierowcow": int(len(kierowcy)),
        "liczba_wynajmow": int(len(wynajmy)),
        "srednia_ocena": int(round(kierowcy["ocena"].mean())),
        "odsetek_szkod": round(float((wynajmy["uszkodzenie_samochodu"] == "tak").mean()), 4),
    }


@app.get("/api/kierowcy")
def lista_kierowcow():
    """Wszyscy kierowcy z oceną 0–100, stylem jazdy, mnożnikiem ceny i liczbą szkód."""
    kierowcy = ocen_kierowcow_ze_szkodami(wczytaj_wynajmy())
    return [kierowca_do_json(k) for _, k in kierowcy.iterrows()]


@app.get("/api/kierowcy/{id_kierowcy}")
def profil_kierowcy(id_kierowcy: int):
    """Profil kierowcy: ocena, styl i historia wynajmów z oceną każdego z nich."""
    wynajmy = wczytaj_wynajmy()
    wynajmy = wynajmy[wynajmy["id_kierowcy"] == id_kierowcy]
    if wynajmy.empty:
        raise HTTPException(status_code=404, detail=f"Nie ma kierowcy o id {id_kierowcy}.")

    profil = kierowca_do_json(ocen_kierowcow_ze_szkodami(wynajmy).iloc[0])
    historia = ocen_wynajmy(stan["model"], wynajmy)
    profil["wynajmy"] = [{
        "id_wynajmu": int(w["id_wynajmu"]),
        "marka_i_model": w["marka_i_model"],
        "typ_samochodu": w["typ_samochodu"],
        "obszar_jazdy": w["obszar_jazdy"],
        "liczba_dni_wynajmu": int(w["liczba_dni_wynajmu"]),
        "km_razem": round(float(w["km_na_dzien"] * w["liczba_dni_wynajmu"]), 1),
        "uszkodzenie_samochodu": w["uszkodzenie_samochodu"],
        "ryzyko": round(float(w["ryzyko"]), 4),
        "ryzyko_stylu": round(float(w["ryzyko_stylu"]), 4),
        "ocena": int(w["ocena"]),
    } for _, w in historia.iterrows()]
    return profil


class ZapytanieOWycene(BaseModel):
    id_kierowcy: int
    stawka_bazowa: float = Field(gt=0, description="stawka bazowa w zł")


@app.post("/api/wycena")
def wycena_wynajmu(zapytanie: ZapytanieOWycene):
    """Cena = stawka bazowa × mnożnik ze stylu jazdy. Nieznany kierowca płaci stawkę standardową."""
    wynajmy = wczytaj_wynajmy()
    wynajmy = wynajmy[wynajmy["id_kierowcy"] == zapytanie.id_kierowcy]
    ocena: Optional[int] = None
    liczba = 0
    if not wynajmy.empty:
        k = ocen_kierowcow(stan["model"], wynajmy).iloc[0]
        ocena, liczba = int(k["ocena"]), int(k["liczba_wynajmow"])
    return {"id_kierowcy": zapytanie.id_kierowcy, **wycena(zapytanie.stawka_bazowa, ocena, liczba)}


@app.get("/api/samochody")
def lista_samochodow():
    """Flota: dane katalogowe aut, liczba wynajmów, szkód i średnia ocena stylu ich kierowców."""
    samochody = zapytanie(ZAPYTANIE_SAMOCHODY)
    wynajmy = ocen_wynajmy(stan["model"], wczytaj_wynajmy())
    wynajmy["szkoda"] = wynajmy["uszkodzenie_samochodu"] == "tak"
    statystyki = wynajmy.groupby("id_samochodu").agg(
        liczba_wynajmow=("id_wynajmu", "size"), liczba_szkod=("szkoda", "sum"),
        srednia_ocena=("ocena", "mean"))
    samochody = samochody.join(statystyki, on="id_samochodu")
    wynik = []
    for _, s in samochody.iterrows():
        liczba = 0 if pd.isna(s["liczba_wynajmow"]) else int(s["liczba_wynajmow"])
        szkody = 0 if pd.isna(s["liczba_szkod"]) else int(s["liczba_szkod"])
        wynik.append({
            "id_samochodu": int(s["id_samochodu"]),
            "marka_i_model": s["marka_i_model"],
            "typ_samochodu": s["typ_samochodu"],
            "klasa_pojazdu": s["klasa_pojazdu"],
            "liczba_koni_mechanicznych": int(s["liczba_koni_mechanicznych"]),
            "spalanie_producenta_l_100km": float(s["spalanie_producenta_l_100km"]),
            "liczba_wynajmow": liczba,
            "liczba_szkod": szkody,
            "odsetek_szkod": round(szkody / liczba, 4) if liczba else None,
            "srednia_ocena": int(round(s["srednia_ocena"])) if liczba else None,
        })
    return wynik


@app.post("/api/import")
async def import_wynajmow(request: Request):
    """Dopisuje wynajmy z pliku CSV (treść pliku w ciele zapytania, kolumny jak w data/wynajmy.csv).

    Wynajem o istniejącym id_wynajmu zostaje nadpisany. Jeśli którykolwiek
    wiersz jest błędny, nic nie zostaje zapisane. Oceny liczymy przy każdym
    zapytaniu, więc nowe wynajmy od razu wpływają na oceny kierowców.
    """
    tresc = await request.body()
    if len(tresc) > MAKS_ROZMIAR_CSV:
        raise HTTPException(status_code=413, detail="Plik CSV jest za duży (maks. 5 MB).")
    try:
        tekst = tresc.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Plik CSV musi być zapisany w kodowaniu UTF-8.")
    try:
        liczba = await run_in_threadpool(import_csv_file, io.StringIO(tekst, newline=""), DEFAULT_DB_PATH)
    except (ValueError, sqlite3.IntegrityError) as blad:
        raise HTTPException(status_code=400, detail=f"Nie zaimportowano pliku. {blad}")
    return {"zaimportowano": liczba, "liczba_wynajmow": int(len(wczytaj_wynajmy()))}


@app.get("/api/mnozniki")
def mnozniki():
    """Mnożniki ceny dla stylów jazdy (jawna formuła wyceny)."""
    return MNOZNIKI


# Panel zarządcy (frontend/) pod adresem głównym, gdy już istnieje w repozytorium
if KATALOG_PANELU.is_dir():
    app.mount("/", StaticFiles(directory=KATALOG_PANELU, html=True), name="panel")
