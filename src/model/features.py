"""Cechy wejściowe modelu wyliczane z surowych danych o wynajmie.

Ten sam kod liczy cechy przy uczeniu (train.py), przy ocenie kierowcy
(scoring.py) i w API, więc model zawsze dostaje dane w tej samej postaci.
"""
import numpy as np
import pandas as pd

# Cechy liczbowe (po przeliczeniu) i kategoryczne, które trafiają do modelu
CECHY_LICZBOWE = [
    "przyspieszenia_na_100km",
    "stosunek_spalania",
    "srednia_predkosc_kmh",
    "najwyzsza_predkosc_kmh",
    "log_km_razem",
    "liczba_koni_mechanicznych",
]
CECHY_KATEGORYCZNE = ["obszar_jazdy", "typ_samochodu"]
CECHY = CECHY_LICZBOWE + CECHY_KATEGORYCZNE

# Punkt odniesienia z karty tematu: ryzyko = liczba gwałtownych przyspieszeń na 100 km
CECHA_BAZOWA = "przyspieszenia_na_100km"

KOLUMNA_CELU = "uszkodzenie_samochodu"


def zbuduj_cechy(df: pd.DataFrame) -> pd.DataFrame:
    """Zwraca tabelę cech modelu dla każdego wynajmu z `df`."""
    km_razem = (df["km_na_dzien"] * df["liczba_dni_wynajmu"]).clip(lower=1.0)
    cechy = pd.DataFrame(index=df.index)
    # Liczba przyspieszeń rośnie z przebiegiem, więc normalizujemy ją do 100 km
    cechy["przyspieszenia_na_100km"] = df["liczba_gwaltownych_przyspieszen"] / km_razem * 100
    # Spalanie porównujemy z katalogowym, bo różne auta palą różnie
    cechy["stosunek_spalania"] = df["spalanie_kierowcy_l_100km"] / df["spalanie_producenta_l_100km"]
    cechy["srednia_predkosc_kmh"] = df["srednia_predkosc_kmh"]
    cechy["najwyzsza_predkosc_kmh"] = df["najwyzsza_predkosc_kmh"]
    # Dłuższy wynajem to więcej okazji do szkody; logarytm spłaszcza bardzo długie wynajmy
    cechy["log_km_razem"] = np.log(km_razem)
    cechy["liczba_koni_mechanicznych"] = df["liczba_koni_mechanicznych"]
    cechy["obszar_jazdy"] = df["obszar_jazdy"]
    cechy["typ_samochodu"] = df["typ_samochodu"]
    return cechy[CECHY]


def zbuduj_etykiete(df: pd.DataFrame) -> pd.Series:
    """1 = auto wróciło uszkodzone, 0 = bez szkody."""
    return (df[KOLUMNA_CELU] == "tak").astype(int)
