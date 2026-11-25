"""Cechy wejściowe modelu wyliczane z surowych danych o wynajmie.

Ten sam kod liczy cechy przy uczeniu (train.py), przy ocenie kierowcy
(scoring.py) i w API, więc model zawsze dostaje dane w tej samej postaci.

Cechy dzielą się na dwie grupy:
- cechy stylu jazdy: jak kierowca jedzie w porównaniu z typowym kierowcą
  w tym samym obszarze (CechyWzgledemObszaru),
- cechy kontekstu: gdzie, ile i czym jeździ (przebieg, obszar, auto).
Model przewiduje szkodę z obu grup, ale ocena kierowcy i cena zależą
tylko od stylu (patrz ModelOceny.ryzyko_stylu).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Surowe miary jazdy, które w różnych obszarach mają zupełnie inne typowe
# wartości (np. najwyższa prędkość: ok. 65 km/h w mieście, ok. 175 km/h na autostradzie)
MIARY_JAZDY = [
    "przyspieszenia_na_100km",
    "stosunek_spalania",
    "srednia_predkosc_kmh",
    "najwyzsza_predkosc_kmh",
]
# Te same miary po porównaniu z typowym kierowcą w danym obszarze (odchylenia standardowe)
CECHY_STYLU = [f"{m}_wzgl" for m in MIARY_JAZDY]

CECHY_KONTEKSTU_LICZBOWE = ["log_km_razem", "liczba_koni_mechanicznych"]
CECHY_KATEGORYCZNE = ["obszar_jazdy", "typ_samochodu"]
CECHY_LICZBOWE = CECHY_STYLU + CECHY_KONTEKSTU_LICZBOWE
CECHY = MIARY_JAZDY + CECHY_KONTEKSTU_LICZBOWE + CECHY_KATEGORYCZNE

# Punkt odniesienia z karty tematu: ryzyko = liczba gwałtownych przyspieszeń na 100 km
CECHA_BAZOWA = "przyspieszenia_na_100km"

KOLUMNA_CELU = "uszkodzenie_samochodu"


def zbuduj_cechy(df: pd.DataFrame) -> pd.DataFrame:
    """Zwraca tabelę cech dla każdego wynajmu z `df` (przed porównaniem z obszarem)."""
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


class CechyWzgledemObszaru(BaseEstimator, TransformerMixin):
    """Porównuje miary jazdy z typowym kierowcą w tym samym obszarze jazdy.

    130 km/h na autostradzie to spokojna jazda, a w mieście bardzo agresywna.
    Dla każdego obszaru zapamiętujemy (tylko na danych uczących) średnią
    i odchylenie standardowe każdej miary, a potem zamieniamy miarę na
    „o ile odchyleń standardowych kierowca odbiega od typowego w tym obszarze”.
    Liczba przyspieszeń ma rozkład mocno skośny, więc porównujemy jej logarytm.
    """

    def fit(self, X: pd.DataFrame, y=None):
        miary = self._miary(X)
        grupy = miary.groupby(X["obszar_jazdy"])
        self.srednie_ = grupy.mean().to_dict("index")
        self.odchylenia_ = grupy.std().to_dict("index")
        # Dla obszaru, którego nie było w danych uczących, porównujemy z całą flotą
        self.srednie_ogolne_ = miary.mean().to_dict()
        self.odchylenia_ogolne_ = miary.std().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        miary = self._miary(X)
        wynik = X.copy()
        for miara, cecha in zip(MIARY_JAZDY, CECHY_STYLU):
            srednia = X["obszar_jazdy"].map(lambda o: self.srednie_.get(o, self.srednie_ogolne_)[miara])
            odch = X["obszar_jazdy"].map(lambda o: self.odchylenia_.get(o, self.odchylenia_ogolne_)[miara])
            wynik[cecha] = (miary[miara] - srednia.astype(float)) / odch.astype(float)
        return wynik

    @staticmethod
    def _miary(X: pd.DataFrame) -> pd.DataFrame:
        miary = X[MIARY_JAZDY].astype(float).copy()
        miary["przyspieszenia_na_100km"] = np.log1p(miary["przyspieszenia_na_100km"])
        return miary
