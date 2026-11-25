"""Ocena kierowcy 0–100, styl jazdy i wycena wynajmu.

Model (model.py) zwraca dwa ryzyka dla każdego wynajmu:
- ryzyko wynajmu: prawdopodobieństwo szkody (styl jazdy + przebieg, obszar, auto),
- ryzyko stylu: to samo, ale w jednakowym kontekście dla wszystkich kierowców.
Ten moduł:
1. liczy oba ryzyka dla każdego wynajmu,
2. uśrednia je po kierowcy,
3. porównuje ryzyko stylu ze średnim we flocie i zamienia na ocenę 0–100,
4. z oceny wyznacza styl jazdy i mnożnik ceny.
Ocena i cena zależą więc tylko od tego, JAK kierowca jeździ, a nie GDZIE
i CZYM: jazda po mieście albo długi wyjazd autostradą nie obniża oceny.

Wycena to jawna formuła (stawka bazowa × mnożnik), a nie część modelu.

Skala oceny (logarytmiczna względem średniego ryzyka stylu we flocie):
    ryzyko = średnia floty       -> 50 pkt
    ryzyko 2× mniejsze           -> 75 pkt
    ryzyko 4× mniejsze lub mniej -> 100 pkt
    ryzyko 2× większe            -> 25 pkt
    ryzyko 4× większe lub więcej -> 0 pkt
Liniowe 100 × (1 − ryzyko) nie nadaje się, bo szkody są rzadkie (~8%)
i prawie każdy kierowca dostawałby 85–97 pkt.

Podgląd na danych z data/wynajmy.csv:
    python -m src.model.scoring
"""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.model.features import zbuduj_cechy

KATALOG_GLOWNY = Path(__file__).resolve().parents[2]
PLIK_MODELU = KATALOG_GLOWNY / "models" / "model.joblib"
PLIK_METRYK = KATALOG_GLOWNY / "models" / "metryki.json"

PUNKTY_ZA_PODWOJENIE_RYZYKA = 25

# Progi stylu jazdy (ocena >= próg)
PROG_SPOKOJNY = 65
PROG_NORMALNY = 35

# Mnożniki ceny: spokojny dostaje zniżkę, agresywny płaci więcej
MNOZNIKI = {"spokojny": 0.90, "normalny": 1.00, "agresywny": 1.25}
STYL_NOWEGO_KIEROWCY = "normalny"


def wczytaj_model(sciezka: Path = PLIK_MODELU):
    if not Path(sciezka).is_file():
        raise FileNotFoundError(
            f"Brak modelu {sciezka}. Najpierw uruchom: python -m src.model.train")
    return joblib.load(sciezka)


def ryzyko_wynajmow(model, wynajmy: pd.DataFrame) -> np.ndarray:
    """Prawdopodobieństwo szkody dla każdego wynajmu (wiersze jak w data/wynajmy.csv)."""
    return model.ryzyko_wynajmu(zbuduj_cechy(wynajmy))


def ryzyko_stylu(model, wynajmy: pd.DataFrame) -> np.ndarray:
    """Ryzyko wynikające tylko ze stylu jazdy: ten sam przebieg, obszar i auto dla wszystkich."""
    return model.ryzyko_stylu(zbuduj_cechy(wynajmy))


def ocena_z_ryzyka(ryzyko: float, srednie_ryzyko: float) -> int:
    """Zamienia ryzyko szkody na ocenę 0–100 (100 = najbezpieczniejszy kierowca)."""
    stosunek = max(ryzyko, 1e-6) / srednie_ryzyko
    ocena = 50 - PUNKTY_ZA_PODWOJENIE_RYZYKA * math.log2(stosunek)
    return int(round(min(100.0, max(0.0, ocena))))


def styl_jazdy(ocena: int | None) -> str:
    if ocena is None:
        return STYL_NOWEGO_KIEROWCY
    if ocena >= PROG_SPOKOJNY:
        return "spokojny"
    if ocena >= PROG_NORMALNY:
        return "normalny"
    return "agresywny"


def ocen_wynajmy(model, wynajmy: pd.DataFrame) -> pd.DataFrame:
    """Dodaje do wynajmów kolumny `ryzyko`, `ryzyko_stylu` i `ocena` (historia ocen kierowcy)."""
    wynik = wynajmy.copy()
    wynik["ryzyko"] = ryzyko_wynajmow(model, wynajmy)
    wynik["ryzyko_stylu"] = ryzyko_stylu(model, wynajmy)
    wynik["ocena"] = [ocena_z_ryzyka(r, model.srednie_ryzyko_stylu) for r in wynik["ryzyko_stylu"]]
    return wynik


def ocen_kierowcow(model, wynajmy: pd.DataFrame) -> pd.DataFrame:
    """Jeden wiersz na kierowcę: liczba wynajmów, ryzyka, ocena, styl i mnożnik.

    `ryzyko` to średnie przewidywane ryzyko szkody w jego wynajmach (z kontekstem),
    a ocena pochodzi ze średniego ryzyka stylu, bo pojedynczy wynajem bywa
    przypadkowy (zgodność z prawdziwą agresywnością jest wyraźnie wyższa dla
    średniej niż dla pojedynczego wynajmu).
    """
    t = wynajmy[["id_kierowcy"]].assign(ryzyko=ryzyko_wynajmow(model, wynajmy),
                                        ryzyko_stylu=ryzyko_stylu(model, wynajmy))
    kierowcy = t.groupby("id_kierowcy").agg(liczba_wynajmow=("ryzyko", "size"),
                                            ryzyko=("ryzyko", "mean"),
                                            ryzyko_stylu=("ryzyko_stylu", "mean")).reset_index()
    kierowcy["ocena"] = [ocena_z_ryzyka(r, model.srednie_ryzyko_stylu) for r in kierowcy["ryzyko_stylu"]]
    kierowcy["styl"] = kierowcy["ocena"].map(styl_jazdy)
    kierowcy["mnoznik"] = kierowcy["styl"].map(MNOZNIKI)
    return kierowcy


def wycena(stawka_bazowa: float, ocena: int | None, liczba_wynajmow: int = 0) -> dict:
    """Cena wynajmu = stawka bazowa × mnożnik zależny od stylu jazdy, z uzasadnieniem."""
    styl = styl_jazdy(ocena)
    mnoznik = MNOZNIKI[styl]
    cena = round(stawka_bazowa * mnoznik, 2)

    if ocena is None:
        uzasadnienie = "Nowy kierowca bez historii wynajmów: stawka standardowa."
    else:
        zmiana = round((mnoznik - 1) * 100)
        if zmiana < 0:
            skutek = f"zniżka {-zmiana}%"
        elif zmiana > 0:
            skutek = f"dopłata {zmiana}%"
        else:
            skutek = "stawka standardowa"
        uzasadnienie = (f"Ocena {ocena}/100 na podstawie {liczba_wynajmow} wynajmów "
                        f"oznacza styl jazdy: {styl}, więc {skutek}.")

    return {"cena": cena, "stawka_bazowa": stawka_bazowa, "mnoznik": mnoznik,
            "ocena": ocena, "styl": styl, "uzasadnienie": uzasadnienie}


def main():
    model = wczytaj_model()
    wynajmy = pd.read_csv(KATALOG_GLOWNY / "data" / "wynajmy.csv")
    kierowcy = ocen_kierowcow(model, wynajmy)

    print(f"Średnie ryzyko stylu we flocie: {model.srednie_ryzyko_stylu:.1%}")
    print(f"Ocena kierowców: mediana {kierowcy['ocena'].median():.0f}, "
          f"od {kierowcy['ocena'].min()} do {kierowcy['ocena'].max()}")

    # Plik walidacyjny służy tylko do sprawdzenia sensowności stylów, nie do oceny
    prawdziwa = (pd.read_csv(KATALOG_GLOWNY / "data" / "walidacja_agresywnosc.csv")
                 .groupby("id_kierowcy")["agresywnosc_prawdziwa"].mean())
    kierowcy = kierowcy.join(prawdziwa, on="id_kierowcy")
    print("\nStyl jazdy     kierowców   średnia prawdziwa agresywność (0–1)")
    for styl in ("spokojny", "normalny", "agresywny"):
        grupa = kierowcy[kierowcy["styl"] == styl]
        print(f"  {styl:<12} {len(grupa):>9}   {grupa['agresywnosc_prawdziwa'].mean():.2f}")

    przyklad = kierowcy.sort_values("ocena").iloc[[0, len(kierowcy) // 2, -1]]
    print("\nPrzykładowe wyceny (stawka bazowa 200 zł/dzień):")
    for _, k in przyklad.iterrows():
        w = wycena(200, int(k["ocena"]), int(k["liczba_wynajmow"]))
        print(f"  kierowca {int(k['id_kierowcy']):>3}: {w['cena']:>6.2f} zł  {w['uzasadnienie']}")
    print(f"  nowy kierowca: {wycena(200, None)['cena']:>6.2f} zł  {wycena(200, None)['uzasadnienie']}")


if __name__ == "__main__":
    main()
