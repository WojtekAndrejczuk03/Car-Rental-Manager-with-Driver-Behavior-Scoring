"""Uczenie modelu ryzyka szkody i porównanie z metodą bazową.

Kroki:
1. Wczytanie danych i wyliczenie cech (features.py).
2. Walidacja krzyżowa GroupKFold według kierowcy: ten sam kierowca nigdy
   nie trafia jednocześnie do części uczącej i testowej.
3. ROC AUC metody bazowej (gwałtowne przyspieszenia na 100 km) i modelu.
4. Korelacja Spearmana oceny z prawdziwą agresywnością z generatora
   (plik walidacyjny służy tylko do sprawdzenia, nie do uczenia).
5. Uczenie końcowego modelu na wszystkich danych i zapis do models/.

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.model.train
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.model.features import (CECHA_BAZOWA, CECHY_KATEGORYCZNE, CECHY_LICZBOWE,
                                zbuduj_cechy, zbuduj_etykiete)

KATALOG_GLOWNY = Path(__file__).resolve().parents[2]
PLIK_DANYCH = KATALOG_GLOWNY / "data" / "wynajmy.csv"
PLIK_WALIDACJI = KATALOG_GLOWNY / "data" / "walidacja_agresywnosc.csv"
KATALOG_MODELI = KATALOG_GLOWNY / "models"
PLIK_MODELU = KATALOG_MODELI / "model.joblib"
PLIK_METRYK = KATALOG_MODELI / "metryki.json"

LICZBA_FOLDOW = 5

# Kryteria sukcesu z karty tematu
PROG_AUC = 0.70
PROG_PRZEWAGI_NAD_BAZA = 0.08
PROG_SPEARMAN = 0.70


def zbuduj_model() -> Pipeline:
    """Regresja logistyczna: cechy liczbowe są standaryzowane, kategoryczne kodowane 0/1."""
    przygotowanie = ColumnTransformer([
        ("liczbowe", StandardScaler(), CECHY_LICZBOWE),
        ("kategoryczne", OneHotEncoder(handle_unknown="ignore"), CECHY_KATEGORYCZNE),
    ])
    return Pipeline([
        ("przygotowanie", przygotowanie),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def walidacja_krzyzowa(X: pd.DataFrame, y: pd.Series, grupy: pd.Series):
    """Zwraca AUC bazy i modelu w każdym foldzie oraz prognozy spoza próby uczącej."""
    auc_baza, auc_model = [], []
    prognozy = np.zeros(len(y))
    for uczace, testowe in GroupKFold(n_splits=LICZBA_FOLDOW).split(X, y, grupy):
        y_test = y.iloc[testowe]
        # Metoda bazowa nie uczy się niczego: ryzyko = przyspieszenia na 100 km
        auc_baza.append(roc_auc_score(y_test, X[CECHA_BAZOWA].iloc[testowe]))
        model = zbuduj_model().fit(X.iloc[uczace], y.iloc[uczace])
        p = model.predict_proba(X.iloc[testowe])[:, 1]
        prognozy[testowe] = p
        auc_model.append(roc_auc_score(y_test, p))
    return np.array(auc_baza), np.array(auc_model), prognozy


def zgodnosc_z_agresywnoscia(df: pd.DataFrame, prognozy: np.ndarray) -> dict:
    """Spearman między przewidywanym ryzykiem a prawdziwą agresywnością z generatora."""
    walidacja = pd.read_csv(PLIK_WALIDACJI)
    t = df[["id_wynajmu", "id_kierowcy"]].assign(ryzyko=prognozy)
    t = t.merge(walidacja[["id_wynajmu", "agresywnosc_prawdziwa"]], on="id_wynajmu")
    na_wynajem = spearmanr(t["ryzyko"], t["agresywnosc_prawdziwa"])[0]
    # Ocena kierowcy to średnia z jego wynajmów, więc tak też ją sprawdzamy
    kierowcy = t.groupby("id_kierowcy")[["ryzyko", "agresywnosc_prawdziwa"]].mean()
    na_kierowce = spearmanr(kierowcy["ryzyko"], kierowcy["agresywnosc_prawdziwa"])[0]
    return {"spearman_wynajmy": float(na_wynajem), "spearman_kierowcy": float(na_kierowce)}


def wspolczynniki(model: Pipeline) -> dict:
    """Współczynniki regresji logistycznej: znak i wielkość wpływu cechy na ryzyko."""
    nazwy = model.named_steps["przygotowanie"].get_feature_names_out()
    wagi = model.named_steps["model"].coef_[0]
    nazwy = [n.split("__", 1)[1] for n in nazwy]
    return dict(sorted(zip(nazwy, map(float, wagi)), key=lambda kv: -abs(kv[1])))


def main():
    df = pd.read_csv(PLIK_DANYCH)
    X, y, grupy = zbuduj_cechy(df), zbuduj_etykiete(df), df["id_kierowcy"]

    auc_baza, auc_model, prognozy = walidacja_krzyzowa(X, y, grupy)
    zgodnosc = zgodnosc_z_agresywnoscia(df, prognozy)

    model = zbuduj_model().fit(X, y)
    KATALOG_MODELI.mkdir(exist_ok=True)
    joblib.dump(model, PLIK_MODELU)

    przewaga = auc_model.mean() - auc_baza.mean()
    metryki = {
        "model": "regresja logistyczna",
        "walidacja": f"GroupKFold po kierowcy, {LICZBA_FOLDOW} foldów",
        "liczba_wynajmow": int(len(df)),
        "liczba_kierowcow": int(grupy.nunique()),
        "odsetek_szkod": round(float(y.mean()), 4),
        "baza_auc": round(float(auc_baza.mean()), 4),
        "baza_auc_std": round(float(auc_baza.std()), 4),
        "model_auc": round(float(auc_model.mean()), 4),
        "model_auc_std": round(float(auc_model.std()), 4),
        "przewaga_nad_baza": round(float(przewaga), 4),
        "spearman_wynajmy": round(zgodnosc["spearman_wynajmy"], 4),
        "spearman_kierowcy": round(zgodnosc["spearman_kierowcy"], 4),
        "kryteria": {
            f"model_auc >= {PROG_AUC}": bool(auc_model.mean() >= PROG_AUC),
            f"przewaga_nad_baza >= {PROG_PRZEWAGI_NAD_BAZA}": bool(przewaga >= PROG_PRZEWAGI_NAD_BAZA),
            f"spearman_kierowcy >= {PROG_SPEARMAN}": bool(zgodnosc["spearman_kierowcy"] >= PROG_SPEARMAN),
        },
        "wspolczynniki": {k: round(v, 4) for k, v in wspolczynniki(model).items()},
    }
    PLIK_METRYK.write_text(json.dumps(metryki, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Dane: {metryki['liczba_wynajmow']} wynajmów, {metryki['liczba_kierowcow']} kierowców, "
          f"szkody w {metryki['odsetek_szkod']:.1%} wynajmów")
    print(f"Walidacja: {metryki['walidacja']}")
    print(f"  Metoda bazowa  ROC AUC = {auc_baza.mean():.3f} ± {auc_baza.std():.3f}")
    print(f"  Model          ROC AUC = {auc_model.mean():.3f} ± {auc_model.std():.3f}")
    print(f"  Przewaga nad bazą      = {przewaga:+.3f}")
    print(f"  Spearman (kierowcy)    = {zgodnosc['spearman_kierowcy']:.3f}")
    print("Kryteria sukcesu z karty:")
    for nazwa, spelnione in metryki["kryteria"].items():
        print(f"  [{'OK' if spelnione else 'NIE'}] {nazwa}")
    print(f"Zapisano model: {PLIK_MODELU.relative_to(KATALOG_GLOWNY)}, "
          f"metryki: {PLIK_METRYK.relative_to(KATALOG_GLOWNY)}")


if __name__ == "__main__":
    main()
