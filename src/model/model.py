"""Model ryzyka szkody z osobnym ryzykiem wynikającym tylko ze stylu jazdy.

Regresja logistyczna przewiduje szkodę z cech stylu jazdy i z kontekstu
wynajmu (przebieg, obszar, typ i moc auta). Kontekst jest potrzebny, żeby
dobrze przewidywać szkody, ale nie powinien wpływać na ocenę kierowcy:
klient z miasta nie może płacić więcej tylko dlatego, że jeździ po mieście,
a klient, który wynajmuje auto sportowe, płaci już za to w stawce bazowej.

Dlatego model daje dwa wyniki:
- ryzyko_wynajmu: prawdopodobieństwo szkody w tym konkretnym wynajmie,
- ryzyko_stylu: ryzyko tego samego kierowcy, gdyby jechał w jednakowym,
  referencyjnym kontekście (ten sam przebieg, obszar i auto dla wszystkich).
Z ryzyka stylu powstaje ocena 0–100, styl jazdy i mnożnik ceny (scoring.py).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.model.features import (CECHY_KATEGORYCZNE, CECHY_KONTEKSTU_LICZBOWE, CECHY_LICZBOWE,
                                CechyWzgledemObszaru)

# Kontekst, w którym porównujemy wszystkich kierowców. Wybór przesuwa ryzyko
# stylu wszystkich kierowców w tę samą stronę, więc nie zmienia ich kolejności.
OBSZAR_REFERENCYJNY = "mieszany"
TYP_REFERENCYJNY = "osobowy"


def zbuduj_pipeline() -> Pipeline:
    """Porównanie z obszarem → standaryzacja i kodowanie 0/1 → regresja logistyczna."""
    przygotowanie = ColumnTransformer([
        ("liczbowe", StandardScaler(), CECHY_LICZBOWE),
        ("kategoryczne", OneHotEncoder(handle_unknown="ignore"), CECHY_KATEGORYCZNE),
    ])
    return Pipeline([
        ("wzgledem_obszaru", CechyWzgledemObszaru()),
        ("przygotowanie", przygotowanie),
        ("model", LogisticRegression(max_iter=1000)),
    ])


class ModelOceny:
    """Regresja logistyczna ryzyka szkody z rozdzieleniem stylu jazdy i kontekstu."""

    def __init__(self):
        self.pipeline = zbuduj_pipeline()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelOceny":
        self.pipeline.fit(X, y)
        self.kontekst_referencyjny = {
            "log_km_razem": float(X["log_km_razem"].median()),
            "liczba_koni_mechanicznych": float(X["liczba_koni_mechanicznych"].median()),
            "obszar_jazdy": OBSZAR_REFERENCYJNY,
            "typ_samochodu": TYP_REFERENCYJNY,
        }
        # Ocena 50 pkt = przeciętny styl jazdy we flocie uczącej
        self.srednie_ryzyko_stylu = float(self.ryzyko_stylu(X).mean())
        return self

    def ryzyko_wynajmu(self, X: pd.DataFrame) -> np.ndarray:
        """Prawdopodobieństwo szkody w danym wynajmie (styl + kontekst)."""
        return self.pipeline.predict_proba(X)[:, 1]

    def ryzyko_stylu(self, X: pd.DataFrame) -> np.ndarray:
        """Prawdopodobieństwo szkody przy tym stylu jazdy w referencyjnym kontekście.

        Najpierw porównujemy jazdę z typowym kierowcą w obszarze, w którym
        kierowca naprawdę jechał, a dopiero potem podmieniamy kontekst.
        """
        porownane = self.pipeline.named_steps["wzgledem_obszaru"].transform(X)
        for kolumna, wartosc in self.kontekst_referencyjny.items():
            porownane[kolumna] = wartosc
        return self.pipeline[1:].predict_proba(porownane)[:, 1]

    def wspolczynniki(self) -> dict:
        """Współczynniki regresji logistycznej: znak i wielkość wpływu cechy na ryzyko."""
        nazwy = self.pipeline.named_steps["przygotowanie"].get_feature_names_out()
        wagi = self.pipeline.named_steps["model"].coef_[0]
        nazwy = [n.split("__", 1)[1] for n in nazwy]
        return dict(sorted(zip(nazwy, map(float, wagi)), key=lambda kv: -abs(kv[1])))


def model_tylko_kontekstu() -> Pipeline:
    """Model, który zna tylko to, co sami wpisaliśmy do generatora poza stylem:
    przebieg, obszar i typ auta. Pokazuje, ile daje sam kontekst bez stylu jazdy."""
    przygotowanie = ColumnTransformer([
        ("liczbowe", StandardScaler(), CECHY_KONTEKSTU_LICZBOWE[:1]),
        ("kategoryczne", OneHotEncoder(handle_unknown="ignore"), CECHY_KATEGORYCZNE),
    ])
    return Pipeline([("przygotowanie", przygotowanie), ("model", LogisticRegression(max_iter=1000))])
