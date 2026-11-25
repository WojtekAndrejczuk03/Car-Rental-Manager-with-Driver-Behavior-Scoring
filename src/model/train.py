"""Uczenie modelu ryzyka szkody i porównanie z metodą bazową.

Kroki:
1. Wczytanie danych i wyliczenie cech (features.py).
2. Walidacja krzyżowa GroupKFold według kierowcy: ten sam kierowca nigdy
   nie trafia jednocześnie do części uczącej i testowej.
3. ROC AUC metody bazowej (gwałtowne przyspieszenia na 100 km) i modelu.
4. Korelacja Spearmana oceny z prawdziwą agresywnością z generatora
   (plik walidacyjny służy tylko do sprawdzenia, nie do uczenia).
5. Analizy do pytań o model:
   - czy obszar jazdy wpływa na ocenę kierowcy o tej samej agresywności,
   - co model wnosi ponad reguły wpisane do generatora (model tylko kontekstu
     i „sufit”, czyli model znający prawdziwą agresywność).
6. Uczenie końcowego modelu na wszystkich danych i zapis do models/.

Uruchomienie (z katalogu głównego repozytorium):
    python -m src.model.train
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.model.features import CECHA_BAZOWA, CechyWzgledemObszaru, zbuduj_cechy, zbuduj_etykiete
from src.model.model import ModelOceny, model_tylko_kontekstu
from src.model.scoring import ocena_z_ryzyka

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


def foldy(X, y, grupy):
    return GroupKFold(n_splits=LICZBA_FOLDOW).split(X, y, grupy)


def walidacja_krzyzowa(X: pd.DataFrame, y: pd.Series, grupy: pd.Series) -> dict:
    """AUC w każdym foldzie oraz prognozy spoza próby uczącej (ryzyko wynajmu i ryzyko stylu)."""
    auc = {"baza": [], "baza_wzgledna": [], "model": [], "ryzyko_stylu": [], "tylko_kontekst": []}
    ryzyko = np.zeros(len(y))
    ryzyko_stylu = np.zeros(len(y))
    for uczace, testowe in foldy(X, y, grupy):
        X_ucz, y_ucz, X_test, y_test = X.iloc[uczace], y.iloc[uczace], X.iloc[testowe], y.iloc[testowe]
        # Metoda bazowa z karty nie uczy się niczego: ryzyko = przyspieszenia na 100 km
        auc["baza"].append(roc_auc_score(y_test, X_test[CECHA_BAZOWA]))
        # Uczciwsza baza: te same przyspieszenia, ale w porównaniu z typowymi dla obszaru
        wzgl = CechyWzgledemObszaru().fit(X_ucz).transform(X_test)[f"{CECHA_BAZOWA}_wzgl"]
        auc["baza_wzgledna"].append(roc_auc_score(y_test, wzgl))

        model = ModelOceny().fit(X_ucz, y_ucz)
        ryzyko[testowe] = model.ryzyko_wynajmu(X_test)
        ryzyko_stylu[testowe] = model.ryzyko_stylu(X_test)
        auc["model"].append(roc_auc_score(y_test, ryzyko[testowe]))
        auc["ryzyko_stylu"].append(roc_auc_score(y_test, ryzyko_stylu[testowe]))

        kontekst = model_tylko_kontekstu().fit(X_ucz, y_ucz)
        auc["tylko_kontekst"].append(roc_auc_score(y_test, kontekst.predict_proba(X_test)[:, 1]))
    return {"auc": {k: np.array(v) for k, v in auc.items()},
            "ryzyko": ryzyko, "ryzyko_stylu": ryzyko_stylu}


def auc_sufitu(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, grupy: pd.Series) -> float:
    """AUC modelu, który zna prawdziwą agresywność i wszystkie pozostałe składniki
    reguły szkody z generatora. Tylko do analizy: lepiej nie da się przewidzieć
    szkody, bo reszta to losowość wpisana do generatora."""
    prawdziwa = df[["id_wynajmu"]].merge(pd.read_csv(PLIK_WALIDACJI), on="id_wynajmu")
    Z = pd.DataFrame({"agresywnosc": prawdziwa["agresywnosc_prawdziwa"].to_numpy(),
                      "log_km": X["log_km_razem"].to_numpy(),
                      "sportowy": (X["typ_samochodu"] == "sportowy").to_numpy(float),
                      "offroad": (X["typ_samochodu"] == "offroad").to_numpy(float),
                      "miasto": (X["obszar_jazdy"] == "miasto").to_numpy(float)})
    auc = []
    for uczace, testowe in foldy(Z, y, grupy):
        m = LogisticRegression(max_iter=1000).fit(Z.iloc[uczace], y.iloc[uczace])
        auc.append(roc_auc_score(y.iloc[testowe], m.predict_proba(Z.iloc[testowe])[:, 1]))
    return float(np.mean(auc))


def analiza_kierowcow(df: pd.DataFrame, ryzyko: np.ndarray, ryzyko_stylu: np.ndarray) -> dict:
    """Zgodność ocen z prawdziwą agresywnością i wpływ obszaru jazdy na ocenę.

    Wpływ obszaru mierzymy regresją liniową oceny kierowcy względem jego
    prawdziwej agresywności oraz udziału wynajmów w mieście i na autostradzie.
    Współczynnik przy udziale miasta mówi, o ile punktów zmienia się ocena
    kierowcy jeżdżącego tylko po mieście w porównaniu z kierowcą o tej samej
    agresywności, który jeździ w obszarach mieszanych i na trasach.
    """
    walidacja = pd.read_csv(PLIK_WALIDACJI)
    t = df[["id_wynajmu", "id_kierowcy", "obszar_jazdy"]].assign(ryzyko=ryzyko, ryzyko_stylu=ryzyko_stylu)
    t = t.merge(walidacja[["id_wynajmu", "agresywnosc_prawdziwa"]], on="id_wynajmu")
    t["miasto"] = (t["obszar_jazdy"] == "miasto").astype(float)
    t["autostrada"] = (t["obszar_jazdy"] == "autostrada").astype(float)
    kierowcy = t.groupby("id_kierowcy")[["ryzyko", "ryzyko_stylu", "agresywnosc_prawdziwa",
                                         "miasto", "autostrada"]].mean()

    def wplyw_obszaru(kolumna_ryzyka: str) -> dict:
        srednie = kierowcy[kolumna_ryzyka].mean()
        ocena = [ocena_z_ryzyka(r, srednie) for r in kierowcy[kolumna_ryzyka]]
        reg = LinearRegression().fit(kierowcy[["agresywnosc_prawdziwa", "miasto", "autostrada"]], ocena)
        return {"tylko_miasto": round(float(reg.coef_[1]), 1),
                "tylko_autostrada": round(float(reg.coef_[2]), 1)}

    return {
        "spearman_wynajmy": float(spearmanr(t["ryzyko_stylu"], t["agresywnosc_prawdziwa"])[0]),
        # Ocena kierowcy to średnia z jego wynajmów, więc tak też ją sprawdzamy
        "spearman_kierowcy": float(spearmanr(kierowcy["ryzyko_stylu"], kierowcy["agresywnosc_prawdziwa"])[0]),
        "spearman_kierowcy_ryzyko_wynajmu": float(
            spearmanr(kierowcy["ryzyko"], kierowcy["agresywnosc_prawdziwa"])[0]),
        "wplyw_obszaru_ocena_ze_stylu": wplyw_obszaru("ryzyko_stylu"),
        "wplyw_obszaru_ocena_z_ryzyka_wynajmu": wplyw_obszaru("ryzyko"),
    }


def main():
    df = pd.read_csv(PLIK_DANYCH)
    X, y, grupy = zbuduj_cechy(df), zbuduj_etykiete(df), df["id_kierowcy"]

    wynik = walidacja_krzyzowa(X, y, grupy)
    auc = {k: float(v.mean()) for k, v in wynik["auc"].items()}
    kierowcy = analiza_kierowcow(df, wynik["ryzyko"], wynik["ryzyko_stylu"])
    sufit = auc_sufitu(df, X, y, grupy)

    model = ModelOceny().fit(X, y)
    KATALOG_MODELI.mkdir(exist_ok=True)
    joblib.dump(model, PLIK_MODELU)

    przewaga = auc["model"] - auc["baza"]
    metryki = {
        "model": "regresja logistyczna",
        "walidacja": f"GroupKFold po kierowcy, {LICZBA_FOLDOW} foldów",
        "ocena_kierowcy": "z ryzyka samego stylu jazdy (kontekst wynajmu taki sam dla wszystkich)",
        "liczba_wynajmow": int(len(df)),
        "liczba_kierowcow": int(grupy.nunique()),
        "odsetek_szkod": round(float(y.mean()), 4),
        "srednie_ryzyko_stylu": round(model.srednie_ryzyko_stylu, 4),
        "baza_auc": round(auc["baza"], 4),
        "baza_auc_std": round(float(wynik["auc"]["baza"].std()), 4),
        "model_auc": round(auc["model"], 4),
        "model_auc_std": round(float(wynik["auc"]["model"].std()), 4),
        "przewaga_nad_baza": round(przewaga, 4),
        "spearman_wynajmy": round(kierowcy["spearman_wynajmy"], 4),
        "spearman_kierowcy": round(kierowcy["spearman_kierowcy"], 4),
        "kryteria": {
            f"model_auc >= {PROG_AUC}": bool(auc["model"] >= PROG_AUC),
            f"przewaga_nad_baza >= {PROG_PRZEWAGI_NAD_BAZA}": bool(przewaga >= PROG_PRZEWAGI_NAD_BAZA),
            f"spearman_kierowcy >= {PROG_SPEARMAN}": bool(kierowcy["spearman_kierowcy"] >= PROG_SPEARMAN),
        },
        "analizy": {
            "auc_baza_wzgledem_obszaru": round(auc["baza_wzgledna"], 4),
            "auc_ryzyko_stylu": round(auc["ryzyko_stylu"], 4),
            "auc_tylko_kontekst": round(auc["tylko_kontekst"], 4),
            "auc_sufit_prawdziwa_agresywnosc": round(sufit, 4),
            "spearman_kierowcy_ryzyko_wynajmu": round(kierowcy["spearman_kierowcy_ryzyko_wynajmu"], 4),
            "wplyw_obszaru_na_ocene_pkt": {
                "ocena_ze_stylu": kierowcy["wplyw_obszaru_ocena_ze_stylu"],
                "gdyby_ocena_z_ryzyka_wynajmu": kierowcy["wplyw_obszaru_ocena_z_ryzyka_wynajmu"],
            },
        },
        "wspolczynniki": {k: round(v, 4) for k, v in model.wspolczynniki().items()},
    }
    PLIK_METRYK.write_text(json.dumps(metryki, ensure_ascii=False, indent=2), encoding="utf-8")

    a = metryki["analizy"]
    wplyw = a["wplyw_obszaru_na_ocene_pkt"]
    print(f"Dane: {metryki['liczba_wynajmow']} wynajmów, {metryki['liczba_kierowcow']} kierowców, "
          f"szkody w {metryki['odsetek_szkod']:.1%} wynajmów")
    print(f"Walidacja: {metryki['walidacja']}")
    print(f"  Metoda bazowa  ROC AUC = {auc['baza']:.3f} ± {metryki['baza_auc_std']:.3f}")
    print(f"  Model          ROC AUC = {auc['model']:.3f} ± {metryki['model_auc_std']:.3f}")
    print(f"  Przewaga nad bazą      = {przewaga:+.3f}")
    print(f"  Spearman (kierowcy)    = {kierowcy['spearman_kierowcy']:.3f}  (ocena ze stylu jazdy)")
    print("Kryteria sukcesu z karty:")
    for nazwa, spelnione in metryki["kryteria"].items():
        print(f"  [{'OK' if spelnione else 'NIE'}] {nazwa}")
    print("Co model wnosi ponad reguły z generatora (ROC AUC):")
    print(f"  tylko kontekst (przebieg, obszar, auto) = {a['auc_tylko_kontekst']:.3f}")
    print(f"  baza względem obszaru                  = {a['auc_baza_wzgledem_obszaru']:.3f}")
    print(f"  model                                  = {auc['model']:.3f}")
    print(f"  sufit (prawdziwa agresywność)          = {a['auc_sufit_prawdziwa_agresywnosc']:.3f}")
    print("Wpływ obszaru na ocenę kierowcy o tej samej agresywności (pkt):")
    print(f"  ocena ze stylu:            miasto {wplyw['ocena_ze_stylu']['tylko_miasto']:+.1f}, "
          f"autostrada {wplyw['ocena_ze_stylu']['tylko_autostrada']:+.1f}")
    print(f"  gdyby z ryzyka wynajmu:    miasto {wplyw['gdyby_ocena_z_ryzyka_wynajmu']['tylko_miasto']:+.1f}, "
          f"autostrada {wplyw['gdyby_ocena_z_ryzyka_wynajmu']['tylko_autostrada']:+.1f}")
    print(f"Zapisano model: {PLIK_MODELU.relative_to(KATALOG_GLOWNY)}, "
          f"metryki: {PLIK_METRYK.relative_to(KATALOG_GLOWNY)}")


if __name__ == "__main__":
    main()
