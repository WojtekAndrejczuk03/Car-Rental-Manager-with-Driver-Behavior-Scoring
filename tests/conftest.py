"""Wspólne dane do testów: model uczony raz na data/wynajmy.csv i tymczasowa baza."""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scripts.import_csv import import_csv
from src.model.features import zbuduj_cechy, zbuduj_etykiete
from src.model.model import ModelOceny

KATALOG_GLOWNY = Path(__file__).resolve().parents[1]
PLIK_DANYCH = KATALOG_GLOWNY / "data" / "wynajmy.csv"


@pytest.fixture(scope="session")
def wynajmy() -> pd.DataFrame:
    return pd.read_csv(PLIK_DANYCH)


@pytest.fixture(scope="session")
def model(wynajmy) -> ModelOceny:
    return ModelOceny().fit(zbuduj_cechy(wynajmy), zbuduj_etykiete(wynajmy))


@pytest.fixture
def klient(tmp_path, monkeypatch, model):
    """Klient API na kopii bazy w katalogu tymczasowym (prawdziwa baza zostaje nietknięta)."""
    import src.api as api

    baza = tmp_path / "wynajmy.sqlite3"
    import_csv(PLIK_DANYCH, baza)
    metryki = tmp_path / "metryki.json"
    metryki.write_text(json.dumps({"model_auc": 0.73}), encoding="utf-8")

    monkeypatch.setattr(api, "DEFAULT_DB_PATH", baza)
    monkeypatch.setattr(api, "PLIK_METRYK", metryki)
    monkeypatch.setattr(api, "wczytaj_model", lambda: model)
    with TestClient(api.app) as c:
        yield c
