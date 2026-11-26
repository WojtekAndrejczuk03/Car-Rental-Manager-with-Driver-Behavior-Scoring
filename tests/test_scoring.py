"""Skala oceny 0–100, progi stylu jazdy i formuła wyceny."""
import pytest

from src.model.scoring import MNOZNIKI, ocena_z_ryzyka, styl_jazdy, wycena

SREDNIE = 0.06


@pytest.mark.parametrize("ryzyko, ocena", [
    (SREDNIE, 50),        # przeciętny kierowca
    (SREDNIE / 2, 75),    # 2× mniejsze ryzyko
    (SREDNIE * 2, 25),    # 2× większe ryzyko
    (SREDNIE / 4, 100),   # 4× mniejsze ryzyko
    (SREDNIE * 4, 0),     # 4× większe ryzyko
    (SREDNIE / 100, 100), # skala ucięta do 0–100
    (0.99, 0),
    (0.0, 100),           # zerowe ryzyko nie powoduje błędu logarytmu
])
def test_skala_oceny(ryzyko, ocena):
    assert ocena_z_ryzyka(ryzyko, SREDNIE) == ocena


def test_ocena_maleje_z_ryzykiem():
    oceny = [ocena_z_ryzyka(r / 1000, SREDNIE) for r in range(1, 500, 7)]
    assert oceny == sorted(oceny, reverse=True)


@pytest.mark.parametrize("ocena, styl", [
    (100, "spokojny"), (65, "spokojny"),
    (64, "normalny"), (35, "normalny"),
    (34, "agresywny"), (0, "agresywny"),
    (None, "normalny"),  # nowy kierowca bez historii
])
def test_progi_stylu(ocena, styl):
    assert styl_jazdy(ocena) == styl


def test_wycena_spokojnego_kierowcy():
    w = wycena(200, 80, 5)
    assert w["styl"] == "spokojny"
    assert w["cena"] == pytest.approx(200 * MNOZNIKI["spokojny"])
    assert "zniżka 10%" in w["uzasadnienie"]


def test_wycena_agresywnego_kierowcy():
    w = wycena(200, 10, 3)
    assert w["styl"] == "agresywny"
    assert w["cena"] == pytest.approx(250.0)
    assert "dopłata 25%" in w["uzasadnienie"]


def test_wycena_nowego_kierowcy():
    w = wycena(199.99, None)
    assert w["cena"] == 199.99
    assert w["mnoznik"] == 1.0
    assert "Nowy kierowca" in w["uzasadnienie"]


def test_cena_zaokraglona_do_groszy():
    assert wycena(123.456, 80, 1)["cena"] == 111.11
