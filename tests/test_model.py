"""Ocena kierowcy zależy od stylu jazdy, a nie od obszaru, przebiegu czy auta."""
import numpy as np
import pandas as pd
import pytest

from src.model.features import CECHY_STYLU, MIARY_JAZDY, CechyWzgledemObszaru, zbuduj_cechy


def test_cechy_wzgledne_maja_srednia_zero_w_kazdym_obszarze(wynajmy):
    X = zbuduj_cechy(wynajmy)
    wzgl = CechyWzgledemObszaru().fit(X).transform(X)
    srednie = wzgl.groupby("obszar_jazdy")[CECHY_STYLU].mean()
    assert np.allclose(srednie.to_numpy(), 0, atol=1e-9)


def test_nieznany_obszar_porownywany_z_cala_flota(wynajmy):
    X = zbuduj_cechy(wynajmy)
    transformacja = CechyWzgledemObszaru().fit(X)
    nowy = X.head(3).assign(obszar_jazdy="tor wyścigowy")
    wynik = transformacja.transform(nowy)
    assert wynik[CECHY_STYLU].notna().all().all()


def typowy_kierowca(model, obszar: str) -> pd.DataFrame:
    """Wynajem kierowcy, który jedzie dokładnie jak przeciętny kierowca w danym obszarze."""
    t = model.pipeline.named_steps["wzgledem_obszaru"]
    srednie = t.srednie_[obszar]
    wiersz = {m: srednie[m] for m in MIARY_JAZDY}
    wiersz["przyspieszenia_na_100km"] = np.expm1(wiersz["przyspieszenia_na_100km"])
    wiersz.update(log_km_razem=np.log(300), liczba_koni_mechanicznych=150,
                  obszar_jazdy=obszar, typ_samochodu="osobowy")
    return pd.DataFrame([wiersz])


def test_typowy_kierowca_ma_te_sama_ocene_w_kazdym_obszarze(model):
    ryzyka = [model.ryzyko_stylu(typowy_kierowca(model, o))[0]
              for o in ("miasto", "trasa", "autostrada", "mieszany")]
    assert max(ryzyka) == pytest.approx(min(ryzyka))


def test_auto_i_przebieg_nie_zmieniaja_ryzyka_stylu(model, wynajmy):
    X = zbuduj_cechy(wynajmy.head(50))
    inne = X.assign(typ_samochodu="sportowy", liczba_koni_mechanicznych=500, log_km_razem=np.log(5000))
    assert np.allclose(model.ryzyko_stylu(X), model.ryzyko_stylu(inne))
    # Pełne ryzyko wynajmu nadal uwzględnia kontekst
    assert not np.allclose(model.ryzyko_wynajmu(inne), model.ryzyko_wynajmu(X))


def test_agresywniejsza_jazda_podnosi_ryzyko_stylu(model, wynajmy):
    X = zbuduj_cechy(wynajmy.head(50))
    gorzej = X.assign(stosunek_spalania=X["stosunek_spalania"] * 1.3,
                      srednia_predkosc_kmh=X["srednia_predkosc_kmh"] * 1.2,
                      najwyzsza_predkosc_kmh=X["najwyzsza_predkosc_kmh"] * 1.2,
                      przyspieszenia_na_100km=X["przyspieszenia_na_100km"] * 3 + 1)
    assert (model.ryzyko_stylu(gorzej) > model.ryzyko_stylu(X)).all()
