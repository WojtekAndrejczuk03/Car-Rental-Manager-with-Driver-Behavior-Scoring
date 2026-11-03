#!/usr/bin/env python3
"""Generator syntetycznego zbioru danych o wynajmach samochodów.

Jeden wiersz = jeden wynajem. Każdy kierowca ma ukryty poziom agresywności
(0 = bardzo spokojny, 1 = bardzo agresywny), z którego wynikają obserwowane
cechy jazdy: spalanie, gwałtowne przyspieszenia, prędkości i ryzyko szkody.
Styl jazdy NIE jest zapisywany w danych, bo ma go wyliczać model.
Prawdziwa agresywność trafia do osobnego pliku i służy wyłącznie do
sprawdzenia, czy ocena wyliczona przez model ma sens.
Poszczególne cechy zależą od agresywności w różnym stopniu i są zaszumione,
więc żadna pojedyncza cecha nie wyznacza stylu jazdy jednoznacznie.

Użycie:
    python3 scripts/generate_dataset.py --wynajmy 3000 --seed 42 --out data/wynajmy.csv
"""
import argparse

import numpy as np
import pandas as pd

# Przybliżone dane katalogowe:
# marka i model, typ, klasa (segment europejski), KM, spalanie producenta [l/100 km], Vmax [km/h]
SAMOCHODY = [
    ("Toyota Aygo X", "miejski", "A", 72, 4.9, 158),
    ("Fiat 500", "miejski", "A", 70, 5.3, 167),
    ("Toyota Yaris", "miejski", "B", 116, 4.0, 175),
    ("Skoda Fabia", "miejski", "B", 95, 5.0, 195),
    ("Opel Corsa", "miejski", "B", 100, 5.2, 194),
    ("Toyota Corolla", "osobowy", "C", 140, 4.5, 180),
    ("Volkswagen Golf", "osobowy", "C", 150, 5.4, 224),
    ("Skoda Octavia Combi", "kombi", "C", 150, 5.6, 224),
    ("Kia Ceed SW", "kombi", "C", 140, 5.9, 205),
    ("Toyota Camry", "osobowy", "D", 218, 5.0, 180),
    ("Volkswagen Passat Variant", "kombi", "D", 150, 5.6, 220),
    ("BMW 520d", "osobowy", "E", 197, 5.3, 235),
    ("Mercedes-Benz E 220d", "osobowy", "E", 200, 5.2, 240),
    ("Mercedes-Benz S 450", "osobowy", "F", 367, 8.6, 250),
    ("BMW 740i", "osobowy", "F", 380, 8.4, 250),
    ("Toyota RAV4", "SUV", "J", 222, 5.6, 180),
    ("Kia Sportage", "SUV", "J", 150, 6.6, 190),
    ("Hyundai Tucson", "SUV", "J", 150, 6.4, 191),
    ("Suzuki Jimny", "offroad", "J", 102, 7.7, 145),
    ("Jeep Wrangler", "offroad", "J", 272, 11.0, 180),
    ("Toyota Land Cruiser", "offroad", "J", 204, 9.6, 175),
    ("Mazda MX-5", "sportowy", "S", 184, 6.9, 219),
    ("Toyota GR86", "sportowy", "S", 234, 9.0, 226),
    ("Ford Mustang GT", "sportowy", "S", 446, 12.4, 250),
    ("BMW M4", "sportowy", "S", 510, 10.2, 250),
    ("Porsche 911 Carrera", "sportowy", "S", 385, 10.1, 293),
]

# Typowe wartości dla obszaru jazdy
OBSZARY = {
    #             udział, km/dzień (średnia, odch.), średnia prędkość, limit, gwałtowne przysp./100 km, mnożnik spalania
    "miasto":     (0.35, 45, 20, 27, 50, 6.0, 1.18),
    "trasa":      (0.25, 160, 60, 63, 90, 2.0, 0.95),
    "autostrada": (0.15, 280, 100, 108, 140, 0.8, 1.08),
    "mieszany":   (0.25, 110, 45, 48, 90, 3.5, 1.05),
}


def generuj(n_wynajmow: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_kierowcow = max(1, n_wynajmow // 5)

    # Kierowcy: ukryta agresywność i indywidualne odchylenia poszczególnych nawyków
    agresywnosc = rng.beta(2.0, 2.2, n_kierowcow)
    odch = {k: rng.normal(0, 0.12, n_kierowcow) for k in ("przysp", "predk", "spal")}
    sklonnosc_szkody = rng.normal(0, 0.35, n_kierowcow)

    samochody = pd.DataFrame(SAMOCHODY, columns=["marka_i_model", "typ_samochodu", "klasa_pojazdu",
                                                 "liczba_koni_mechanicznych", "spalanie_producenta_l_100km",
                                                 "vmax"])
    sportowe = samochody.index[samochody.typ_samochodu == "sportowy"].to_numpy()

    obszary = list(OBSZARY)
    p_obszar = np.array([OBSZARY[o][0] for o in obszary])

    wiersze = []
    for i in range(n_wynajmow):
        k = rng.integers(n_kierowcow)
        a = float(np.clip(agresywnosc[k] + rng.normal(0, 0.07), 0, 1))  # forma danego wynajmu

        # Agresywni kierowcy nieco częściej wybierają auta sportowe
        if rng.random() < 0.05 + 0.25 * a:
            s = samochody.loc[rng.choice(sportowe)]
        else:
            s = samochody.loc[rng.integers(len(samochody))]

        typ = s.typ_samochodu
        p = p_obszar.copy()
        if typ == "offroad":
            p = p * np.array([0.6, 1.6, 0.6, 1.2])
        p /= p.sum()
        obszar = obszary[rng.choice(len(obszary), p=p)]
        _, km_sr, km_sd, v_sr, limit, przysp_baza, mn_spal = OBSZARY[obszar]

        dni = int(min(21, rng.geometric(0.25)))
        km_dzien = max(5.0, rng.normal(km_sr * (0.9 + 0.2 * a), km_sd))
        km_razem = km_dzien * dni

        c_przysp = np.clip(a + odch["przysp"][k] + rng.normal(0, 0.06), 0, 1)
        c_predk = np.clip(a + odch["predk"][k] + rng.normal(0, 0.06), 0, 1)
        c_spal = np.clip(a + odch["spal"][k] + rng.normal(0, 0.06), 0, 1)
        moc = s.liczba_koni_mechanicznych / 150

        # Prędkości
        korek = rng.uniform(0.75, 1.1)  # natężenie ruchu niezależne od kierowcy
        v_srednia = v_sr * korek * (0.88 + 0.24 * c_predk) + rng.normal(0, 3)
        v_srednia = float(np.clip(v_srednia, 12, s.vmax * 0.8))
        v_max = limit * (1.05 + 0.45 * c_predk) + 6 * np.log(moc) * c_predk + rng.normal(0, 8)
        v_max = float(np.clip(max(v_max, v_srednia * 1.4), 0, s.vmax))

        # Gwałtowne przyspieszenia: zależą od stylu, obszaru i mocy auta
        intensywnosc = przysp_baza * np.exp(2.4 * (c_przysp - 0.5)) * moc ** 0.35 * rng.lognormal(0, 0.25)
        n_przysp = int(rng.poisson(intensywnosc * km_razem / 100))

        # Spalanie rzeczywiste
        spal = s.spalanie_producenta_l_100km * mn_spal * 1.08 * (0.88 + 0.38 * c_spal)
        if obszar == "autostrada":
            spal *= 1 + 0.006 * max(0.0, v_srednia - 110)
        spal *= rng.normal(1, 0.05)
        spal = float(max(spal, 0.75 * s.spalanie_producenta_l_100km))

        # Szkoda: zależy od stylu jazdy, typu auta, przebiegu i indywidualnej skłonności
        logit = (-3.2 + 4.5 * (a - 0.5) + sklonnosc_szkody[k]
                 + 0.45 * np.log(km_razem / 300)
                 + {"sportowy": 0.5, "offroad": 0.4}.get(typ, 0.0)
                 + (0.3 if obszar == "miasto" else 0.0))
        szkoda = rng.random() < 1 / (1 + np.exp(-logit))

        wiersze.append({
            "id_wynajmu": i + 1,
            "id_kierowcy": int(k + 1),
            "marka_i_model": s.marka_i_model,
            "typ_samochodu": typ,
            "klasa_pojazdu": s.klasa_pojazdu,
            "liczba_koni_mechanicznych": int(s.liczba_koni_mechanicznych),
            "spalanie_producenta_l_100km": float(s.spalanie_producenta_l_100km),
            "spalanie_kierowcy_l_100km": round(spal, 1),
            "liczba_gwaltownych_przyspieszen": n_przysp,
            "srednia_predkosc_kmh": round(v_srednia, 1),
            "najwyzsza_predkosc_kmh": round(v_max, 1),
            "obszar_jazdy": obszar,
            "km_na_dzien": round(km_dzien, 1),
            "liczba_dni_wynajmu": dni,
            "uszkodzenie_samochodu": "tak" if szkoda else "nie",
            "_agresywnosc_prawdziwa": round(a, 3),
        })
    return pd.DataFrame(wiersze)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wynajmy", type=int, default=3000, help="liczba wynajmów (wierszy)")
    ap.add_argument("--seed", type=int, default=42, help="ziarno losowania")
    ap.add_argument("--out", default="data/wynajmy.csv", help="ścieżka pliku wynikowego")
    ap.add_argument("--out-ukryte", default="data/walidacja_agresywnosc.csv",
                    help="plik z prawdziwą agresywnością (tylko do walidacji, nie do uczenia)")
    args = ap.parse_args()
    df = generuj(args.wynajmy, args.seed)
    df.drop(columns="_agresywnosc_prawdziwa").to_csv(args.out, index=False)
    df[["id_wynajmu", "id_kierowcy", "_agresywnosc_prawdziwa"]].rename(
        columns={"_agresywnosc_prawdziwa": "agresywnosc_prawdziwa"}).to_csv(args.out_ukryte, index=False)
    print(f"Zapisano {len(df)} wierszy do {args.out} oraz {args.out_ukryte}")


if __name__ == "__main__":
    main()
