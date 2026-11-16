#!/usr/bin/env python3
"""Import rental rows from data/wynajmy.csv into the SQLite database."""

import argparse
import csv
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db import DEFAULT_DB_PATH, connect_database, initialize_database

DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "wynajmy.csv"
EXPECTED_COLUMNS = (
    "id_wynajmu",
    "id_kierowcy",
    "marka_i_model",
    "typ_samochodu",
    "klasa_pojazdu",
    "liczba_koni_mechanicznych",
    "spalanie_producenta_l_100km",
    "spalanie_kierowcy_l_100km",
    "liczba_gwaltownych_przyspieszen",
    "srednia_predkosc_kmh",
    "najwyzsza_predkosc_kmh",
    "obszar_jazdy",
    "km_na_dzien",
    "liczba_dni_wynajmu",
    "uszkodzenie_samochodu",
)

INSERT_CAR = """
INSERT INTO samochody (
    marka_i_model,
    typ_samochodu,
    klasa_pojazdu,
    liczba_koni_mechanicznych,
    spalanie_producenta_l_100km
)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT (
    marka_i_model,
    typ_samochodu,
    klasa_pojazdu,
    liczba_koni_mechanicznych,
    spalanie_producenta_l_100km
) DO NOTHING
"""

UPSERT_RENTAL = """
INSERT INTO wynajmy (
    id_wynajmu,
    id_kierowcy,
    id_samochodu,
    spalanie_kierowcy_l_100km,
    liczba_gwaltownych_przyspieszen,
    srednia_predkosc_kmh,
    najwyzsza_predkosc_kmh,
    obszar_jazdy,
    km_na_dzien,
    liczba_dni_wynajmu,
    uszkodzenie_samochodu
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (id_wynajmu) DO UPDATE SET
    id_kierowcy = excluded.id_kierowcy,
    id_samochodu = excluded.id_samochodu,
    spalanie_kierowcy_l_100km = excluded.spalanie_kierowcy_l_100km,
    liczba_gwaltownych_przyspieszen =
        excluded.liczba_gwaltownych_przyspieszen,
    srednia_predkosc_kmh = excluded.srednia_predkosc_kmh,
    najwyzsza_predkosc_kmh = excluded.najwyzsza_predkosc_kmh,
    obszar_jazdy = excluded.obszar_jazdy,
    km_na_dzien = excluded.km_na_dzien,
    liczba_dni_wynajmu = excluded.liczba_dni_wynajmu,
    uszkodzenie_samochodu = excluded.uszkodzenie_samochodu
"""


def _integer(row: dict[str, str], column: str, line_number: int) -> int:
    try:
        return int(row[column])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Wiersz {line_number}: kolumna {column!r} musi być liczbą całkowitą."
        ) from error


def _number(row: dict[str, str], column: str, line_number: int) -> float:
    try:
        value = float(row[column])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Wiersz {line_number}: kolumna {column!r} musi być liczbą."
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"Wiersz {line_number}: kolumna {column!r} musi być skończoną liczbą."
        )
    return value


def _import_row(connection, row: dict[str, str], line_number: int) -> None:
    if None in row:
        raise ValueError(
            f"Wiersz {line_number}: znaleziono więcej pól niż nagłówków CSV."
        )

    rental_id = _integer(row, "id_wynajmu", line_number)
    driver_id = _integer(row, "id_kierowcy", line_number)
    car = (
        row["marka_i_model"].strip(),
        row["typ_samochodu"].strip(),
        row["klasa_pojazdu"].strip(),
        _integer(row, "liczba_koni_mechanicznych", line_number),
        _number(row, "spalanie_producenta_l_100km", line_number),
    )
    if not all(car[:3]):
        raise ValueError(f"Wiersz {line_number}: dane samochodu nie mogą być puste.")

    connection.execute(
        "INSERT OR IGNORE INTO kierowcy (id_kierowcy) VALUES (?)",
        (driver_id,),
    )
    connection.execute(INSERT_CAR, car)
    car_row = connection.execute(
        """
        SELECT id_samochodu FROM samochody
        WHERE marka_i_model = ?
          AND typ_samochodu = ?
          AND klasa_pojazdu = ?
          AND liczba_koni_mechanicznych = ?
          AND spalanie_producenta_l_100km = ?
        """,
        car,
    ).fetchone()
    if car_row is None:
        raise RuntimeError(f"Wiersz {line_number}: nie udało się zapisać samochodu.")

    damage = row["uszkodzenie_samochodu"].strip().lower()
    if damage not in ("tak", "nie"):
        raise ValueError(
            f"Wiersz {line_number}: uszkodzenie_samochodu musi mieć wartość 'tak' lub 'nie'."
        )

    connection.execute(
        UPSERT_RENTAL,
        (
            rental_id,
            driver_id,
            car_row["id_samochodu"],
            _number(row, "spalanie_kierowcy_l_100km", line_number),
            _integer(row, "liczba_gwaltownych_przyspieszen", line_number),
            _number(row, "srednia_predkosc_kmh", line_number),
            _number(row, "najwyzsza_predkosc_kmh", line_number),
            row["obszar_jazdy"].strip(),
            _number(row, "km_na_dzien", line_number),
            _integer(row, "liczba_dni_wynajmu", line_number),
            damage,
        ),
    )


def import_csv(csv_path: Path, db_path: Path) -> int:
    """Validate and import a CSV file atomically; return its rental row count."""
    connection = connect_database(db_path)
    try:
        initialize_database(connection)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError("Plik CSV jest pusty albo nie zawiera nagłówka.")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("Nagłówek CSV zawiera powtórzone nazwy kolumn.")

            missing = set(EXPECTED_COLUMNS) - set(reader.fieldnames)
            extra = set(reader.fieldnames) - set(EXPECTED_COLUMNS)
            if missing or extra:
                raise ValueError(
                    "Nieprawidłowe kolumny CSV. "
                    f"Brakujące: {sorted(missing)}; nieoczekiwane: {sorted(extra)}."
                )

            imported_rows = 0
            with connection:
                for row in reader:
                    _import_row(connection, row, reader.line_num)
                    imported_rows += 1
            return imported_rows
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="ścieżka do pliku CSV (domyślnie data/wynajmy.csv)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="ścieżka do bazy SQLite (domyślnie data/wynajmy.sqlite3)",
    )
    args = parser.parse_args()

    imported_rows = import_csv(args.csv, args.db)
    print(f"Zaimportowano {imported_rows} wynajmów do {args.db}.")


if __name__ == "__main__":
    main()
