"""SQLite schema and connection helpers for the rental dataset."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "wynajmy.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS kierowcy (
    id_kierowcy INTEGER PRIMARY KEY CHECK (id_kierowcy > 0)
);

CREATE TABLE IF NOT EXISTS samochody (
    id_samochodu INTEGER PRIMARY KEY,
    marka_i_model TEXT NOT NULL,
    typ_samochodu TEXT NOT NULL,
    klasa_pojazdu TEXT NOT NULL,
    liczba_koni_mechanicznych INTEGER NOT NULL
        CHECK (liczba_koni_mechanicznych > 0),
    spalanie_producenta_l_100km REAL NOT NULL
        CHECK (spalanie_producenta_l_100km >= 0),
    UNIQUE (
        marka_i_model,
        typ_samochodu,
        klasa_pojazdu,
        liczba_koni_mechanicznych,
        spalanie_producenta_l_100km
    )
);

CREATE TABLE IF NOT EXISTS wynajmy (
    id_wynajmu INTEGER PRIMARY KEY CHECK (id_wynajmu > 0),
    id_kierowcy INTEGER NOT NULL
        REFERENCES kierowcy (id_kierowcy),
    id_samochodu INTEGER NOT NULL
        REFERENCES samochody (id_samochodu),
    spalanie_kierowcy_l_100km REAL NOT NULL
        CHECK (spalanie_kierowcy_l_100km >= 0),
    liczba_gwaltownych_przyspieszen INTEGER NOT NULL
        CHECK (liczba_gwaltownych_przyspieszen >= 0),
    srednia_predkosc_kmh REAL NOT NULL
        CHECK (srednia_predkosc_kmh >= 0),
    najwyzsza_predkosc_kmh REAL NOT NULL
        CHECK (najwyzsza_predkosc_kmh >= 0),
    obszar_jazdy TEXT NOT NULL,
    km_na_dzien REAL NOT NULL CHECK (km_na_dzien > 0),
    liczba_dni_wynajmu INTEGER NOT NULL CHECK (liczba_dni_wynajmu > 0),
    uszkodzenie_samochodu TEXT NOT NULL
        CHECK (uszkodzenie_samochodu IN ('tak', 'nie'))
);

CREATE INDEX IF NOT EXISTS idx_wynajmy_id_kierowcy
    ON wynajmy (id_kierowcy);
"""


def connect_database(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> sqlite3.Connection:
    """Open a database connection with foreign-key checks enabled."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    """Create the tables and indexes if they do not already exist."""
    connection.executescript(SCHEMA)
