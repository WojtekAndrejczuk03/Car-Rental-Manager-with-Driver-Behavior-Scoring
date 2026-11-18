# Car Rental Manager with Driver Behavior Scoring

Projekt wypożyczalni samochodów, który analizuje dane z wynajmów i przewiduje
ryzyko uszkodzenia auta. Na podstawie przewidywanego ryzyka oblicza ocenę
kierowcy w skali 0–100, przypisuje styl jazdy i proponuje cenę kolejnego wynajmu.

Panel zarządcy pokazuje podsumowanie, porównanie modelu z metodą bazową,
listę kierowców oraz profil wybranego kierowcy z historią wynajmów i wyceną.
Panel komunikuje się z API FastAPI, a wyniki ocen pochodzą z wytrenowanego
modelu.

## Dane

Dane w `data/wynajmy.csv` są syntetyczne. Jeden wiersz opisuje jeden wynajem.
`data/walidacja_agresywnosc.csv` zawiera ukrytą agresywność kierowców i służy
wyłącznie do sprawdzenia jakości ocen — nie jest używany do uczenia modelu.
Więcej informacji znajduje się w [data/README.md](data/README.md).

## Wymagania

- Python 3.9 lub nowszy
- Dostęp do internetu podczas pierwszego otwarcia panelu (Vue jest ładowane z CDN)

## Uruchomienie w Windows (PowerShell)

W katalogu głównym repozytorium wykonaj:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/import_csv.py
python -m src.model.train
uvicorn src.api:app --reload
```

Jeśli PowerShell blokuje aktywację środowiska, uruchom polecenia przez
`.\.venv\Scripts\python.exe` oraz `.\.venv\Scripts\uvicorn.exe` zamiast
aktywować `.venv`.

## Uruchomienie w macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/import_csv.py
python -m src.model.train
uvicorn src.api:app --reload
```

Import tworzy lokalną bazę `data/wynajmy.sqlite3`. Trening zapisuje
`models/model.joblib` i `models/metryki.json`. Serwer API potrzebuje tych
plików oraz bazy, dlatego wykonaj import i trening przed jego uruchomieniem.

Po uruchomieniu serwera pozostaw terminal otwarty i odwiedź:

- Panel zarządcy: <http://127.0.0.1:8000/>
- Interaktywna dokumentacja API: <http://127.0.0.1:8000/docs>

Nie otwieraj `frontend/index.html` bezpośrednio z dysku — panel pobiera dane
z API serwowanego przez FastAPI.

## Najważniejsze endpointy

- `GET /api/podsumowanie` — dane do kafelków panelu.
- `GET /api/metryki` — wyniki modelu i metody bazowej.
- `GET /api/kierowcy` — oceny i style jazdy kierowców.
- `GET /api/kierowcy/{id_kierowcy}` — profil kierowcy i historia wynajmów.
- `POST /api/wycena` — wycena dla kierowcy i stawki bazowej.
