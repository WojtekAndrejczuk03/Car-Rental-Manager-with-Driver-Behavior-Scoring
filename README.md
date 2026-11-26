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

## Szybki start

Jedna komenda tworzy środowisko, importuje dane do bazy, uczy model i startuje
API z panelem:

- **Windows:** dwuklik na `run.bat` albo w PowerShellu `.\run.bat`
- **macOS / Linux:** `./run.sh`

Pierwsze uruchomienie trwa 1–2 minuty, bo instaluje biblioteki. Potem otwórz
<http://127.0.0.1:8000/>. Serwer zatrzymasz klawiszami Ctrl+C. Gdy port 8000 jest
zajęty: `$env:PORT=8080; .\run.bat` (PowerShell) albo `PORT=8080 ./run.sh`.

Te same kroki można wykonać ręcznie, jak opisano niżej.

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
- `GET /api/samochody` — flota: dane aut, liczba wynajmów i szkód, średnia ocena kierowców.
- `POST /api/import` — import wynajmów z pliku CSV (treść pliku w ciele zapytania,
  kolumny jak w `data/wynajmy.csv`). W panelu: zakładka „Import CSV”.

## Testy

```bash
python -m pip install -r requirements.txt
python -m pytest
```

Testy sprawdzają skalę oceny, progi stylu jazdy, wycenę, endpointy API
(na tymczasowej kopii bazy) oraz to, że obszar jazdy, przebieg i auto
nie zmieniają oceny kierowcy.

## Ocena kierowcy zależy tylko od stylu jazdy

Model przewiduje szkodę z dwóch grup cech:

- **styl jazdy**: przyspieszenia na 100 km, spalanie względem katalogowego,
  prędkość średnia i najwyższa. Każdą z tych miar porównujemy z typowym
  kierowcą w tym samym obszarze, bo 130 km/h na autostradzie to spokojna jazda,
  a w mieście bardzo agresywna;
- **kontekst wynajmu**: przebieg, obszar jazdy, typ i moc auta.

Kontekst pomaga przewidzieć szkodę, ale nie powinien zmieniać ceny dla klienta.
Dlatego ocenę 0–100 liczymy z ryzyka, jakie kierowca miałby przy tym samym
stylu jazdy w jednakowym, referencyjnym kontekście (ten sam przebieg, obszar
i auto dla wszystkich). Pole `ryzyko` w API to nadal pełne ryzyko szkody
w wynajmach kierowcy, a `ryzyko_stylu` i `ocena` opisują tylko jego styl.

Trening (`python -m src.model.train`) zapisuje w `models/metryki.json`
w sekcji `analizy`:

- wpływ obszaru jazdy na ocenę kierowcy o tej samej prawdziwej agresywności
  (obecnie ok. 0 pkt; gdyby ocena wynikała z pełnego ryzyka, kierowca jeżdżący
  tylko po mieście dostawałby ok. +13 pkt, a jeżdżący autostradą ok. −6 pkt),
- ROC AUC modelu, który zna tylko kontekst wpisany do generatora (0,64), oraz
  „sufitu”, czyli modelu znającego prawdziwą agresywność (0,76). Model osiąga
  0,73, więc większość przewagi nad samym kontekstem pochodzi z odczytania
  stylu jazdy z danych.
