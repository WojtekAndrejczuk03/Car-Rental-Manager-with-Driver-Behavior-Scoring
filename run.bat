@echo off
rem Uruchamia caly system od zera na Windowsie: srodowisko, baza, model, API z panelem.
rem Uzycie: dwuklik na run.bat albo w terminalu:  run.bat
rem Inny port:  set PORT=8080 ^& run.bat
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if "%PORT%"=="" set PORT=8000
set PY=.venv\Scripts\python.exe

echo == 1/4 Srodowisko Pythona .venv
if exist "%PY%" goto venv_jest
where py >nul 2>nul
if errorlevel 1 (python -m venv .venv) else (py -3 -m venv .venv)
if errorlevel 1 goto blad
"%PY%" -m pip install --quiet --upgrade pip
if errorlevel 1 goto blad
"%PY%" -m pip install --quiet -r requirements.txt
if errorlevel 1 goto blad
echo Utworzono .venv i zainstalowano biblioteki z requirements.txt
goto import
:venv_jest
echo .venv juz istnieje - po zmianie requirements.txt: .venv\Scripts\python.exe -m pip install -r requirements.txt

:import
echo == 2/4 Import wynajmow z CSV do bazy SQLite
"%PY%" scripts\import_csv.py
if errorlevel 1 goto blad

echo == 3/4 Uczenie modelu i porownanie z metoda bazowa
"%PY%" -m src.model.train
if errorlevel 1 goto blad

echo == 4/4 Start API i panelu: http://127.0.0.1:%PORT%/   dokumentacja API: /docs, zatrzymanie: Ctrl+C
"%PY%" -m uvicorn src.api:app --port %PORT%
goto koniec

:blad
echo.
echo BLAD: krok sie nie powiodl, szczegoly powyzej.
pause
exit /b 1

:koniec
endlocal
