# Karta zgłoszenia tematu projektu

**Grupa:** L2 · **Zespół:** Z5 (Drużyna R) · **Data:** 22.10.2026
Sztuczna inteligencja — laboratorium · Collegium Da Vinci · rok akad. 2026/2027

## 1. Skład zespołu

| Imię i nazwisko | Nr albumu | Rola (wstępna) |
|---|---|---|
| Wojciech Andrejczuk | 29202 | CEO |
| Błażej Kłoda | 31132 | Szeregowy |

## 2. Roboczy tytuł (PL)

Menedżer wypożyczalni samochodów z prognozą popytu

## 3. Tytuł (EN)

Car Rental Manager with Demand Forecasting

## 4. Problem — co system ma robić i dla kogo

Zarządcy małych wypożyczalni samochodów muszą z wyprzedzeniem decydować, ile aut danej klasy trzymać w gotowości i kiedy planować przeglądy. Robią to dziś na wyczucie. System zbiera dane o wypożyczeniach w jednym miejscu i prognozuje, ile aut danej klasy zostanie wypożyczonych w kolejnych dniach. Dzięki temu zarządca może zawczasu planować dostępność floty i przeglądy w okresach małego popytu.

## 5. Dlaczego SI i dlaczego ta technologia

**a) Dlaczego zwykły algorytm nie wystarczy.**

- [ ] nie umiemy zapisać reguły krok po kroku
- [ ] przepis istnieje, ale jest obliczeniowo za drogi
- [x] reguł jest za dużo i zmieniają się w czasie

Uzasadnienie:

Popyt zależy jednocześnie od dnia tygodnia, sezonu, świąt, pogody i wydarzeń lokalnych, a te zależności zmieniają się z roku na rok. Reguła typu „w piątek wypożyczamy więcej” będzie błędna już w długi weekend albo w sezonie urlopowym. Zamiast ręcznie utrzymywać dziesiątki reguł, model uczy się tych zależności z danych historycznych.

**b) Technologia SI:**

- [ ] sieć neuronowa
- [ ] system ekspertowy / wnioskowanie regułowe
- [ ] algorytm genetyczny
- [x] klasyczne uczenie maszynowe

Dlaczego ta technologia pasuje do tego problemu:

Mamy dane tabelaryczne (data, cechy kalendarzowe, pogoda) i prognozujemy wartość liczbową, czyli regresję. Do tego typu danych dobrze nadają się lasy losowe lub gradient boosting. Wymagają mniej danych niż sieć neuronowa i są łatwe do zinterpretowania (ważność cech), co przyda się w dokumentacji.

## 6. Dane

- Źródło:
- Rozmiar / format:
- Czy dane są już dostępne? Nie — plan pozyskania:

## 7. Stos technologiczny

Model: Python, pandas, scikit-learn. Backend: FastAPI (API udostępniające prognozy i dane o wypożyczeniach), baza SQLite. Frontend: aplikacja jednostronnicowa (SPA) w JavaScript (React lub Vue).

## 8. Kryterium sukcesu

- Metryka i próg: MAE na zbiorze testowym niższe o co najmniej 15% niż w metodzie bazowej
- Punkt odniesienia: prognoza naiwna, czyli „tyle, ile w tym samym dniu tygodnia przed tygodniem” lub średnia dla danego dnia tygodnia

## 9. Zakres minimalny (co powstanie na pewno)

Baza wypożyczeń z podstawowym panelem zarządcy (lista aut, wypożyczenia), model prognozujący liczbę wypożyczeń na dany dzień, porównanie z punktem odniesienia oraz widok prognozy dla zarządcy.

## 10. Zakres opcjonalny (co dorobimy, jeśli starczy czasu)

Podział prognozy na klasy aut, dodatkowe cechy (pogoda, święta), rekomendacja terminów przeglądów na podstawie prognozowanego popytu i przebiegu.

## 11. Repozytorium

Link: https://github.com/WojtekAndrejczuk03/Car-Rental-Manager-with-Demand-Forecasting

## 12. Główne ryzyko i plan awaryjny

Ryzyko: brak realnych danych z wypożyczalni. Plan awaryjny: użyć publicznego zbioru z danymi o wypożyczeniach albo wygenerować dane symulacyjne z sezonowością i wyraźnie to opisać w dokumentacji.

---

*Decyzja prowadzącego:* [ ] zatwierdzony  [ ] do poprawy · *Uwagi:*
