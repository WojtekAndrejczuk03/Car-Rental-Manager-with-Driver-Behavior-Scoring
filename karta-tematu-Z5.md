# Karta zgłoszenia tematu projektu

**Grupa:** L2 · **Zespół:** Z5 (Drużyna R) · **Data:** 22.10.2026
Sztuczna inteligencja — laboratorium · Collegium Da Vinci · rok akad. 2026/2027

## 1. Skład zespołu

| Imię i nazwisko | Nr albumu | Rola (wstępna) |
|---|---|---|
| Wojciech Andrejczuk | 29202 | Model ML i API, koordynacja zespołu |
| Błażej Kłoda | 31132 | Baza danych i panel zarządcy (frontend) |

## 2. Roboczy tytuł (PL)

Menedżer wypożyczalni samochodów z oceną stylu jazdy kierowcy

## 3. Tytuł (EN)

Car Rental Manager with Driver Behavior Scoring

## 4. Problem — co system ma robić i dla kogo

Małe wypożyczalnie samochodów pobierają od wszystkich klientów tę samą stawkę, choć klienci jeżdżą bardzo różnie. Agresywna jazda (gwałtowne przyspieszanie i hamowanie, szybkie pokonywanie zakrętów, przekraczanie prędkości) szybciej zużywa hamulce, opony i sprzęgło oraz zwiększa ryzyko szkody. Koszt ponoszą wszyscy klienci po równo, a spokojni kierowcy dopłacają za agresywnych.

System dla zarządcy wypożyczalni zbiera dane o jeździe z każdego wynajmu (spalanie, gwałtowne przyspieszenia, prędkość średnią i maksymalną, obszar jazdy, przebieg) i informację, czy auto zostało uszkodzone. Model uczy się na tej podstawie przewidywać ryzyko szkody, a system wylicza z niego ocenę kierowcy i jego styl jazdy. Na tej podstawie system wycenia kolejne wynajmy: spokojny kierowca dostaje zniżkę, a agresywny wyższą stawkę lub wyższą kaucję. Zarządca widzi w panelu historię wynajmów, ocenę kierowcy i uzasadnienie proponowanej ceny.

## 5. Dlaczego SI i dlaczego ta technologia

**a) Dlaczego zwykły algorytm nie wystarczy.**

- [x] nie umiemy zapisać reguły krok po kroku
- [ ] przepis istnieje, ale jest obliczeniowo za drogi
- [ ] reguł jest za dużo i zmieniają się w czasie

Uzasadnienie:

Styl jazdy to wzorzec widoczny w wielu sygnałach naraz, a nie pojedyncze przekroczenie progu. Reguła typu „hamowanie mocniejsze niż 0,4 g oznacza agresywną jazdę” myli się w obie strony: jedno awaryjne hamowanie nie czyni kierowcy agresywnym, a ktoś, kto stale jeździ nerwowo, ale tuż poniżej progu, nie zostanie wykryty. Znaczenie tych samych wartości zależy też od prędkości, rodzaju drogi i natężenia ruchu. Nie potrafimy ręcznie zapisać, jak połączyć te sygnały w ocenę, dlatego model uczy się tego z historii wynajmów, w których wiadomo, czy auto wróciło uszkodzone.

**b) Technologia SI:**

- [ ] sieć neuronowa
- [ ] system ekspertowy / wnioskowanie regułowe
- [ ] algorytm genetyczny
- [x] klasyczne uczenie maszynowe

Dlaczego ta technologia pasuje do tego problemu:

Dla każdego wynajmu mamy dane tabelaryczne: moc auta, spalanie katalogowe i rzeczywiste, liczbę gwałtownych przyspieszeń, prędkość średnią i maksymalną, obszar jazdy i dzienny przebieg. Gotowej etykiety stylu jazdy nie ma i nikt jej ręcznie nie nadaje. Wypożyczalnia rejestruje za to, czy auto wróciło uszkodzone. Model klasyfikacji binarnej (regresja logistyczna, las losowy lub gradient boosting) uczy się więc przewidywać prawdopodobieństwo szkody na podstawie sposobu jazdy. Z tego prawdopodobieństwa wyliczamy ocenę kierowcy w skali 0–100, a z oceny styl jazdy: spokojny, normalny albo agresywny. Te modele wymagają mniej danych niż sieć neuronowa i pokazują ważność cech, co pozwala wytłumaczyć klientowi i zarządcy, dlaczego cena wzrosła. Przejrzystość jest tu ważna, bo ocena bezpośrednio wpływa na to, ile klient zapłaci. Sama wycena to jawna formuła oparta na ocenie, a nie część modelu.

## 6. Dane

- Źródło: dane syntetyczne z własnego generatora `scripts/generate_dataset.py` (opis w `data/README.md`). Prawdziwych danych z wypożyczalni, które łączyłyby dane o jeździe ze szkodami, nie udało się pozyskać (zob. punkt 12).
- Rozmiar / format: CSV, 3000 wynajmów, 598 kierowców, 15 kolumn (`data/wynajmy.csv`) oraz plik walidacyjny z ukrytą agresywnością kierowców (`data/walidacja_agresywnosc.csv`, nieużywany do uczenia).
- Czy dane są już dostępne? Tak — są w repozytorium i można je odtworzyć poleceniem `python3 scripts/generate_dataset.py --wynajmy 3000 --seed 42`.

## 7. Stos technologiczny

Model: Python, pandas, scikit-learn. Backend: FastAPI (API udostępniające prognozy i dane o wypożyczeniach), baza SQLite. Frontend: aplikacja jednostronicowa (SPA) w JavaScript (Vue 3).

## 8. Kryterium sukcesu

- Metryka i próg: ROC AUC przewidywania szkody na zbiorze testowym co najmniej 0,70 i co najmniej o 0,08 wyżej niż w metodzie bazowej. Zbiór testowy wydzielamy według kierowców, aby model nie oceniał kierowców, których widział podczas uczenia. Dodatkowo na danych syntetycznych sprawdzamy, czy ocena kierowcy jest zgodna z jego prawdziwą, ukrytą agresywnością (korelacja rang Spearmana co najmniej 0,70).
- Punkt odniesienia: reguła oceniająca ryzyko wyłącznie na podstawie liczby gwałtownych przyspieszeń na 100 km.

## 9. Zakres minimalny (co powstanie na pewno)

Baza danych aut, kierowców i wynajmów z podstawowym panelem zarządcy. Import danych o jeździe z wynajmu z pliku CSV. Model przewidujący ryzyko szkody, z którego wyliczamy ocenę kierowcy w skali 0–100 i styl jazdy (spokojny, normalny, agresywny). Porównanie modelu z punktem odniesienia. Wycena wynajmu według formuły „stawka bazowa × mnożnik zależny od oceny” oraz widok profilu kierowcy z historią ocen i proponowaną ceną.

## 10. Zakres opcjonalny (co dorobimy, jeśli starczy czasu)

Zbieranie danych na żywo z telefonu kierowcy (akcelerometr, żyroskop, GPS) zamiast importu z pliku. Wykrywanie pojedynczych niebezpiecznych zdarzeń i pokazywanie ich na mapie przejazdu. Szczegółowe wyjaśnienie oceny dla klienta, np. wkład poszczególnych cech. Szacowanie zużycia auta i rekomendacja terminów przeglądów na podstawie stylu jazdy jego kierowców. Panel klienta z podglądem własnej oceny.

## 11. Repozytorium

Link: https://github.com/WojtekAndrejczuk03/Car-Rental-Manager-with-Driver-Behavior-Scoring

## 12. Główne ryzyko i plan awaryjny

Ryzyko: brak prawdziwych danych z wypożyczalni, w których byłyby jednocześnie dane o jeździe i informacja o szkodach. Plan awaryjny: używamy danych syntetycznych z generatora w repozytorium. Generator nadaje każdemu kierowcy ukrytą agresywność, z której wynikają dane o jeździe i szkody. Dzięki temu możemy sprawdzić, czy ocena wyliczona przez model zgadza się z prawdziwym stylem jazdy. Syntetyczne pochodzenie danych opiszemy wyraźnie w dokumentacji. Dodatkowe ryzyko to ochrona danych o jeździe kierowców (RODO): we wdrożeniu potrzebna byłaby zgoda klienta na zbieranie i ocenę tych danych.

---

*Decyzja prowadzącego:* [ ] zatwierdzony  [ ] do poprawy · *Uwagi:*
