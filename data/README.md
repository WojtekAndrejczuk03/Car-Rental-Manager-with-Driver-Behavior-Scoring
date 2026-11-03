# Zbiór danych: wynajmy samochodów

Pliki `wynajmy.csv` i `walidacja_agresywnosc.csv` zawierają **dane syntetyczne** wygenerowane skryptem `scripts/generate_dataset.py`. Jeden wiersz to jeden wynajem. Dane można odtworzyć poleceniem:

```
python3 scripts/generate_dataset.py --wynajmy 3000 --seed 42 --out data/wynajmy.csv
```

## Kolumny

| Kolumna | Opis |
|---|---|
| id_wynajmu | identyfikator wynajmu |
| id_kierowcy | identyfikator kierowcy (jeden kierowca ma kilka wynajmów) |
| marka_i_model | marka i model samochodu |
| typ_samochodu | miejski, osobowy, kombi, SUV, offroad, sportowy |
| klasa_pojazdu | segment europejski: A, B, C, D, E, F, J (SUV i terenowe), S (sportowe) |
| liczba_koni_mechanicznych | moc silnika [KM] |
| spalanie_producenta_l_100km | średnie spalanie katalogowe [l/100 km] |
| spalanie_kierowcy_l_100km | rzeczywiste średnie spalanie podczas wynajmu [l/100 km] |
| liczba_gwaltownych_przyspieszen | liczba gwałtownych przyspieszeń w całym wynajmie |
| srednia_predkosc_kmh | średnia prędkość jazdy [km/h] |
| najwyzsza_predkosc_kmh | najwyższa zarejestrowana prędkość [km/h] |
| obszar_jazdy | miasto, trasa, autostrada, mieszany |
| km_na_dzien | średni dzienny przebieg [km] |
| liczba_dni_wynajmu | długość wynajmu w dniach |
| uszkodzenie_samochodu | czy kierowca uszkodził samochód (tak / nie), zmienna, której uczy się model |

## Jak powstały dane

Każdy kierowca ma ukryty poziom agresywności. Wpływa on na spalanie, liczbę gwałtownych przyspieszeń, prędkości i ryzyko szkody, ale każda z tych cech zależy też od obszaru jazdy, mocy auta, natężenia ruchu i indywidualnych nawyków kierowcy. Dzięki temu żadna pojedyncza cecha nie wyznacza stylu jazdy jednoznacznie. Dane katalogowe aut (moc, spalanie) są przybliżone.

## Styl jazdy nie jest w danych

Zbiór celowo nie zawiera stylu jazdy. Model uczy się przewidywać szkodę, a z przewidywanego ryzyka wyliczamy ocenę kierowcy i jego styl jazdy.

Plik `walidacja_agresywnosc.csv` zawiera prawdziwą agresywność z generatora (0 = bardzo spokojny, 1 = bardzo agresywny). **Nie wolno go używać do uczenia.** Służy tylko do sprawdzenia, czy ocena wyliczona przez model zgadza się z rzeczywistością. W prawdziwej wypożyczalni takiego pliku by nie było.

## Uwagi do modelowania

- Liczba gwałtownych przyspieszeń rośnie z przebiegiem, więc warto ją przeliczyć na 100 km: `liczba_gwaltownych_przyspieszen / (km_na_dzien * liczba_dni_wynajmu) * 100`.
- Zbiór testowy należy wydzielać według `id_kierowcy`, aby ten sam kierowca nie trafił do uczenia i testu.
