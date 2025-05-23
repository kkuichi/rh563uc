# Používateľská príručka k bakalárskej práci  


Tento repozitár obsahuje implementáciu a analýzu, ktorá bola vypracovaná v rámci bakalárskej práce s názvom:  
**„Segmentovanie zákazníkov pre distribúciu elektrickej energie“**.

Cieľom je segmentovať zákazníkov na základe ich správania a úrovne spotreby elektrickej energie.  
Na tento účel sa používajú 15-minútové časové vzory spotreby , ktoré sú ďalej zhlukovane na úrovni dní, týždňov a rokov pomocou algoritmu K-Medoids a metriky DTW (Dynamic Time Warping).


## Dáta
Dáta boli získané z Dataportu , najväčšieho svetového zdroja údajov o spotrebe energie v domácnostiach. Táto platforma je súčasťou výskumnej organizácie **[Pecan Street ](https://www.pecanstreet.org)** , ktorá sa zaoberá zberom a analýzou dát o spotrebe energie a vody. Údaje použité v tejto práci obsahujú záznamy spotreby elektrickej energie domácností z troch regiónov:

- **New York**  
- **California**  
- **Austin, Texas**

Kvôli licenčným podmienkam a ochrane osobných údajov dáta nie je možné verejne zdieľať.
Aby ste mohli použíť úplné dáta, musíte si vytvoriť účet a získať prístup ako súčasť univerzitnho výskumu  na [https://dataport.pecanstreet.org](https://dataport.pecanstreet.org) a manuálne si stiahnuť:

- `newyork.csv`
- `california.csv`
- `austin.csv`

Tieto súbory je potrebné uložiť do koreňového priečinka pred spustením kódu.



## Požiadavky

#### Verzia Pythonu: 3.9.6
#### Požiadavky na knižnice použité v kóde sú uvedené v súbore _requirements.txt_ a návod na ich stiahnutie sa nachádza v súbore _NAVOD.md_.


## Prehľad súborov a ich účel (v poradí spustenia)

| Súbor                  | Popis                                                                             |
|------------------------|-----------------------------------------------------------------------------------|
| `pochopenie_dat.py`    | Základný prieskum datasetov, počty, typy premenných, NaN, extrémy                 |
| `priprava_dat.py`      | Čistenie a agregácia dát, tvorba denných vektorov                                 |
| `DTW_matica_dni.py`    | Výpočet DTW matice pre denné vektory                                              |
| `Modelovanie_dni.py`   | Zhlukovanie dní a výber optimálneho počtu zhlukov                                 |
| `Modelovanie_tyz.py`   | Týždenné vzory, ich zhlukovanie a výber optimálneho počtu zhlukov, najčastejšie vzory v každom zhluku |
| `Modelovanie_rok.py`   | Ročné vektory, DTW medzi nimi, zhlukovanie, sezónna analýza                       |
| `vizual.py`            | Vizualizácie: sezónnosť, entropia, frekvencie týždenných a denných zhlukov       |



## Výstupy

Po úspešnom spustení sa vytvoria nasledujúce výstupné súbory:

- `daily_15min_vectors.csv`  
  Kompletné denné profily (96 hodnôt pre každý deň a domácnosť) po čistení a agregácii.

- `dtw_matrix_days.npy`  
  Matica DTW vzdialeností medzi dennými profilmi.

- `dtw_index_map.csv`  
  Mapa indexov (dataid, day) pre `dtw_matrix_days.npy`.

- `daily_clusters_6.csv`  
  Výsledky zhlukovania dní do 6 denných typov (k=6) s priradením ku každému dňu.

- `cluster_similarity_matrix.csv`  
  Matica DTW vzdialeností medzi priemernými profilmi denných zhlukov.

- `weekly_patterns.csv`  
  Týždenné vzory reprezentované ako reťazce 7 denných typov (napr. `0123204`).

- `weekly_patterns_clustered_5.csv`  
  Výsledky zhlukovania týždenných šablón do 5 týždenných typov (k=5).

- `weekly_cluster_similarity_matrix.csv`  
  Matica podobnosti medzi medoidmi týždenných zhlukov, použitá na ročné DTW.

- `yearly_weekly_vectors.csv`  
  Ročné vektory pre každú domácnosť (sekvencie 52 týždňových zhlukov).

- `yearly_clustered_k4.csv`  
  Finálne zhlukovanie domácností do 4 ročných skupín (k=4) na základe ich ročného profilu.

- `cluster_monthly_statistics.csv`  
  Agregovaná mesačná spotreba podľa ročného zhluku, vrátane priemeru za celý rok.

- V priečinku `Grafy` sa nachádzajú aj vizualizácie vytvorené na základe kompletných dát zo všetkých fáz analýzy
