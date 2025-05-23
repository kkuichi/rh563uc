## Inštalácia požiadaviek a špeciálnej knižnice

Na spustenie projektu je potrebné najskôr vytvoriť virtuálne prostredie a nainštalovať všetky požiadavky uvedené v súbore requirements.txt.

### 1. Inštalácia požiadaviek

V termináli spustite:

```bash
pip install -r requirements.txt
```
---

### 2. Knižnica `scikit-learn-extra`

Knižnica `scikit-learn-extra`, ktorá sa používa na algoritmus K-Medoids, nie je v súbore `requirements.txt` zahrnutá, pretože jej inštalácia štandardným spôsobom často zlyhá pri novších verziách Pythonu. Problém vzniká pri kompilácii kvôli tomuto riadku v zdrojovom súbore:

```bash
from numpy.math cimport INFINITY
```

Tento import spôsobí chybu „numpy/math.pxd not found“.

---

### 3. Inštalácia `scikit-learn-extra` 

1. Otvorte terminál v koreňovom priečinku projektu.

2. Vytvorte nový skript:

```bash
nano install_kmedoids.sh
```

3. Vložte do súboru nasledujúci obsah:

```bash
#!/bin/bash

git clone https://github.com/scikit-learn-contrib/scikit-learn-extra.git
cd scikit-learn-extra || { echo "zlyhanie cd"; exit 1; }

TARGET_FILE="sklearn_extra/robust/_robust_weighted_estimator_helper.pyx"

#problémový riadok
grep INFINITY "$TARGET_FILE"

#úprava
sed -i '' '/from numpy.math cimport INFINITY/d' "$TARGET_FILE"

#kontrola
grep INFINITY "$TARGET_FILE"

#inštalácia
pip install .
```

4. Uložte a zatvorte súbor.

5. Nastavte práva na spustenie:

```bash
chmod +x install_kmedoids.sh
```

6. Spustite skript:

```bash
./install_kmedoids.sh
```

Na macOS sa používa sed s parametrom `-i ''`. Ale ak používate Linux treba tento riadok v skripte upraviť:

```bash
sed -i '/from numpy.math cimport INFINITY/d' "$TARGET_FILE"
```
---
Základná knižnica [`scikit-learn`](https://scikit-learn.org/stable/) nepodporuje zhlukovanie s použitím custom metriky, ako je DTW. Preto bola použitá knižnica [`scikit-learn-extra`](https://github.com/scikit-learn-contrib/scikit-learn-extra), ktorá obsahuje algoritmus `KMedoids`, schopný pracovať s ľubovoľnou maticou vzdialeností. Táto knižnica je oficiálne súčasťou projektu [scikit-learn-contrib](https://scikit-learn-contrib.github.io/), je dostupná aj cez [PyPI](https://pypi.org/project/scikit-learn-extra/) a nejedná sa o žiaden neoficiálny alebo nedôveryhodný zdroj.
