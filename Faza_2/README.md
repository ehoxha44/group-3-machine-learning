<table>
  <tr>
    <td width="150" align="center" valign="middle">
      <img src="https://upload.wikimedia.org/wikipedia/commons/e/e1/University_of_Prishtina_logo.svg" width="120" alt="University of Prishtina Logo" />
    </td>
    <td valign="top">
      <p><strong>Universiteti i Prishtinës</strong></p>
      <p>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</p>
      <p>Inxhinieri Kompjuterike dhe Softuerike – Programi Master</p>
      <p><strong>Lënda:</strong> Machine Learning</p>
      <p><strong>Profesorët:</strong> Prof. Dr. Lule Ahmedi dhe Dr. Sc. Mërgim Hoti</p>
      <p><strong>Studentët (Gr. 3):</strong> Enis Hoxha · Fisnik Hazrolli · Endri Binaku</p>
    </td>
  </tr>
</table>

---

# Faza 2 – Analizë e Rritjes Ekonomike, Trajnimi i Modeleve dhe Parashikimet e Tregut

## Përmbajtja

1. [Qëllimi dhe Pyetjet Kërkimore](#1-qëllimi-dhe-pyetjet-kërkimore)
2. [Nga të Dhënat e Papërpunuara tek Dataseti i Modelimit](#2-nga-të-dhënat-e-papërpunuara-tek-dataseti-i-modelimit)
3. [Inxhinieria e Tipareve dhe Parandalimi i Data Leakage](#3-inxhinieria-e-tipareve-dhe-parandalimi-i-data-leakage)
4. [Algoritmet e Përdorura – Pasqyrë e Përgjithshme](#4-algoritmet-e-përdorura--pasqyrë-e-përgjithshme)
5. [Algoritmi 1 – Random Forest Classifier](#5-algoritmi-1--random-forest-classifier)
6. [Algoritmi 2 – XGBoost Classifier](#6-algoritmi-2--xgboost-classifier)
7. [Krahasimi i Klasifikuesve dhe Verdikti](#7-krahasimi-i-klasifikuesve-dhe-verdikti)
8. [Algoritmi 3 – Linear Regression](#8-algoritmi-3--linear-regression)
9. [Algoritmi 4 – Random Forest Regressor](#9-algoritmi-4--random-forest-regressor)
10. [Algoritmi 5 – XGBoost Regressor](#10-algoritmi-5--xgboost-regressor)
11. [Krahasimi i Regresorëve](#11-krahasimi-i-regresorëve)
12. [Algoritmi 6 – K-Means Clustering](#12-algoritmi-6--k-means-clustering)
13. [Algoritmi 7 – Principal Component Analysis (PCA)](#13-algoritmi-7--principal-component-analysis-pca)
14. [Analiza Kryqëzore: Sektor × Komunë (Heatmaps)](#14-analiza-kryqëzore-sektor--komunë-heatmaps)
15. [Parashikimet e Tregut 2026](#15-parashikimet-e-tregut-2026)
16. [Udhëzime për Ekzekutim](#16-udhëzime-për-ekzekutim)
17. [Struktura e Skedarëve dhe Output-et](#17-struktura-e-skedarëve-dhe-output-et)
18. [Lidhja me Fazën 1 dhe Ndikimi i Parapërpunimit](#18-lidhja-me-fazën-1-dhe-ndikimi-i-parapërpunimit)

---

## 1. Qëllimi dhe Pyetjet Kërkimore

Faza 2 ka një qëllim të vetëm, të fokusuar: **të kuptojë dhe parashikojë rritjen ekonomike të bizneseve në Kosovë sipas qytetit, sektorit dhe kombinimit sektor × qytet.**

Pyetjet konkrete të hulumtimit janë:

| # | Pyetja | Algoritmi që i përgjigjet |
|---|--------|--------------------------|
| 1 | A rriti qarkullimin kombinimi (komunë, sektor) krahasuar me vitin e mëparshëm? | RF Classifier, XGBoost Classifier |
| 2 | Me sa përqind ndryshoi qarkullimi vit-mbi-vit? | Linear Reg., RF Regressor, XGBoost Regressor |
| 3 | Cilat komuna kanë ndarë profile të ngjashme rritjeje gjatë viteve 2020–2025? | K-Means (trajektore komunash) |
| 4 | Cilët sektorë kanë sjellë dinamika të ngjashme ndër vite? | K-Means (trajektore sektorësh) |
| 5 | Si duket struktura latente e të dhënave? A ka grupime natyrale? | PCA + K-Means |
| 6 | Cilat kombinime (komunë, sektor) pritet të rriten më shumë në vitin 2026? | Të 5 modelet supervized |

> **Ndryshimi kritik nga qasja e mëparshme**: Versioni i parë i projektit klasifikonte *llojin ligjor të entitetit* (`registration_status`: SH.P.K., INDIVIDUAL, etj.) — pyetje me zero vlerë ekonomike. Faza 2 e rindërtoi plotësisht problemin: agregohet e gjithë baza sipas `(vit, komunë, sektor)` dhe target-i bëhet `growth_class` (GROWING / STABLE / DECLINING), i llogaritur nga ndryshimi YoY i qarkullimit.

---

## 2. Nga të Dhënat e Papërpunuara tek Dataseti i Modelimit

### 2.1 Burimi i të Dhënave

Dataseti kryesor është `model_ready.csv`, produkti i pipeline-it të Fazës 1, me origjinë nga [ATK Open Data](https://www.atk-ks.org/en/open-data/):

| Karakteristikë | Vlera |
|----------------|-------|
| Rreshta bruto (pas pastrimit Faza 1) | 650,913 |
| Kolonat kryesore | `year`, `municipality`, `primary_sector`, `turnover_eur`, `num_taxpayers` |
| Periudha kohore | 2019 – 2025 |
| Njësia e observimit | Një biznes (tatimpagues) në një muaj/vit |

### 2.2 Agregimi dhe Ndërtimi i Dataset-it të Rritjes

Meqenëse pyetja jonë nuk është "si performon biznesi X?", por "si performon *sektori Y* në *komunën Z* çdo vit?", të gjitha rreshtat agregohen:

```
(year, municipality, primary_sector)
    → sum(turnover_eur)     = total_turnover
    → sum(num_taxpayers)    = num_businesses
```

**Hapat e transformimit:**

```
Hapi 1 – Agregim:
  650,913 rreshta  →  4,649 grupe (vit × komunë × sektor)

Hapi 2 – Lag feature (qarkullimi i vitit të mëparshëm):
  prev_turnover[t] = total_turnover[t-1]  për të njëjtin grup

Hapi 3 – Shkalla YoY e rritjes:
  growth_rate = (total_turnover[t] − prev_turnover[t]) / prev_turnover[t]
  → clipped në [−2.0, +5.0] për të shmangur vlera ekstreme nga biznese me bazë shumë të vogël

Hapi 4 – Transformimi log1p:
  total_turnover_log1p = log(1 + total_turnover)
  prev_turnover_log1p  = log(1 + prev_turnover)
  num_businesses_log1p = log(1 + num_businesses)
  → normalizon shpërndarjen e shtrembëruar djathtas të të dhënave financiare

Hapi 5 – Klasifikimi i rritjes:
  growth_rate > +5%  → GROWING   (2,322 rreshta – 58.7%)
  growth_rate < −5%  → DECLINING (1,226 rreshta – 31.0%)
  ndërmjet           → STABLE    (408 rreshta   – 10.3%)

Hapi 6 – Heqja e vitit të parë (pa lag):
  4,649 − 693 = 3,956 rreshta finale
```

**Dataset-i final: 3,956 rreshta × 7 kolona kryesore**

| Kolona | Lloji | Shpjegimi |
|--------|-------|-----------|
| `year` | Numerike | Viti (2020–2025) |
| `municipality` | Kategorike | Emri i komunës (38 të ndryshme) |
| `primary_sector` | Kategorike | Sektori ekonomik (21 të ndryshëm) |
| `prev_turnover_log1p` | Numerike | log1p i qarkullimit të vitit të mëparshëm |
| `num_businesses_log1p` | Numerike | log1p i numrit të tatimpaguesve |
| `growth_rate` | Numerike | Target-i i regresionit (−2 deri +5) |
| `growth_class` | Kategorike | Target-i i klasifikimit (GROWING/STABLE/DECLINING) |

---

## 3. Inxhinieria e Tipareve dhe Parandalimi i Data Leakage

### 3.1 Tiparet e Modelimit

```python
NUMERIC_FEATURES = [
    "year",                  # sinjali kohor – a ka trende sezonale/globale?
    "prev_turnover_log1p",   # qarkullimi i vitit të mëparshëm – momentum ekonomik
    "num_businesses_log1p",  # madhësia e tregut lokal – densiteti i aktivitetit
]

CATEGORICAL_FEATURES = [
    "municipality",    # identiteti gjeografik – enkoduar me LabelEncoder
    "primary_sector",  # identiteti sektorial – enkoduar me LabelEncoder
]
```

**Enkodimi:** `LabelEncoder` konverton emrat e komunave dhe sektorëve në integer. `StandardScaler` e centron dhe normalizon të gjitha tiparet numerike (zero meze, variancë 1).

**Matrica finale X:** `3,956 × 5` (3 numerike + 2 të enkuduara)

### 3.2 Ç'është Data Leakage dhe Si e Shmangëm

**Leakage** ndodh kur tiparet e modelit "të zbulojnë" drejtpërdrejt target-in, duke prodhuar saktësi artificiale 100% që zhduket në prodhim.

Dy kolonat e ekskluzuara me qëllim:

| Kolona | Pse do ishte leakage |
|--------|----------------------|
| `growth_rate` | **IS** target-i i regresionit dhe bazë matematikore e `growth_class` — ta përfshish si tipar do të thotë që modeli "sheh" përgjigjen |
| `total_turnover_log1p` | Rezultati i vitit aktual — nuk dihet në kohën e parashikimit |

> **Shembull konkret:** Para rregullimit, modeli kishte accuracy 100% — sepse `growth_rate` ishte si tipar dhe njëkohësisht si target. Pasi u hoq, accuracy zbriti në 59–63%, që është saktësia e vërtetë e modelit.

Tiparet e lejuara janë vetëm ato **të njohura para se të mbarohet viti aktual**: qarkullimi i vitit të kaluar (`prev_turnover_log1p`), numri i bizneseve, viti dhe identiteti gjeografik/sektorial.

---

## 4. Algoritmet e Përdorura – Pasqyrë e Përgjithshme

Faza 2 implementon **7 algoritme machine learning** të organizuara në tre grupe:

| # | Algoritmi | Grupi | Target | Qëllimi i Detajuar |
|---|-----------|-------|--------|---------------------|
| 1 | **Random Forest Classifier** | Supervised – Klasifikim | `growth_class` | Klasifikon çdo (komunë, sektor, vit) si GROWING / STABLE / DECLINING |
| 2 | **XGBoost Classifier** | Supervised – Klasifikim | `growth_class` | Alternativë gradient-boosting për klasifikim me saktësi të lartë |
| 3 | **Linear Regression** | Supervised – Regresion | `growth_rate` | Benchmark linear: sa % ndryshon qarkullimi? |
| 4 | **Random Forest Regressor** | Supervised – Regresion | `growth_rate` | Regresion jo-linear me ensemble pemësh |
| 5 | **XGBoost Regressor** | Supervised – Regresion | `growth_rate` | Modeli kryesor për parashikimet e tregut 2026 |
| 6 | **K-Means Clustering** | Unsupervised | — | Grupon komuna dhe sektorë me profile të ngjashme rritjeje |
| 7 | **PCA** | Unsupervised | — | Redukton dimensionalitetin dhe vizualizon strukturën latente |

---

## 5. Algoritmi 1 – Random Forest Classifier

### 5.1 Ç'është Random Forest

Random Forest është një metodë **ensemble learning** që kombinom parashikimet e shumë pemëve vendimi (Decision Trees) të trajnuara paralelisht.

**Ideja themelore:**
- Një pemë vendimi e vetme është e prirur të mbipërshtatej (overfitting) me të dhënat e trajnimit.
- Nëse trajnohen 300 pemë, secila në një nënkampion të ndryshëm të të dhënave dhe tipareve, dhe pastaj kombinohen me votim shumice, gabimi individual i çdo peme "mesatarizohet" dhe modeli final është shumë më i qëndrueshëm.

**Bagging (Bootstrap Aggregating):**
Çdo pemë trajnohet mbi një kampion bootstrap të të dhënave (me zëvendësim). Rreth 37% e rreshtave nuk hyjnë në trajnimin e asnjë peme të caktuar — këto quhen *out-of-bag samples* dhe mund të përdoren për vlerësim të brendshëm.

**Feature Randomness:**
Në çdo ndarje brenda pemës, algoritmi konsideron vetëm `sqrt(n_features)` tipare të zgjedhura rastësisht. Kjo i shton diversitetin pemëve dhe zvogëlon korrelacionin ndërmjet tyre.

### 5.2 Pse e Zgjodhëm për Këtë Problem

| Arsyeja | Shpjegimi |
|---------|-----------|
| **Tipare të përziera** | Dataset-i ka kolona numerike (qarkullim, vit) dhe të enkuduara (komunë, sektor). Random Forest nuk kërkon presupozime lineare — ndarjet e pemëve trajtojnë të dyja llojet njësoj mirë. |
| **Robustësi ndaj outliers** | Të dhënat financiare kanë shpërndarje shumë të shtrembëruar — disa biznese me qarkullim shumë të lartë. RF vendos mbi bazë ndarjesh, jo distancash, kështu que outliers nuk dominojnë. |
| **Imbalancë e klasave** | GROWING = 58.7%, DECLINING = 31%, STABLE = 10.3%. Parametri `class_weight="balanced"` shumëfishon automatikisht peshat e klasave të vogla, duke parandaluar që modeli të parashikojë gjithmonë "GROWING". |
| **Interpretueshmëria** | `feature_importances_` tregon drejtpërdrejt cilat tipare — viti, qarkullimi i vitit të kaluar, sektori apo komuna — kanë ndikimin më të madh. |
| **Dataseti i vogël** | Me vetëm 3,956 rreshta, bagging zvogëlon overfitting-un ndjeshëm krahasuar me modele të thella si neural networks. |

### 5.3 Hiperparametrat

```python
RF_PARAMS = {
    "n_estimators":    300,        # 300 pemë — kompromis midis saktësisë dhe shpejtësisë
    "max_depth":       10,         # thellësia maksimale — parandalon memorimin e të dhënave
    "min_samples_split": 4,        # një ndarje kërkon të paktën 4 rreshta — rregullarizim
    "class_weight":    "balanced", # kompensim automatik i imbalancës
    "random_state":    42,         # riprodukshmëria
    "n_jobs":          -1,         # paralelizim maksimal i CPU
}
```

### 5.4 Procesi i Trajnimit dhe Testimit

```
Dataset: 3,956 rreshta
  ↓
Split 80/20 stratifikuar (stratifikuar = ruan proporcionet e klasave):
  Train: 3,164 rreshta
  Test:  792 rreshta

Trajnim: 300 peme × bootstrap × random features

Parashikim: votim shumice ndër 300 pemë
  → dalje: GROWING / STABLE / DECLINING për çdo rresht test
```

### 5.5 Metrikat e Evaluimit – Ç'masim dhe Pse

| Metrika | Formula | Interpretimi |
|---------|---------|--------------|
| **Accuracy** | (TP + TN) / Total | % e parashikimeve korrekte. Mashtrues me klasa të pabarabarta. |
| **Precision (macro)** | avg(TP / (TP + FP)) per klasë | Sa i saktë është modeli kur thotë "GROWING"? |
| **Recall (macro)** | avg(TP / (TP + FN)) per klasë | Sa sektorë DECLINING i zbulon vërtet? |
| **F1-Score (macro)** | 2 × (Precision × Recall) / (P + R) | Balancë midis saktësisë dhe mbulimit. |
| **Cohen's Kappa** | (Accuracy − Expected) / (1 − Expected) | Korrektësi e saktësisë duke zbritur rasësinë. 0 = rastësor, 1 = perfekt. |
| **Confusion Matrix** | Matrica TP/FP/FN/TN per klasë | Tregon kur ngatërrrohet STABLE me DECLINING. |
| **ROC-AUC (OvR)** | Sipërfaqja nën kurbën ROC | Shkëmbimi TP-rate vs FP-rate për secilën klasë. |

**Pse Kappa dhe jo vetëm Accuracy?**
Nëse modeli parashikon gjithmonë "GROWING" (klasa me 58.7% raste), fiton accuracy 58.7% pa mësuar asgjë. Kappa korrigjon këtë: një model rastësor ka Kappa=0, dhe modeli ynë me Kappa=0.25 tregon mësim real mbi rasësinë.

### 5.6 Rezultatet e Random Forest

#### Hold-out Test Set (792 rreshta)

| Klasa | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| DECLINING | 0.485 | **0.588** | 0.531 | 245 |
| GROWING | **0.706** | 0.667 | **0.686** | 465 |
| STABLE | 0.232 | 0.159 | 0.188 | 82 |
| **macro avg** | **0.474** | **0.471** | **0.469** | 792 |
| weighted avg | 0.589 | 0.590 | 0.587 | 792 |
| **Accuracy** | | | **0.590** | |
| **Cohen's Kappa** | | | **0.256** | |

#### 5-Fold Cross-Validation

| Metrika | Mean | Std |
|---------|------|-----|
| Accuracy | 0.5811 | ±0.0091 |
| Precision (macro) | 0.4536 | ±0.0107 |
| Recall (macro) | **0.4574** | ±0.0111 |
| F1-Score (macro) | **0.4533** | ±0.0110 |
| Cohen's Kappa | 0.2425 | ±0.0138 |

**Interpretimi i Rezultateve:**
- Klasa GROWING (F1=0.686) identifikohet mirë — ka shumë shembuj dhe sinjal i qartë.
- Klasa STABLE (F1=0.188) identifikohet dobët — "stabiliteti" i vërtetë ekonomik (ndryshim brenda ±5%) është shumë i ngjashëm me GROWING dhe DECLINING dhe ka vetëm 82 rreshta trajnimi.
- Devijimi standard i ulët (±0.009) konfirmon qëndrueshmëri ndër foldat — nuk ka "fat të mirë".

#### Grafiqet e gjeneruara:
- `outputs/plots/RandomForest_confusion_matrix.png`
- `outputs/plots/RandomForest_roc_curves.png`
- `outputs/plots/RandomForest_feature_importance.png`

---

## 6. Algoritmi 2 – XGBoost Classifier

### 6.1 Ç'është XGBoost

XGBoost (Extreme Gradient Boosting) është një metodë **gradient boosting** ku pemët ndërtohen **sekuencialisht**, jo paralelisht si në Random Forest.

**Ideja themelore:**
- Modeli 1: trajnohet mbi të dhënat origjinale. Bën gabime të caktuara.
- Modeli 2: trajnohet mbi *mbetjet* (residuals) e modelit 1 — duke korrigjuar gabimet e bëra.
- Modeli 3: trajnohet mbi mbetjet e kombinimit 1+2. Dhe kështu me radhë.
- Parashikimi final = shuma e ponderuar e 300 pemëve.

**Funksioni objektiv:**
```
L(θ) = Σ l(yᵢ, ŷᵢ) + Σ Ω(fₖ)
```
- `l` = humbja (cross-entropy për klasifikim)
- `Ω` = rregullarizimi (L1 + L2) — penalizon pemët e komplikuara

**Pse "Extreme"?**
- Zbaton *second-order Taylor expansion* mbi funksionin e humbjes për llogaritje shumë efikase të gradientëve.
- Mbështet *column subsampling* dhe *row subsampling* për të shtuar randomness dhe zvogëluar overfitting.

### 6.2 Pse e Zgjodhëm

| Arsyeja | Shpjegimi |
|---------|-----------|
| **Performancë superiore mbi tabelar** | XGBoost ka vendosur rekorde në dhjetëra competition-e Kaggle mbi të dhëna tabulare strukturore — pikërisht ky lloj i dataset-it tonë. |
| **Korrigjim sekuencial i gabimeve** | Ndërsa RF mesatarizon, XGBoost mëson drejtpërdrejt nga gabimet e mëparshme — konvergjencë më e shpejtë në saktësi. |
| **Regularizim i integruar** | L1 (sparse) + L2 (smooth) rregullarizim parandalon overfitting-un pa nevojë konfigurimi shtesë. |
| **Subsampling** | `colsample_bytree=0.8` (80% tipareve per pemë) + `subsample=0.8` (80% rreshtave) shtojnë diversitetin si në RF, por brenda boosting-ut. |

### 6.3 Hiperparametrat

```python
XGB_PARAMS = {
    "n_estimators":    300,        # 300 iteracione boosting
    "max_depth":       6,          # më i cekët se RF — boosting nuk ka nevojë pemë të thella
    "learning_rate":   0.05,       # hapi i korrigjimit — i vogël = konvergjencë e qëndrueshme
    "subsample":       0.8,        # 80% rreshtave per iteracion
    "colsample_bytree": 0.8,       # 80% tipareve per pemë
    "eval_metric":     "mlogloss", # log-loss multi-klasë si metrikë monitorimi
    "objective":       "multi:softprob",  # për 3 klasa (automatik)
    "num_class":       3,          # GROWING, STABLE, DECLINING
    "random_state":    42,
}
```

### 6.4 Rezultatet e XGBoost

#### Hold-out Test Set (792 rreshta)

| Klasa | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| DECLINING | 0.524 | 0.445 | 0.481 | 245 |
| GROWING | 0.673 | **0.822** | **0.740** | 465 |
| STABLE | 0.250 | 0.049 | 0.082 | 82 |
| **macro avg** | **0.482** | 0.438 | 0.434 | 792 |
| weighted avg | 0.583 | 0.625 | 0.592 | 792 |
| **Accuracy** | | | **0.625** | |
| **Cohen's Kappa** | | | **0.243** | |

#### 5-Fold Cross-Validation

| Metrika | Mean | Std |
|---------|------|-----|
| Accuracy | **0.6317** | ±0.0059 |
| Precision (macro) | **0.4948** | ±0.0204 |
| Recall (macro) | 0.4481 | ±0.0060 |
| F1-Score (macro) | 0.4472 | ±0.0082 |
| Cohen's Kappa | **0.2594** | ±0.0130 |

**Interpretimi:**
- XGBoost arrin **Recall=0.822 për GROWING** — kap 82 prej 100 kombinimeve që vërtet u rritën.
- Klasa STABLE gati zhduket (Recall=0.049): XGBoost e grupon si GROWING ose DECLINING pothuaj gjithmonë. Kjo është e arsyeshme ekonomikisht — bizneset rrallë qëndrojnë "stabile" saktësisht ndërmjet ±5% për një periudhë 5-vjeçare.
- Devijimi standard shumë i ulët (±0.006) tregon qëndrueshmëri të lartë ndër foldat.

#### Grafiqet e gjeneruara:
- `outputs/plots/XGBoost_confusion_matrix.png`
- `outputs/plots/XGBoost_roc_curves.png`
- `outputs/plots/XGBoost_feature_importance.png`

---

## 7. Krahasimi i Klasifikuesve dhe Verdikti

### 7.1 Tabela e Verdiktit Kokë-më-Kokë

| Metrika | Random Forest | XGBoost | Fituesi |
|---------|---------------|---------|---------|
| Accuracy (hold-out) | 0.5896 | **0.6250** | XGBoost ✓ |
| Precision macro (hold-out) | 0.4744 | **0.4822** | XGBoost ✓ |
| Recall macro (hold-out) | **0.4710** | 0.4384 | Random Forest ✓ |
| F1-Score macro (hold-out) | **0.4685** | 0.4342 | Random Forest ✓ |
| Cohen's Kappa (hold-out) | **0.2556** | 0.2433 | Random Forest ✓ |
| Accuracy (CV mean) | 0.5811±0.0091 | **0.6317±0.0059** | XGBoost ✓ |
| Precision macro (CV) | 0.4536±0.0107 | **0.4948±0.0204** | XGBoost ✓ |
| Recall macro (CV) | **0.4574±0.0111** | 0.4481±0.0060 | Random Forest ✓ |
| F1-Score macro (CV) | **0.4533±0.0110** | 0.4472±0.0082 | Random Forest ✓ |
| Cohen's Kappa (CV) | 0.2425±0.0138 | **0.2594±0.0130** | XGBoost ✓ |
| **Fitore totale** | **5** | **5** | **Barazim** |

### 7.2 Kur të Zgjidhni Secilën

**Zgjidhni XGBoost nëse:**
- Prioriteti juaj është *Accuracy* dhe *Precision* — doni të jeni të sigurt kur deklaroni "GROWING".
- Keni nevojë të minimizoni alarmet e rremë (False Positives).
- Parashikimet do të prezantohen si rekomandime investimi ku besueshmëria është kritike.

**Zgjidhni Random Forest nëse:**
- Prioriteti juaj është *Recall* dhe *F1* — doni të kapni sa më shumë raste reale të rënies.
- Kostoja e "mos-zbulimit të DECLINING" është e lartë (p.sh. për politikë publike, ndihmë ndaj sektorëve në vështirësi).
- Keni nevojë për shpjegueshmëri: Random Forest jep `feature_importances_` më të qëndrueshme.

> **Rekomandimi ynë për analizë ekonomike dhe politikë publike**: **Random Forest**, sepse Recall i lartë i DECLINING (0.588 vs 0.445) do të thotë që modeli "nuk harron" sektorët/qytetet që vërtet po tkurren.

### 7.3 Pse Saktësia është 59–63% dhe Jo 95%+?

Saktësia e moderuar nuk është dështim i modelit — është reflektim i realitetit ekonomik:

1. **Rritja ekonomike ndërvaret me faktorë makro** të panjohur: inflacioni, kursi valutor, politika fiskale, investimet e huaja direkte, çmimet globale të energjisë — asnjë prej tyre nuk është në datasetin ATK.
2. **Kufijtë e klasës janë arbitrarë**: ndryshimi midis +4.9% (STABLE) dhe +5.1% (GROWING) është matematikisht i vogël por klasifikuesi e trajton si ndryshim total.
3. **Dataset i vogël**: 3,956 rreshta pas agregimit — shumë pak për të kaptur dinamika komplekse ndërmjet 38 komunave dhe 21 sektorëve.
4. **Klasa STABLE është shumë e vështirë** ekonomikisht — "stabiliteti i vërtetë" midis ±5% është fenomen i rrallë dhe i ngjashëm me kufijtë e GROWING/DECLINING.

#### Grafiqet e gjeneruara:
- `outputs/plots/model_comparison.png` — krahasim vizual i 4 metrikave
- `outputs/plots/cv_score_boxplot.png` — shpërndarja e skoreve ndër 5 foldat (boxplot)

---

## 8. Algoritmi 3 – Linear Regression

### 8.1 Ç'është Regresioni Linear

Regresioni Linear (OLS – Ordinary Least Squares) modelon lidhjen midis tipareve dhe target-it si një **funksion linear**:

```
growth_rate = β₀ + β₁·year + β₂·prev_turnover_log1p + β₃·num_businesses_log1p
            + β₄·municipality_enc + β₅·sector_enc + ε
```

Koeficientët `β` minimizohen duke minimizuar **shumën e gabimeve quadratike (SSE)**:

```
SSE = Σᵢ (yᵢ − ŷᵢ)²    →    min w.r.t. β
```

Zgjidhja analitike: `β = (XᵀX)⁻¹ Xᵀy`

### 8.2 Pse e Kemi si Algoritëm

| Arsyeja | Shpjegimi |
|---------|-----------|
| **Benchmark referencial** | Nëse modele komplekse si XGBoost nuk e tejkalojnë rreshtin e drejtë, sinjali jo-linear mungon. |
| **Interpretueshmëria maksimale** | Koeficienti β₂ = 0.12 do të thotë: "për çdo njësi log-EUR qarkullim paraprak, rritja rritet me 12 pp". Kjo ka kuptim biznesi. |
| **Shpejt dhe i qëndrueshëm** | OLS ka zgjidhje analitike — nuk nevojitet optimizim iterativ. |
| **Detektor leakage** | Nëse Linear Regression ka R²=1.0, gjendet leakage. R² i ulët (0.05) konfirmon se dataset-i është i pastër. |

### 8.3 Rezultatet e Regresionit Linear

| Metrika | Vlera | Interpretimi |
|---------|-------|--------------|
| **MAE** | 0.5216 | Gabim mesatar 52.16 pp mbi normalen — mbi shkallën tipike të rritjes |
| **RMSE** | 0.8605 | Gabim RMS-i; penalizon gabimet e mëdha; 86 pp gabim "tipik" |
| **R²** | 0.0495 | Modeli shpjegon vetëm **4.95%** të variancës — pak mbi mesataren |
| **MAPE** | 228% | Gabimi relativ është i lartë — vlerat reale afër zero e zmadhojnë këtë metrikë |

**Gabimi mesatar absolut prej 0.52 në shkallën [-2, +5]** nënkupton që modeli parashikon "GROWING" kur mund të jetë "DECLINING", dhe kjo është e pritshme — rritja ekonomike nuk ndiqet nga një vijë e drejtë.

#### Grafiqet e gjeneruara:
- `outputs/plots/LinearRegression_actual_vs_predicted.png`
- `outputs/plots/LinearRegression_residuals.png`

---

## 9. Algoritmi 4 – Random Forest Regressor

### 9.1 Ç'është RF Regressor

Random Forest Regressor zbaton të njëjtin parim ensemble si RF Classifier, por parashikon **vlerën e vazhdueshme** `growth_rate` në vend të klasës.

Parashikimi final për çdo rresht është **mesatarja** (jo votimi) e parashikimeve të 300 pemëve:

```
growth_rate_pred = (1/300) × Σₖ fₖ(X)
```

Ku `fₖ(x)` është parashikimi i pemës k-të duke ndekur pemën sipas ndarjeve dhe duke kthyer mesataren e mostrave të fletës.

**Pse mesatarja zvogëlon variancën?**
Nëse secila pemë ka variancë σ² dhe gabimet janë të pakorreluara, gabimi i mesatares ka variancë σ²/300 — 300 herë më e vogël.

### 9.2 Hiperparametrat

```python
RF_REG_PARAMS = {
    "n_estimators":    300,   # ensemble i madh për stabilitet
    "max_depth":       10,    # thellësia e pemëve — mbështet ndërveprime komplekse
    "min_samples_split": 4,   # ndalon mbipërshtatjen e fletëve shumë specifike
    "random_state":    42,
    "n_jobs":          -1,
}
```

> Ndryshimi nga RF Classifier: nuk ka `class_weight` — regresioni nuk ka klasa.

### 9.3 Rezultatet e RF Regressor

| Metrika | Vlera | Interpretimi |
|---------|-------|--------------|
| **MAE** | **0.4932** ✓ | **Gabim mesatar 49.32 pp — modeli më i mirë MAE** |
| **RMSE** | **0.8577** ✓ | **RMSE më i ulët ndër të tre regresorët** |
| **R²** | **0.0556** ✓ | **Shpjegon 5.56% — modeli më i mirë R²** |
| **MAPE** | **205.5%** ✓ | **Gabimi relativ minimal** |

RF Regressor fiton **të gjitha 4 metrikat** ndaj Linear Regression dhe XGBoost Regressor. Megjithëse R²=0.056 duket i ulët, shqyrtimi i kontekstit tregon se kjo është performancë e mirë për parashikim të rritjes ekonomike me tipare të kufizuara.

#### Grafiqet e gjeneruara:
- `outputs/plots/RandomForestRegressor_actual_vs_predicted.png`
- `outputs/plots/RandomForestRegressor_residuals.png`
- `outputs/plots/RandomForestRegressor_feature_importance.png`

---

## 10. Algoritmi 5 – XGBoost Regressor

### 10.1 Ç'është XGBoost Regressor

XGBoost Regressor zbaton të njëjtin gradient boosting si klasifikuesi, por me **funksion objektiv regresion**:

```
objective = "reg:squarederror"
L(θ) = Σᵢ (yᵢ − ŷᵢ)² + Σₖ Ω(fₖ)
```

Minimizimi i MSE bën që modeli të jetë optimal statistikisht (ekuivalent me minimizimin e variancës së gabimit). Rregullarizimi `Ω` parandalon pemë shumë komplekse.

### 10.2 Hiperparametrat

```python
XGB_REG_PARAMS = {
    "n_estimators":    300,      # iteracione boosting
    "max_depth":       6,        # pemë të cekëta — boosting nuk ka nevojë për thellësi
    "learning_rate":   0.05,     # eta — shkalla e mësimit
    "subsample":       0.8,      # stokasticitet — 80% rreshtave per iteracion
    "colsample_bytree": 0.8,     # 80% tipareve per pemë
    "eval_metric":     "rmse",   # RMSE si metrikë e brendshme
    "random_state":    42,
}
```

### 10.3 Rezultatet e XGBoost Regressor

| Metrika | Vlera | Interpretimi |
|---------|-------|--------------|
| **MAE** | 0.5147 | Mbi RF Regressor — nuk është modeli më i mirë MAE |
| **RMSE** | 0.8931 | RMSE-i më i lartë ndër të tre regresorët |
| **R²** | **−0.0237** | **Negativ** — parashikon **keq** se mesatarja e thjeshtë |
| **MAPE** | 213.1% | Gabim relativ i lartë |

**Pse R² Negativ?**
R² negativ nënkupton që parashikimet e XGBoost janë *më larg vlerave reale* se sa parashikimi i thjeshtë "mesatarja e të gjithave". Kjo mund të ndodhë sepse:
- XGBoost overfitton tepër mbi trajnimin e vogël (3,164 rreshta)
- Boosting-u sekuencial me `learning_rate=0.05` dhe 300 iteracione mund të akumulojë gabime mbi testimin
- RF mesatarizon gabimet, ndërkohë boosting-u i amplifikon ato kur dataset-i është i vogël dhe heterogjen

Pavarësisht R² negativ, XGBoost Regressor **shërben si modeli kryesor i parashikimeve 2026** — jo sepse ka R² më të lartë, por sepse parashikimet e tij të **renditjes relative** (cili kombinim është "potencialisht" me rritje të lartë) janë të vlefshme për identifikimin e mundësive.

#### Grafiqet e gjeneruara:
- `outputs/plots/XGBoostRegressor_actual_vs_predicted.png`
- `outputs/plots/XGBoostRegressor_residuals.png`
- `outputs/plots/XGBoostRegressor_feature_importance.png`

---

## 11. Krahasimi i Regresorëve

### 11.1 Tabela e Krahasimit

| Modeli | MAE ↓ | RMSE ↓ | R² ↑ | MAPE ↓ | Fituesi |
|--------|--------|--------|------|--------|---------|
| Linear Regression | 0.5216 | 0.8605 | 0.0495 | 228.3% | — |
| **RF Regressor** | **0.4932** | **0.8577** | **0.0556** | **205.5%** | **✓ Të 4 metrikat** |
| XGBoost Regressor | 0.5147 | 0.8931 | −0.024 | 213.1% | — |

### 11.2 Diskutimi i R² të Ulët (~5%)

R²=0.056 nuk është dështim — është kufizim i njohur i parashikimit ekonomik me tipare administrative:

| Faktori | Shpjegimi |
|---------|-----------|
| **Faktorë makro mungesë** | Inflacioni (2022: 11.6%), kursi EUR/USD, çmimet e energjisë pas luftës Ukrainë, rimëkëmbja post-COVID — asnjë nuk është i disponueshëm në ATK |
| **Ngjarje rrëzuese** | Pandemia 2020–2021 ka shkaktuar luhatje extreme të parashikueshme vetëm me të dhëna epidemiologjike |
| **Heterogjenitet i lartë** | 38 komuna × 21 sektorë × 6 vite = 4,788 kombinime me dinamika shumë të ndryshme |
| **Grupi i vogël** | 3,956 rreshta është i pamjaftueshëm për modele komplekse non-lineare |
| **Klasifikimi më i mirë se regresioni** | Pyetja "a rritet apo jo?" (59–63% accuracy) është bërë njohshme nga tiparet — por vlera ekzakte (sa %) jo |

**Rekomandimi:** Për vendimmarrje strategjike, **klasifikuesit** (RF, XGBoost) janë modelet kryesore. Regresorët shërbejnë si mbështetës për **renditje relative** të mundësive.

#### Grafiku i gjeneruar:
- `outputs/plots/regressor_comparison.png` — bar chart MAE / RMSE / R² me fituesin të theksuar

---

## 12. Algoritmi 6 – K-Means Clustering

### 12.1 Ç'është K-Means

K-Means është algoritmi klasik i grupimit të pambikëqyrur (unsupervised). Ai ndan `n` pika në `k` grupe duke minimizuar **shumën e distancave quadratike** nga çdo pikë tek centroidi i grupit të saj:

```
Inertia = Σₖ Σᵢ∈Cₖ ||xᵢ − μₖ||²
```

**Algoritmi EM (Expectation-Maximization):**
1. **Initialization:** Zgjidh `k` centroide fillestare (metoda K-Means++)
2. **E-step (Assignment):** Çdo pikë caktohet grupit me centroid-in më të afërt
3. **M-step (Update):** Çdo centroid ricaktohet si mesatarja e pikave të grupit
4. **Përsërit** deri në konvergjencë (centroidet nuk lëvizin)

**K-Means++** zgjedh centroidet fillestare me probabilitet proporcional me distancën — parandalon konvergjencën në lokale minimume.

### 12.2 Dy Mënyra të Aplikimit

#### Mënyra 1: K-Means mbi Matricën e Tipareve

K-Means aplikohet drejtpërdrejt mbi matricën e skaluar `X_scaled` (3,956 × 5). Çdo rresht (një (komunë, sektor, vit) kombinim) ndahet në 4 grupe sipas afërsisë në hapësirën e tipareve.

**Qëllimi:** Gjetja e grupimeve natyrale — a kanë disa kombinime profile shumë të ngjashme edhe pa etiketë?

#### Mënyra 2: K-Means mbi Trajektore Rritjeje (aplikimi i specializuar)

Ndërtohet një matricë pivot ku:
- Rreshtat = komunë (ose sektor)
- Kolonat = vitet 2020–2025
- Çeliza = shkalla mesatare YoY e rritjes

```
Komuna        2020    2021    2022    2023    2024    2025
PRISHTINË    -0.15   +0.28   +0.12   +0.08   +0.11   +0.09
FERIZAJ      -0.20   +0.35   +0.18   +0.05   +0.14   +0.07
...
```

K-Means grupëzon komuna me **profile të ngjashme ndër vite** — jo vetëm në një moment, por të gjithë trajektorën.

### 12.3 Gjetja e k Optimal

Tre metrika të kombinuara:

**Inertia / Elbow Method:** Inertia zvogëlohet me rritjen e k. Kërkohet "bërryli" — pika ku zvogëlimi ngadalësohet.

**Silhouette Score:** Masë e kohezionit brenda grupit dhe ndarjes nga grupet e tjera.
```
s(i) = (b(i) − a(i)) / max(a(i), b(i))
```
- `a(i)` = distanca mesatare nga pikat e grupit të njëjtë
- `b(i)` = distanca mesatare nga grupi fqinj
- Vlera 1 = grupim perfekt, 0 = kufij të paqartë, negativ = pikë e greqisur

**Davies-Bouldin Index:** Mesataret e raporteve (shpërhapja brenda grupit) / (distanca mes centroideve). Më i ulët = më mirë.

| k | Inertia | Silhouette ↑ | Davies-Bouldin ↓ |
|---|---------|-------------|-----------------|
| 2 | 14,793 | **0.2158** | 1.6670 |
| 3 | 12,716 | 0.1890 | 1.7108 |
| **4** | **11,301** | **0.1806** | **1.5839** |
| 5 | 10,137 | 0.1840 | 1.4628 |
| 6 | 9,247 | 0.1927 | 1.3844 |
| 7 | 8,609 | 0.1880 | 1.3200 |
| 8 | 7,954 | 0.1955 | 1.3682 |

**Pse u zgjodh k=4?**
- Silhouette maksimal është në k=2, por 2 grupe janë tepër të thjeshtuara ekonomikisht.
- Bërryli i Inertia shfaqet qartë pas k=4 (diferenca zvogëlohet drastikisht).
- 4 grupe ofrojnë interpretim natyral: *rritje e fortë / rritje e moderuar / stagnacion / rënie*.
- Vlera e Davies-Bouldin për k=4 (1.584) është e pranueshme dhe nën k=2,3.

### 12.4 Rezultatet Finale K-Means (k=4)

| Metrika | Vlera | Interpretimi |
|---------|-------|--------------|
| Inertia | 11,300.97 | Ngjeshmëria totale brenda grupeve |
| **Silhouette Score** | **0.1806** | Grupim i pranueshëm; klasa ekonomike janë natyrshëm të mbivendosura |
| **Davies-Bouldin** | **1.5839** | Ndarje e moderuar midis grupeve |

**Klasterimi i Trajektoreve të Komunave** (top 20 komuna):
- Silhouette = 0.235, Davies-Bouldin = 1.143
- Komuna si Prishtina, Ferizaj, Prizreni ndahen nga komunat e vogla me profile shumë të ndryshme

**Klasterimi i Trajektoreve të Sektorëve** (top 15 sektorë):
- Silhouette = 0.184, Davies-Bouldin = 0.882
- Ndarje më e qartë se komunat — sektorët kanë profile specifike të pandryshueshme (p.sh. ndërtimtaria ka vzgjedhje ciklist të dallueshme)

### 12.5 Hiperparametrat

```python
km = KMeans(
    n_clusters  = 4,        # k i zgjedhur nga analiza elbow+silhouette+DB
    random_state = 42,      # riprodhim
    n_init      = 10,       # 10 inicializime të ndryshme — merr rezultatin më të mirë
    max_iter    = 300,      # iteracione maksimale
)
```

#### Grafiqet e gjeneruara:
- `outputs/plots/kmeans_elbow_silhouette.png`
- `outputs/plots/municipality_trajectory_clusters.png`
- `outputs/plots/primary_sector_trajectory_clusters.png`
- `outputs/plots/pca_kmeans_clusters.png`

---

## 13. Algoritmi 7 – Principal Component Analysis (PCA)

### 13.1 Ç'është PCA

PCA (Principal Component Analysis) është teknikë klasike e reduktimit të dimensionalitetit. Gjen drejtimin e variancës maksimale në hapësirën shumëdimensionale dhe projekton të dhënat mbi *komponentë kryesorë* ortogonalë.

**Matematika:**

Gjen vetorët dhe vlerat e veta të matricës së kovariancës:
```
C = (1/n) XᵀX    →    C·v = λ·v
```

- `v` = vektori i vet (drejtimi i komponentit kryesor)
- `λ` = vlera e vet (varianca e shpjeguar nga ai komponent)
- Komponentët renditen sipas `λ` zbritës — PC1 ka variancën më të madhe

**Projekcioni:**
```
X_pca = X · V[:, :2]     (reduktim në 2D)
```

**Varianca e shpjeguar:**
```
explained_ratio[k] = λₖ / Σᵢ λᵢ
```

### 13.2 Dy Mënyrat e Aplikimit

#### PCA 2D — Vizualizim

Projekton matricën e skaluar X (3,956 × 5) në 2 dimensione:

```
PC1 shpjegon 39.7% të variancës
PC2 shpjegon 20.2% të variancës
─────────────────────────────
2D kumulativ: 59.9%
```

Dy grafikët 2D:
1. **PCA me etiketat reale** — pikët ngjyrosen sipas GROWING/STABLE/DECLINING
2. **PCA me grupet K-Means** — pikët ngjyrosen sipas klasterit 0/1/2/3

Grafiku PCA + K-Means tregon nëse grupet K-Means korrespondojnë me struktura reale në hapësirën e tipareve.

#### PCA Full — Analiza e Variancës

Trajnohet PCA mbi të gjitha 5 komponentët. Kurba kumulatiove tregon:

| Komponentët | Varianca Kumulative |
|-------------|---------------------|
| PC1 | 39.7% |
| PC1–PC2 | 59.9% |
| PC1–PC3 | ~78% |
| **PC1–PC4** | **≥95%** |
| PC1–PC5 | 100% |

**Gjetja kryesore:** Nevojiten vetëm **4 komponentë** për të mbajtur 95% të informacionit — kjo tregon se 5 tiparet tona nuk janë shumë të korreluar ndërmjet tyre dhe ka informacion real në çdo dimension.

### 13.3 Pse e Kemi PCA

| Arsyeja | Shpjegimi |
|---------|-----------|
| **Vizualizimi i pamundur** | 5D nuk mund të shikohet; 2D ju lejon të shihni grupe, outliers dhe separueshmëri klase |
| **Validimi i K-Means** | Nëse grupet K-Means janë qartë të ndarë në 2D, ndarjet janë reale dhe jo artificiale |
| **Detektimi i redundancës** | Nëse 2 komponentë shpjegojnë 95% (do ishte shumë), tiparet janë shumë të korreluar — Standard: 4 për 95% tregon diversitet të mirë informacioni |
| **Kompresim i mundshëm** | Nëse RAM ose shpejtësia janë problem, mund të reduktohet në 4 dimensione pa humbur informacion esencial |

### 13.4 Hiperparametrat

```python
# PCA 2D
pca = PCA(n_components=2, random_state=42)

# PCA Full
pca_full = PCA(random_state=42)  # mban të gjitha komponentët
```

#### Grafiqet e gjeneruara:
- `outputs/plots/pca_variance_explained.png`
- `outputs/plots/pca_true_labels.png`
- `outputs/plots/pca_kmeans_clusters.png`

---

## 14. Analiza Kryqëzore: Sektor × Komunë (Heatmaps)

Kjo është analiza **ekonomikisht më e rëndësishme** e projektit — përgjigjet direkt pyetjeve:

> *"Si ka performuar sektori i ndërtimtarisë në Ferizaj krahasuar me Prishtinën?"*
> *"Cilët sektorë kanë rritje të qëndrueshme në të gjitha komunat?"*
> *"Kur ndodhi rimëkëmbja post-COVID sipas sektorit dhe qytetit?"*

### 14.1 Lloji 1: Komunë × Vit

Çdo rresht = komunë, çdo kolonë = vit, çdo qelizë = shkalla mesatare YoY e rritjes:

```
Metodologjia:
  pivot = growth_df
    .groupby(['municipality', 'year'])['growth_rate']
    .mean()
    .unstack('year')

Shkalla ngjyrash: RdYlGn [-50%, +50%]
  🟢 Gjelbër e fortë = +50%+ rritje
  🟡 Verdhë = afër 0% (stagnacion)
  🔴 E kuqe e fortë = -50%+ rënie
```

Shpjegon pyetjen: *"Në cilin vit ka pasur kjo komunë rritjen/rënien e saj më të madhe?"*

Grafiku: `outputs/plots/municipality_growth_by_year.png`

### 14.2 Lloji 2: Sektor × Vit

Identik me mësipër por rreshtat janë sektorët:

Shpjegon pyetjen: *"Cili sektor u rikuperua më shpejt pas 2020?"*

Grafiku: `outputs/plots/sector_growth_by_year.png`

### 14.3 Lloji 3: Sektor × Komunë (Kryqëzimi Kryesor)

Çdo rresht = sektor, çdo kolonë = komunë, çdo qelizë = shkalla mesatare e rritjes (të gjitha vitet / vitin e zgjedhur):

```
Metodologjia:
  pivot = growth_df
    .groupby(['primary_sector', 'municipality'])['growth_rate']
    .mean()
    .unstack('municipality')

Filtrim: top 20 komuna × top 15 sektorë sipas qarkullimit total
```

Shpjegon pyetjen: *"Cilët sektorë kanë potencial vetëm në qytete të caktuara?"*

**Katër versione të gjeneruara:**

| Skedari | Periudha |
|---------|---------|
| `sector_x_municipality_growth_all.png` | 2020–2025 (mesatare) |
| `sector_x_municipality_growth_2023.0.png` | Vetëm 2023 |
| `sector_x_municipality_growth_2024.0.png` | Vetëm 2024 |
| `sector_x_municipality_growth_2025.0.png` | Vetëm 2025 |

Secili heatmap ruhet edhe si CSV në `outputs/metrics/` për analiza të mëtejshme.

---

## 15. Parashikimet e Tregut 2026

### 15.1 Metodologjia e Parashikimit

Duke mbajtur parimin e *data leakage*, parashikimi për 2026 ndërtohet kështu:

```
Viti 2025 (i njohur) → shërben si "viti paraprak" → parashikim 2026

Për çdo (komunë, sektor) que ekzistoi në 2025:
  prev_turnover_log1p = log1p(total_turnover_2025)   ← i njohur
  num_businesses_log1p = log1p(num_businesses_2025)  ← i njohur
  year = 2026                                         ← target kohor
  municipality_enc = kodim i njëjtë si trajnimi      ← i njohur
  sector_enc = kodim i njëjtë si trajnimi            ← i njohur

→ Të 5 modelet parashikojnë për 659 kombinime
```

### 15.2 Output-i i Parashikimit

Skedari `outputs/metrics/predictions_2026.csv` ka **659 rreshta** me 12 kolona:

| Kolona | Shpjegimi |
|--------|-----------|
| `municipality` | Emri i komunës |
| `primary_sector` | Sektori ekonomik |
| `turnover_2025_eur` | Qarkullimi bazë 2025 (€) |
| `forecast_year` | 2026 |
| `growth_class_RandomForest` | GROWING / STABLE / DECLINING (RF) |
| `growth_class_XGBoost` | GROWING / STABLE / DECLINING (XGB) |
| `growth_rate_LinearRegression` | Parashikimi % (Linear Reg.) |
| `growth_rate_RandomForestRegressor` | Parashikimi % (RF Reg.) |
| `growth_rate_XGBoostRegressor` | Parashikimi % (XGB Reg.) |
| `est_turnover_LinearRegression` | Qarkullimi i vlerësuar 2026 (€) |
| `est_turnover_RandomForestRegressor` | Qarkullimi i vlerësuar 2026 (€) |
| `est_turnover_XGBoostRegressor` | Qarkullimi i vlerësuar 2026 (€) |

### 15.3 Top 10 Mundësi Rritjeje 2026 (XGBoost Regressor)

| # | Komuna | Sektori | Rritja e Parashikuar | Est. Qarkullim 2026 |
|---|--------|---------|---------------------|---------------------|
| 1 | SUHAREKË | Aktivitetet e pasurive të paluajtshme | +479.5% | — |
| 2 | GJAKOVË | Aktivitetet e pasurive të paluajtshme | +214.4% | — |
| 3 | FERIZAJ | Aktivitetet e pasurive të paluajtshme | +180.2% | — |
| 4 | MITROVICË | Aktivitetet e pasurive të paluajtshme | +142.7% | — |
| 5 | PRIZREN | Aktivitetet e pasurive të paluajtshme | +138.1% | — |
| 6 | VUSHTRRI | Aktivitetet e pasurive të paluajtshme | +135.0% | — |
| 7 | LIPJAN | Aktivitetet e pasurive të paluajtshme | +132.8% | — |
| 8 | GJILAN | Aktivitetet e pasurive të paluajtshme | +131.9% | — |
| 9 | DRENAS | Aktivitetet e pasurive të paluajtshme | +127.4% | — |
| 10 | PEJË | Aktivitetet e pasurive të paluajtshme | +122.6% | — |

> **⚠ Paralajmërim i Rëndësishëm:** Parashikimet mbi +100% janë **tregues trendi**, jo vlera absolute. R²≈5% e modelit të regresionit nënkupton pasiguri të lartë. Ato janë të dobishme për **renditje relative** (cilat komuna kanë potencial më të lartë), jo si numra të saktë. Sektori i pasurive të paluajtshme ka variancë shumë të lartë historike dhe parashikimet ektreme reflektojnë momentin e vitit 2025.

### 15.4 Vizualizimet e Parashikimeve

| Grafiku | Shpjegimi |
|---------|-----------|
| `predicted_growth_2026_XGBoostRegressor.png` | Heatmap sektor × komunë me parashikimet 2026 (ngjyrë e gjelbër = rritje, e kuqe = rënie) |
| `predicted_top_2026_XGBoostRegressor.png` | Top 15 komunat me rritje + top 15 sektorët me rritje (bar charts horizontale) |

---

## 16. Udhëzime për Ekzekutim

### 16.1 Kërkesat e Sistemit

- Python 3.10 ose më i ri
- RAM ≥ 4 GB (të dhënat: 99.2 MB CSV → ~500 MB në memorie)
- macOS/Linux/Windows

### 16.2 Instalimi

```bash
# Klonimi i repozitorit
git clone <url-repozitorit>
cd "group-3-machine-learning"

# Navigimi në Faza_2
cd Faza_2

# Krijimi i mjedisit virtual (rekomandohet)
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# ose: venv\Scripts\activate  # Windows

# Instalimi i varësive
pip install -r requirements.txt
```

**Varësitë kryesore:**

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
joblib>=1.3
imbalanced-learn>=0.11
```

> **Shënim macOS:** Nëse XGBoost dështon me `libxgboost.dylib could not be loaded`, ekzekutoni: `brew install libomp`

### 16.3 Parakusht: Faza 1

Faza 2 lexon datasetin e pastruar nga Faza 1:
```
Faza 1/outputs/clean/model_ready.csv  (99.2 MB)
```

Nëse skedari nuk ekziston, ekzekutoni Fazën 1 fillimisht:
```bash
cd "../Faza 1"
python3 -m src.main --step all
```

### 16.4 Ekzekutimi i Pipeline-it të Plotë

```bash
cd Faza_2

# Të 7 algoritmet + të 5 hapat + parashikimet 2026
python -m src.main --step all
```

Koha e ekzekutimit: **3–8 minuta** (varësisht nga CPU)

### 16.5 Ekzekutimi i Hapave Individualë

```bash
# Step A – Klasifikuesit supervized (RF + XGBoost)
python -m src.main --step supervised

# Step B – Regresorët (Linear Reg + RF Reg + XGB Reg)
python -m src.main --step regression

# Step C – Unsupervised (K-Means sweep + K-Means final + PCA)
python -m src.main --step unsupervised

# Step D – Analiza kryqëzore (heatmaps sektor × komunë × vit)
python -m src.main --step analysis

# Step E – Parashikimet 2026 (kërkon modelet të trajnuara nga A dhe B)
python -m src.main --step predictions
```

### 16.6 Ngarkimi i Modeleve të Ruajtura

Nëse doni të ekzekutoni vetëm `--step predictions` pa ri-trajnuar:

```python
from src.supervised import load_model

rf   = load_model("random_forest")
xgb  = load_model("xgboost")
lr   = load_model("linear_regression")
rfr  = load_model("rf_regressor")
xgbr = load_model("xgb_regressor")
```

Modelet ruhen si skedarë `.pkl` me `joblib` në `outputs/models/`.

---

## 17. Struktura e Skedarëve dhe Output-et

### 17.1 Kodi Burimor

```text
Faza_2/
├── requirements.txt              # varësitë Python
└── src/
    ├── __init__.py
    ├── config.py                 # konfigurim i centralizuar
    ├── data_loader.py            # ngarkimi dhe ndërtimi i dataset-it të rritjes
    ├── feature_engineering.py   # enkodimi dhe skalimi
    ├── supervised.py             # 5 modelet supervized
    ├── unsupervised.py           # K-Means + PCA
    ├── evaluation.py             # të gjitha metrikat dhe grafikët
    └── main.py                   # CLI – hyrja e pipeline-it
```

**Funksionet kryesore sipas skedarit:**

| Skedari | Funksioni | Çfarë bën |
|---------|-----------|-----------|
| `data_loader.py` | `load_data()` | Lexon model_ready.csv |
| `data_loader.py` | `build_growth_dataset()` | 650K rreshta → 3,956 (agregim + lag + klasifikim) |
| `data_loader.py` | `build_prediction_input()` | Ndërton 659 rreshta parashikimi 2026 |
| `data_loader.py` | `build_growth_pivot()` | Matrica trajektore grup × vit |
| `feature_engineering.py` | `encode_and_scale()` | LabelEncoder + StandardScaler → X_scaled |
| `feature_engineering.py` | `encode_for_prediction()` | Aplikon enkoduesit e trajnimit mbi 2026 |
| `supervised.py` | `train_random_forest()` | RF Classifier |
| `supervised.py` | `train_xgboost()` | XGBoost Classifier |
| `supervised.py` | `train_linear_regression()` | OLS Linear Regression |
| `supervised.py` | `train_rf_regressor()` | RF Regressor |
| `supervised.py` | `train_xgb_regressor()` | XGBoost Regressor |
| `unsupervised.py` | `find_optimal_k()` | K-Means sweep k=2..8 |
| `unsupervised.py` | `train_kmeans()` | K-Means final k=4 |
| `unsupervised.py` | `cluster_growth_trajectories()` | Klasterim trajektoresh |
| `unsupervised.py` | `apply_pca()` | PCA 2D |
| `unsupervised.py` | `apply_pca_full()` | PCA full për kurbën e variancës |
| `evaluation.py` | `evaluate_classifier()` | Raporti i klasifikimit + CSV |
| `evaluation.py` | `cross_validate_models()` | 5-fold CV me Kappa |
| `evaluation.py` | `build_verdict_table()` | Tabela kokë-më-kokë |
| `evaluation.py` | `evaluate_regressor()` | MAE, RMSE, R², MAPE |
| `evaluation.py` | `save_predictions()` | Parashikimet 2026 — të 5 modelet |

### 17.2 Output-et e Gjeneruara

```text
outputs/
├── models/
│   ├── random_forest.pkl              # RF Classifier (300 pemë, balanced)
│   ├── xgboost.pkl                    # XGBoost Classifier
│   ├── linear_regression.pkl          # OLS Linear Regression
│   ├── rf_regressor.pkl               # RF Regressor
│   └── xgb_regressor.pkl              # XGBoost Regressor
│
├── metrics/
│   ├── RandomForest_classification_report.csv    # Precision/Recall/F1 per klasë
│   ├── XGBoost_classification_report.csv
│   ├── model_comparison.csv                      # Accuracy/Precision/Recall/F1 (RF vs XGB)
│   ├── cross_validation_results.csv              # 5-fold CV mean±std
│   ├── algorithm_verdict.csv                     # Tabela kokë-më-kokë me ✓
│   ├── LinearRegression_regression_metrics.csv   # MAE/RMSE/R²/MAPE
│   ├── RandomForestRegressor_regression_metrics.csv
│   ├── XGBoostRegressor_regression_metrics.csv
│   ├── regressor_comparison.csv                  # Krahasimi i 3 regresorëve
│   ├── kmeans_sweep_metrics.csv                  # Inertia/Silhouette/DB per k=2..8
│   ├── unsupervised_metrics.csv                  # Metrikat finale K-Means + PCA
│   ├── municipality_growth_by_year.csv           # Heatmap komunë × vit (vlerat)
│   ├── sector_growth_by_year.csv                 # Heatmap sektor × vit (vlerat)
│   ├── sector_x_municipality_growth_all.csv      # Kryqëzim të gjitha vitet
│   ├── sector_x_municipality_growth_2023.0.csv
│   ├── sector_x_municipality_growth_2024.0.csv
│   ├── sector_x_municipality_growth_2025.0.csv
│   └── predictions_2026.csv                      # 659 parashikime nga 5 modele
│
└── plots/
    ├── RandomForest_confusion_matrix.png          # Matrica e konfuzionit (normalizuar)
    ├── RandomForest_roc_curves.png                # ROC One-vs-Rest për 3 klasa
    ├── RandomForest_feature_importance.png        # Rëndësia e tipareve
    ├── XGBoost_confusion_matrix.png
    ├── XGBoost_roc_curves.png
    ├── XGBoost_feature_importance.png
    ├── model_comparison.png                       # Bar chart RF vs XGB (4 metrika)
    ├── cv_score_boxplot.png                       # Boxplot 5-fold ndër metrika
    ├── LinearRegression_actual_vs_predicted.png   # Scatter vlerat reale vs parashikimet
    ├── LinearRegression_residuals.png             # Mbetjet vs fitted + histogram
    ├── RandomForestRegressor_actual_vs_predicted.png
    ├── RandomForestRegressor_residuals.png
    ├── RandomForestRegressor_feature_importance.png
    ├── XGBoostRegressor_actual_vs_predicted.png
    ├── XGBoostRegressor_residuals.png
    ├── XGBoostRegressor_feature_importance.png
    ├── regressor_comparison.png                   # Bar chart MAE/RMSE/R² × 3 modele
    ├── kmeans_elbow_silhouette.png                # Inertia + Silhouette + DB vs k
    ├── municipality_trajectory_clusters.png       # Komunat sipas profilit 2020–2025
    ├── primary_sector_trajectory_clusters.png     # Sektorët sipas profilit 2020–2025
    ├── pca_true_labels.png                        # PCA 2D me ngjyrë sipas growth_class
    ├── pca_kmeans_clusters.png                    # PCA 2D me ngjyrë sipas K-Means
    ├── pca_variance_explained.png                 # Kurba e variancës (individual + kumulative)
    ├── municipality_growth_by_year.png            # Heatmap komunë × vit
    ├── sector_growth_by_year.png                  # Heatmap sektor × vit
    ├── sector_x_municipality_growth_all.png       # Kryqëzim kryesor (2020–2025)
    ├── sector_x_municipality_growth_2023.0.png
    ├── sector_x_municipality_growth_2024.0.png
    ├── sector_x_municipality_growth_2025.0.png
    ├── predicted_growth_2026_XGBoostRegressor.png # Heatmap parashikimesh 2026
    └── predicted_top_2026_XGBoostRegressor.png    # Top 15 komunë + top 15 sektor
```

---

## 18. Lidhja me Fazën 1 dhe Ndikimi i Parapërpunimit

Faza 1 nuk ishte vetëm "pastrimi i të dhënave" — çdo vendim atje kishte pasoja të drejtpërdrejta në cilësinë e analizës ekonomike:

### 18.1 Heqja e 102,977 Duplikateve

**Ndikimi direkt:** Duplikatat fryenin artificialisht qarkullimin e disa (komunë, sektor, vit) kombinimeve. Nëse Ferizaj-Ndërtimtaria 2023 kishte 500 rreshta duplikatë, qarkullimi agreguar do ishte 500× i fryerë, dhe growth_rate do ishte jorealike.

**Pas heqjes:** Agregimi pasqyron vlerën e vërtetë ekonomike dhe growth_rate është llogaritje e saktë e tregut.

### 18.2 Transformimi log1p

**Arsyeja statistike:** Prishtina ka qarkullim ~51 miliardë EUR, ndërkohë komuna të vogla kanë ~50 milionë — një raport 1,000:1. Nëse K-Means llogarit distanca Euklidiane mbi vlerat bruto, Prishtina dominon kompletisht çdo grupim. log1p e ngjeshin këtë hendek:
```
log1p(51,000,000,000) ≈ 24.65
log1p(50,000,000)     ≈ 17.73
Raporti pas log1p:    ≈ 1.4:1  (nga 1000:1)
```

### 18.3 Normalizimi NFKC i Tekstit

**Çfarë ndodh pa normalizim:** `"Prishtinë"` dhe `"PRISHTINË"` dhe `"Prishtine"` trajtohen si tre komuna të ndryshme. Pas agregimt, komuna Prishtina do ishte fragmentuar në dhjetëra grupe me qarkullim të ulët secila — duke shkatërruar çdo analizë trajektore.

**Pas normalizimit:** Të gjitha variacionet e shqipes (ë, ç, dhe kapitalizmi) normalizohen → grupimi korrekt.

### 18.4 Analiza e Outliers (e Raportuara, Pa Hequr)

Faza 1 zbuloi 96,473 rreshta me qarkullim ekstream (IQR method). Kjo shpjegon pse disa kombinime kanë growth_rate ekstreme (+200%, -80%) — biznese individuale të mëdha që hyjnë ose dalin nga tregu. Prandaj growth_rate **u clip-ua** në [-2.0, +5.0] për të parandaluar vlerat e papjekura nga dominimi i trajnimit.

### 18.5 Agregimi si Zgjidhje e Imbalancës

Imbalanca origjinale e `registration_status` (SH.P.K. 49%, KOMPANI E HUAJ 1.7%) bëhet e parëndësishme: pasi agregohen të gjithë tatimpaguesit sipas (komunë, sektor, vit), lloji ligjor i entitetit **nuk ekziston** si kolona. Target-i i ri `growth_class` ka shpërndarje të ndryshme (GROWING 58.7%, DECLINING 31%, STABLE 10.3%) — e trajtuar me `class_weight="balanced"`.

---

## Përmbledhje Finale

| Aspekti | Detajet |
|---------|---------|
| **Dataset final** | 3,956 rreshta × 5 tipare (pas agregimit të 650K rreshtave) |
| **Algoritme të trajnuara** | 7 (2 klasifikues + 3 regresorë + K-Means + PCA) |
| **Modele të ruajtura** | 5 skedarë `.pkl` |
| **Grafiqe të gjeneruara** | 30+ grafikë PNG |
| **Parashikime 2026** | 659 kombinime (komunë × sektor) nga 5 modele |
| **Klasifikuesi më i mirë (Accuracy)** | XGBoost: 62.5% |
| **Klasifikuesi më i mirë (Recall/F1)** | Random Forest: Recall 47.1%, F1 46.9% |
| **Regresori më i mirë** | RF Regressor: MAE=0.493, R²=0.056 |
| **K-Means optimal k** | k=4 (Silhouette=0.181, DB=1.584) |
| **PCA 2D varianca** | 59.9% (4 komponentë → 95%) |

---

*Faza 2 e projektit Group 3 – Machine Learning, Semestri 2, Master FIEK, Universiteti i Prishtinës.*
*Studentët: Enis Hoxha · Fisnik Hazrolli · Endri Binaku*
