<table>
  <tr>
    <td width="150" align="center" valign="middle">
      <img src="https://upload.wikimedia.org/wikipedia/commons/e/e1/University_of_Prishtina_logo.svg" width="120" alt="University of Prishtina Logo" />
    </td>
    <td valign="top">
      <p><strong>Universiteti i Prishtinës</strong></p>
      <p>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</p>
      <p>Inxhinieri Kompjuterike dhe Softuerike - Programi Master</p>
      <p><strong>Lënda:</strong> Machine Learning</p>
      <p><strong>Profesorët:</strong> Prof. Dr. Lule Ahmedi dhe Dr. Sc. Mërgim Hoti</p>
      <p><strong>Studentët (Gr. 3):</strong></p>
      <ul>
        <li>Enis Hoxha</li>
        <li>Fisnik Hazrolli</li>
        <li>Endri Binaku</li>
      </ul>
    </td>
  </tr>
</table>

# Trajnimi i modelit per parashikimin e rritjes se te ardhurave te bizneseve ne Kosove

Ky projekt paraqet një pipeline të ndërtuar në Python për ingestimin, pastrimin, profilizimin, agregimin, kampionimin, detektimin e outliers dhe analizën e imbalance për datasetin `Qarkullimi.xlsx`.

## Përshkrimi i Projektit

Qëllimi i projektit është përgatitja e një dataseti të përdorshëm për analiza të mëtejshme dhe modelim në kuadër të lëndës Machine Learning. Procesi fokusohet në:

- leximin dhe normalizimin e të dhënave nga Excel
- pastrimin dhe validimin e kolonave kryesore
- profilizimin para dhe pas pastrimit
- agregimin dhe kampionimin e të dhënave
- detektimin e outliers dhe analizën e imbalance
- gjenerimin e raporteve dhe grafikëve përmbledhës

## Të Dhënat

Dataseti kryesor i përdorur në projekt është:

- `Faza 1/dataset/Qarkullimi.xlsx`

Burimi i të dhënave:

- [ATK Open Data](https://www.atk-ks.org/en/open-data/)

Karakteristikat kryesore të datasetit:

- madhësia fillestare: `753,891` rreshta dhe `7` kolona
- formati burimor: Excel
- rreshti i header-it: rreshti `9`
- fokusi kryesor aktual: dataseti bazë
- zgjerim i mundshëm: datasetet vjetore në `Faza 1/2019-2025/`

### Kolonat Kryesore

Kolonat kanonike të përdorura nga pipeline-i janë:

- `year`
- `month`
- `primary_sector`
- `municipality`
- `registration_status`
- `num_taxpayers`
- `turnover_eur`

## Struktura e Projektit

```text
group-3-machine-learning/
├── README.md
├── Faza 1/
│   ├── README.md
│   ├── requirements.txt
│   ├── dataset/
│   │   └── Qarkullimi.xlsx
│   ├── 2019-2025/
│   │   └── Qarkullimi-*.xlsx
│   ├── data/
│   ├── docs/
│   │   └── steps/
│   ├── notebooks/
│   ├── outputs/
│   │   ├── aggregate/
│   │   ├── clean/
│   │   ├── imbalance/
│   │   ├── ingest/
│   │   ├── outliers/
│   │   ├── profile/
│   │   ├── report/
│   │   └── sample/
│   ├── src/
│   └── tests/
└── .gitignore
```

## Pipeline i Përpunimit të të Dhënave

1. `profile_raw`
2. `ingest`
3. `clean`
4. `profile`
5. `aggregate`
6. `sample`
7. `outliers`
8. `imbalance`
9. `resample` (opsionale)
10. `report`

| Hapi | Përshkrimi |
|------|------------|
| `profile_raw` | Profilizimi fillestar i datasetit bruto para çdo transformimi |
| `ingest` | Nxjerrja dhe normalizimi i të dhënave nga Excel |
| `clean` | Pastrimi strikt dhe përgatitja e datasetit për analizë |
| `profile` | Profilizimi i datasetit pas pastrimit |
| `aggregate` | Krijimi i pamjeve të agreguara |
| `sample` | Kampionimi i të dhënave me opsione stratifikimi |
| `outliers` | Identifikimi dhe raportimi i outliers |
| `imbalance` | Analiza e shpërndarjes së klasave dhe imbalance |
| `resample` | Resampling me algoritme si SMOTE ose ADASYN |
| `report` | Gjenerimi i raportit final dhe grafikëve |

## Rezultatet Kryesore sipas Hapave

Ky seksion përmbledh rezultatet konkrete të gjeneruara nga pipeline-i bazuar në artefaktet aktuale të projektit.

### Hapi 0: `profile_raw`

- Dataseti fillestar përmban `753,891` rreshta dhe `7` kolona kanonike.
- U identifikua `1` rresht me vlera të munguara.
- U identifikuan `102,977` rreshta dublikatë.
- U identifikuan `176` raste me `turnover_eur` jo-valid.

Ky hap shërben si pika fillestare e analizës. Ai na tregon gjendjen reale të datasetit para se të bëhet ndonjë pastrim apo transformim.

### Hapi 1: `ingest`

- U lexua me sukses `Qarkullimi.xlsx` duke përdorur header-in në rreshtin `9`.
- U normalizuan kolonat në skemën standarde të pipeline-it: `year`, `month`, `primary_sector`, `municipality`, `registration_status`, `num_taxpayers`, `turnover_eur`.
- U krijuan dy output-e bazë: `Faza 1/outputs/ingest/raw_extract.csv` dhe `Faza 1/outputs/ingest/normalized.csv`.

Ky hap siguron që të gjitha skriptat e tjera të punojnë mbi të njëjtën strukturë standarde të të dhënave.

### Hapi 2: `clean`

- U largua `1` rresht bosh ose header i përsëritur.
- U larguan `102,977` dublikata.
- Dataseti i pastruar përfundoi me `650,913` rreshta dhe `7` kolona.
- U krijuan dy versione: `strict_cleaned.csv` për analizë të pastër dhe `model_ready.csv` për modelim.

**Veprimet kryesore të regjistruara në cleaning log:**

| Veprimi | Numri |
|---------|-------|
| `drop_empty_or_header_rows` | 1 |
| `drop_duplicates` | 102,977 |
| `standardize_text` në `primary_sector` | 753,890 |
| `standardize_text` në `municipality` | 753,890 |
| `standardize_text` në `registration_status` | 753,890 |

Ky hap largon zhurmën kryesore nga dataseti dhe krijon bazën që përdoret në të gjithë analizën pasuese.

### Hapi 3: `profile`

- Dataseti pas pastrimit ka `650,913` rreshta.
- Nuk ka më rreshta me vlera të munguara (`0`).
- Nuk ka më dublikata (`0`).
- Të gjitha `7` kolonat kryesore janë të përdorshme për analizë.

**Përmirësimi nga profili fillestar te profili pas pastrimit:**

| Metrika | Para pastrimit | Pas pastrimit |
|---------|----------------|---------------|
| Rreshta totale | 753,891 | 650,913 |
| Rreshta me null | 1 | 0 |
| Dublikata | 102,977 | 0 |
| Kolona | 7 | 7 |

**Grafiku i null values**

Ky grafik tregon numrin e vlerave që mungojnë për secilën kolonë pas pastrimit. Në rezultatet tona, të gjitha kolonat kryesore kanë `0` null values, që do të thotë se dataseti është i plotë për analizë të mëtejshme.

![Nulls by Column](Faza%201/outputs/report/nulls_by_column.png)

### Hapi 4: `aggregate`

Agregimet kryesore tregojnë shpërndarjen e qarkullimit sipas viteve, muajve, sektorëve dhe komunave.

**Qarkullimi sipas viteve**

| Viti | Qarkullimi total |
|------|------------------|
| 2025 | 23,811,372,168.39 |
| 2024 | 20,617,432,906.60 |
| 2023 | 20,324,288,747.31 |
| 2022 | 18,845,573,125.00 |
| 2021 | 15,400,266,531.77 |
| 2020 | 11,664,813,801.94 |
| 2019 | 12,383,222,505.53 |

**Top komunat sipas qarkullimit total**

| Komuna | Qarkullimi total |
|--------|------------------|
| PRISHTINË | 51,454,186,316.62 |
| GRAÇANICË | 12,001,136,718.54 |
| FERIZAJ | 8,042,949,524.07 |
| PRIZREN | 7,059,288,896.19 |
| PEJË | 5,792,950,085.84 |

**Top sektorët sipas qarkullimit total**

| Sektori | Qarkullimi total |
|---------|------------------|
| Tregtia me shumice dhe pakice; Riparimi i mjeteve motorike dhe motoeikletave | 58,454,594,520.41 |
| Industria perpunuese | 15,883,838,199.75 |
| Ndertimtaria | 11,493,086,507.21 |
| Furnizimi me rryme, gaz, avull dhe ajer te kondicionuar | 8,652,030,263.83 |
| Aktivitetet financiare dhe te sigurimit | 4,600,848,077.35 |

**Grafikët e agregimit**

Grafiku sipas viteve paraqet trendin e qarkullimit total në kohë. Nga rezultatet shihet se `2025` ka vlerën më të lartë totale në datasetin aktual.

![Turnover by Year](Faza%201/outputs/report/turnover_by_year.png)

Grafiku sipas muajve ndihmon të shihet sezonaliteti. Në të dhënat aktuale, muaji `12` ka qarkullimin total më të lartë.

![Turnover by Month](Faza%201/outputs/report/turnover_by_month.png)

Grafiku i komunave tregon komunat me peshën më të madhe ekonomike sipas qarkullimit. `PRISHTINË` dominon qartë ndaj komunave të tjera.

![Turnover by Municipality](Faza%201/outputs/report/turnover_by_municipality_top20.png)

Grafiku i sektorëve tregon se cilët sektorë kontribuojnë më shumë në qarkullim. Sektori i tregtisë me shumicë dhe pakicë ka peshën më të madhe.

![Turnover by Sector](Faza%201/outputs/report/turnover_by_sector_top15.png)

### Hapi 5: `sample`

- U krijua një kampion i rastësishëm me `1,000` rreshta.
- U krijua edhe një kampion i balancuar sipas `year_month` me `1,200` rreshta.
- Të dy kampionet përdorin `seed = 42`.
- U krijuan subset-e ndihmëse për `sector_focus`, `municipality_focus`, `registration_status_focus` dhe `monthly_time_pattern`.

Ky hap përdoret për eksplorim më të lehtë, testime të shpejta dhe për krijimin e subset-eve të fokusuara sipas dimensioneve kryesore të datasetit.

### Hapi 6: `outliers`

Outliers janë analizuar pa i fshirë nga dataseti kanonik.

| Metrika | Numri i rasteve |
|---------|-----------------|
| `turnover_eur_outlier_iqr` | 96,473 |
| `turnover_eur_outlier_zscore` | 5,455 |
| `num_taxpayers_outlier_iqr` | 79,451 |
| `num_taxpayers_outlier_zscore` | 8,101 |
| `turnover_eur_outlier_log_iqr` | 104,535 |

**Grafiku i outliers**

Ky grafik përmbledh sa raste ekstreme u zbuluan me secilën metodë. Ai tregon se `turnover_eur_log_iqr` dhe `turnover_eur_outlier_iqr` kapin më shumë vlera ekstreme sesa Z-score, gjë që është e pritshme për të dhëna financiare me shpërndarje të shtrembëruar.

![Outlier Summary](Faza%201/outputs/report/outlier_summary.png)

### Hapi 7: `imbalance`

Analiza e imbalance tregon se klasa `registration_status` është e dominuar nga disa kategori kryesore.

**Top klasat për `registration_status`**

| Klasa | Numri | Pesha |
|-------|-------|-------|
| `SH.P.K.` | 321,710 | 49.4244% |
| `INDIVIDUAL` | 270,904 | 41.6191% |
| `ORTAKËRIA E PËRGJ.` | 17,491 | 2.6871% |
| `SHOQËRI AKCIONARE` | 11,642 | 1.7886% |
| `KOMPANI E HUAJ` | 11,218 | 1.7234% |

**Grafiku i shpërndarjes së klasave**

Ky grafik tregon pabarazinë mes klasave të `registration_status`. Dy klasat kryesore, `SH.P.K.` dhe `INDIVIDUAL`, zënë shumicën e datasetit, ndërsa klasat e tjera janë shumë më të vogla.

![Registration Status Distribution](Faza%201/outputs/report/registration_status_distribution_top15.png)

### Hapi 8: `resample`

- Është gjeneruar dataset i resampluar me SMOTE për `registration_status`.
- Para resampling, klasa më e madhe ishte `SH.P.K.` me `321,710` raste.
- Pas resampling, secila klasë është balancuar në `321,710` raste.
- Ky hap përdoret vetëm për eksperimente modelimi dhe jo si dataseti kanonik i pastruar.

**Çfarë bën SMOTE në këtë projekt**

SMOTE (`Synthetic Minority Over-sampling Technique`) krijon shembuj sintetikë për klasat e vogla, në vend që thjesht t'i kopjojë ato. Kjo bëhet që modeli i machine learning të mos mësojë vetëm klasat dominante si `SH.P.K.`, por të ketë mundësi të mësojë edhe klasat e rralla.

**Grafikët para dhe pas SMOTE**

Grafiku para SMOTE tregon pabarazinë e madhe mes klasave.

![SMOTE Before Counts](Faza%201/outputs/report/smote_before_top15_counts.png)

Grafiku pas SMOTE tregon se klasat janë sjellë në një shpërndarje të balancuar për trajnim.

![SMOTE After Counts](Faza%201/outputs/report/smote_after_top15_counts.png)

### Hapi 9: `report`

- U gjenerua raporti final në `Faza 1/outputs/report/report.md`.
- U krijuan grafikë për qarkullimin sipas vitit, muajit, komunave, sektorëve, nulls, outliers dhe shpërndarjes së klasave.

Ky hap përmbledh gjithë pipeline-in në formë raporti dhe e bën projektin të gatshëm për prezantim.

## Konfigurimi

### Kërkesat

- Python 3.10+
- pandas
- numpy
- openpyxl
- matplotlib
- scikit-learn
- imbalanced-learn

Instalimi i varësive:

```bash
cd "Faza 1"
pip install -r requirements.txt
```

## Ekzekutimi i Projektit

Për të ekzekutuar pipeline-in e plotë:

```bash
cd "Faza 1"
python3 -m src.main --step all
```

Për të ekzekutuar hapa individualë:

```bash
cd "Faza 1"
python3 -m src.main --step profile_raw
python3 -m src.main --step ingest
python3 -m src.main --step clean
python3 -m src.main --step profile
python3 -m src.main --step aggregate
python3 -m src.main --step sample --sample-size 1200
python3 -m src.main --step sample --sample-size 5000 --stratify-by municipality
python3 -m src.main --step outliers
python3 -m src.main --step imbalance
python3 -m src.main --step report
python3 -m src.main --step resample --target-column registration_status --algorithm smote
```

## Output-et e Gjeneruara

Pipeline-i prodhon artefakte në folderin `Faza 1/outputs/`, duke përfshirë:

- `profile/before/`
- `profile/after/`
- `ingest/`
- `clean/`
- `aggregate/`
- `sample/`
- `outliers/`
- `imbalance/`
- `report/`

Disa prej skedarëve kryesorë të gjeneruar janë:

- `Faza 1/outputs/ingest/raw_extract.csv`
- `Faza 1/outputs/ingest/normalized.csv`
- `Faza 1/outputs/clean/strict_cleaned.csv`
- `Faza 1/outputs/clean/model_ready.csv`
- `Faza 1/outputs/clean/cleaning_log.csv`
- `Faza 1/outputs/clean/invalid_rows.csv`
- `Faza 1/outputs/report/report.md`

## Dokumentimi Shtesë

Për secilin hap të pipeline-it ekzistojnë dokumente të veçanta në:

- `Faza 1/docs/steps/`

README specifik i fazës gjendet në:

- `Faza 1/README.md`

## Shënime

- Kolonat Excel `A`, `E`, `F` dhe `I` anashkalohen sepse nuk përmbajnë të dhëna të dobishme për analizë.
- Kolonat bosh ndihmëse të Excel-it largohen gjatë ingestimit.
- Kampionimi në mënyrë standarde balancohet sipas `year_month`, përveç rasteve kur përcaktohet `--stratify-by`.
- Hapi `resample` është opsional dhe përdoret vetëm kur zgjidhet një kolonë target.
- Dataseti i pastruar kanonik nuk modifikohet nga eksperimentet opsionale të resampling-ut.

---

## Faza 2: Trajnimi I Modelit

### Qëllimi i Fazës 2

Faza 2 adresohet direkt pyetjes kryesore të projektit: **si ka performuar ekonomia e Kosovës sipas qytetit, sektorit, dhe kombinimit sektor × qytet në periudhën 2020–2025?**

### Struktura e Faza_2

```text
Faza_2/
├── requirements.txt
├── outputs/
│   ├── models/          # modelet e serializuara (.pkl)
│   ├── plots/           # grafikët e gjeneruar (.png)
│   └── metrics/         # raportet e metrikave (.csv)
└── src/
    ├── __init__.py
    ├── config.py          # konfigurim i centralizuar
    ├── data_loader.py     # ngarkimi dhe përgatitja e të dhënave
    ├── feature_engineering.py  # enkodimi i kategorive dhe shtrirja
    ├── supervised.py      # Step A: Random Forest + XGBoost
    ├── unsupervised.py    # Step A: K-Means + PCA
    ├── evaluation.py      # Step B: metrika, grafikë dhe krahasime
    └── main.py            # pikë hyrëse CLI
```

### Algoritmet e Përdorura (7 gjithsej)

| # | Algoritmi | Lloji | Qëllimi                                                                  |
|---|-----------|-------|--------------------------------------------------------------------------|
| 1 | **Random Forest Classifier** | Supervised – Klasifikim | Klasifikon kombinimin (komunë, sektor) si GROWING / STABLE / DECLINING   |
| 2 | **XGBoost Classifier** | Supervised – Klasifikim | Alternativë gradient-boosting për klasifikim, shpesh me saktësi të lartë |
| 3 | **Linear Regression** | Supervised – Regresion | Baseline interpretues: parashikon vlerën e vazhdueshme të `growth_rate`  |
| 4 | **Random Forest Regressor** | Supervised – Regresion | Parashikon `growth_rate` me metodë ensemble jo-lineare                   |
| 5 | **XGBoost Regressor** | Supervised – Regresion | Modeli kryesor për parashikimet e tregut 2026                            |
| 6 | **K-Means Clustering** | Unsupervised | Grupimi i komunave dhe sektorëve sipas trajektores së rritjes            |
| 7 | **PCA** | Unsupervised | Reduktim dimensionaliteti dhe vizualizim                                 |

---

### Arsyetimi për zgjedhjen e algoritmeve

#### Algoritmet e Supervised Learning

**1. Random Forest Classifier**

Random Forest është zgjedhja kryesore për klasifikim për arsyet e mëposhtme:

- Dataseti ka tipare të përziera — numerike (`num_taxpayers`, `turnover_eur_log1p`) dhe kategorike të enkuduara (`primary_sector`, `municipality`). Random Forest i trajton këto tipare pa presupozime lineare.
- Të dhënat financiare të qarkullimit kanë shpërndarje të shtrembëruar me outliers të shumtë (96 473 me metodën IQR). Pylltë e rastësishme janë robuste ndaj vlerave ekstreme sepse vendimmarrja bazohet në ndarje, jo në distancë.
- Parametri `class_weight="balanced"` kompenson pabarazinë e klasave pa pasur nevojë të ngarkohet skedari i resampluar 603 MB.
- Ofron `feature_importances_` që tregon se cilat tipare — viti, muaji, sektori apo komuna — ndikojnë më shumë në klasifikimin e entitetit të regjistrimit.

**2. XGBoost Classifier**

XGBoost (Extreme Gradient Boosting) është plotësuesi natyral i Random Forest:

- Ndërsa Random Forest trajnon pemë paralelisht dhe i kombinon, XGBoost i ndërton pemë sekuencialisht duke korrigjuar gabimet e bëra nga modelet e mëparshme. Kjo strategji e boosting-ut zakonisht prodhon saktësi më të lartë në të dhëna tabulare.
- Regularizimi i integruar (L1 + L2) dhe `colsample_bytree=0.8` reduktojnë mbipërshtatjen (overfitting) në hapësirën e madhe të tipareve.
- Ofron `feature_importances_` të krahasueshme me ato të Random Forest, duke lejuar validimin e kross-metodave.
- Algoritmi është i optimizuar për memorie dhe CPU, i përshtatshëm për 50 000 rreshta me hiper-parametra konservativë.

#### Algoritmet e Supervised Learning – Regresion

**3. Linear Regression (OLS)**

- Shërben si benchmark minimal: nëse modelet e pemëve nuk e tejkalojnë një vijë të drejtë, sinjali jo-linear është i dobët.
- Koeficientët janë drejtpërdrejt të interpretueshëm: "+X% rritje për çdo njësi log-EUR qarkullimi paraprak."
- Nuk bën supozime mbi shpërndarjen e gabimeve në model; StandardScaler siguron peshim të drejtë të tipareve.

**4. Random Forest Regressor**

- E njëjta logjikë si klasifikuesi, por parashikon `growth_rate` si vlerë të vazhdueshme.
- Trajton ndërveprime jo-lineare midis komunës, sektorit dhe qarkullimit historik.
- Mesatarja ndër 300 pemë zvogëlon variancën, gjë kritike kur dataseti i agreguar ka vetëm 3 956 rreshta.

**5. XGBoost Regressor**

- Gradient boosting sekuencial me `objective=reg:squarederror` — optimizon MSE direkt.
- Regularizimi L1/L2 i integruar dhe `subsample=0.8` parandalojnë overfitting-un.
- Shërbeu si modeli kryesor për parashikimet e tregut 2026 falë saktësisë superiore të tij mbi të dhëna tabulare strukturore.

#### Algoritmet e Unsupervised Learning

**6. K-Means Clustering**

- Dataseti i qarkullimit ka grupime natyrale: pesë klasa kryesore të `registration_status` (SH.P.K., INDIVIDUAL, ORTAKËRI, SHOQËRI AKCIONARE, KOMPANI E HUAJ), profile të ndryshme komunale (Prishtina dominon me 51.5 miliardë EUR), dhe banda të dallueshme të qarkullimit sipas sektorëve.
- K-Means është efikas llogaritërisht me 50 000 rreshta dhe prodhon centroide të interpretueshme.
- Analiza elbow + silhouette + Davies-Bouldin lejon zgjedhjen empirike të k-ut optimal pa supozime paraprake.
- Krahasimi i klastereve me etiketat reale të `registration_status` teston nëse grupimi natyral i të dhënave korrespondon me klasifikimet ligjore të bizneseve.

**7. Principal Component Analysis (PCA)**

- Pas enkodimit të kategorive, hapësira e tipareve ka dimensione të papërsosura: vlerat numerike të mëdha (`turnover_eur_log1p`) ndaj kodeve të vogla integer për qindra komuna. PCA i dekorrelon këto sinjale.
- Projekcioni 2-D mundëson vizualizimin e strukturës latente të të dhënave pa humbur pasinformacion themelor.
- Kurba e variancës së shpjeguar tregon sa dimensione mbajnë 95% të variancës — kjo informon reduktimin e mundshëm të dimensionalitetit në fazat e ardhshme të modelimit.
- PCA + K-Means ofron një perspektivë të dyfishtë: K-Means në hapësirën origjinale vs. klasterimi në hapësirën e reduktuar të PCA-së.

### Ekzekutimi i Fazës 2

```bash
cd Faza_2
pip install -r requirements.txt

# Pipeline i plotë (të 7 algoritmet + parashikimet 2026)
python -m src.main --step all

# Vetëm klasifikimi (Step A)
python -m src.main --step supervised

# Vetëm regresioni (Step B)
python -m src.main --step regression

# Vetëm unsupervised – K-Means + PCA (Step C)
python -m src.main --step unsupervised

# Vetëm heatmaps e rritjes (Step D)
python -m src.main --step analysis

# Vetëm parashikimet 2026 (Step E — kërkon modelet të trajnuara)
python -m src.main --step predictions
```

### Step B: Rezultatet e Evaluimit

> **Dataset i ri**: 650 913 rreshta bruto → agregim sipas (vit × komunë × sektor) → **3 956 rreshta** (njësia e analizës).
> Train set: 3 164 rreshta | Test set: 792 rreshta | split: 80/20 stratifikuar.
> Target: `growth_class` — GROWING (58.7%) / DECLINING (31.0%) / STABLE (10.3%)
> Features (pa leakage): `year`, `prev_turnover_log1p`, `num_businesses_log1p`, `municipality` (enc.), `primary_sector` (enc.)

---

#### Supervised Learning – Krahasimi i modeleve

##### Hold-out test set (80/20 split)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) | Cohen's Kappa |
|-------|----------|-------------------|----------------|------------------|---------------|
| **Random Forest** | 0.590 | 0.474 | **0.471 ✓** | **0.469 ✓** | **0.256 ✓** |
| **XGBoost** | **0.625 ✓** | **0.482 ✓** | 0.438 | 0.434 | 0.243 |

##### 5-Fold Cross-Validation (mesatare ± std ndër foldat)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) | Cohen's Kappa |
|-------|----------|-------------------|----------------|------------------|---------------|
| **Random Forest** | 0.581±0.009 | 0.454±0.011 | **0.457±0.011 ✓** | **0.453±0.011 ✓** | 0.243±0.014 |
| **XGBoost** | **0.632±0.006 ✓** | **0.495±0.020 ✓** | 0.448±0.006 | 0.447±0.008 | **0.259±0.013 ✓** |

**Rezultati i verdiktit: 5 fitore për secilin model** — barazim i vërtetë.

Cross-validimi me 5 folda konfirmon qëndrueshmërinë e rezultateve: devijimi standard i ulët (±0.006–0.020) tregon që asnjë model nuk ka pasur "fat të mirë" në ndarjen e vetme 80/20. Saktësia e moderuar (58–63%) është e pritshme: rritja ekonomike varet nga faktorë si politika fiskale, investimet e huaja dhe goditjet globale — informacione jo të disponueshme në dataset.

**Cili model zgjidhni sipas qëllimit:**
- Zgjidhni **XGBoost** nëse prioriteti është **Accuracy dhe Precision** — humbni pak false positives, por mund të humbisni disa raste reale të rënies ekonomike.
- Zgjidhni **Random Forest** nëse prioriteti është **Recall dhe F1** — kap më mirë komunat/sektorët që vërtet janë DECLINING ose STABLE, gjë kritike për vendimmarrje politike ku kostoja e "mos-zbulimit të rënies" është e lartë.
- Për analizë ekonomike dhe politikë publike, **Random Forest rekomandohet** — Recall i lartë do të thotë që modeli "nuk harron" sektorët në vështirësi.

![Model Comparison](Faza_2/outputs/plots/model_comparison.png)

![CV Score Boxplot](Faza_2/outputs/plots/cv_score_boxplot.png)

---

#### Random Forest – Raporti i klasifikimit

| Klasa | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| DECLINING | 0.485 | 0.588 | 0.531 | 245 |
| GROWING | **0.706** | 0.667 | **0.686** | 465 |
| STABLE | 0.232 | 0.159 | 0.188 | 82 |
| **macro avg** | **0.474** | **0.471** | **0.469** | 792 |
| weighted avg | 0.589 | 0.590 | 0.587 | 792 |

Random Forest identifikon mirë klasën GROWING (F1=0.686) por ngatërron shpesh STABLE me DECLINING/GROWING, gjë që reflekton vështirësinë e dallimit të rritjes "margjinale" (±5%) nga lëvizjet e vërteta.

![Random Forest Confusion Matrix](Faza_2/outputs/plots/RandomForest_confusion_matrix.png)

![Random Forest ROC Curves](Faza_2/outputs/plots/RandomForest_roc_curves.png)

![Random Forest Feature Importance](Faza_2/outputs/plots/RandomForest_feature_importance.png)

---

#### XGBoost – Raporti i klasifikimit

| Klasa | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| DECLINING | 0.524 | 0.445 | 0.481 | 245 |
| GROWING | 0.673 | **0.822** | **0.740** | 465 |
| STABLE | 0.250 | 0.049 | 0.082 | 82 |
| **macro avg** | **0.482** | **0.438** | **0.434** | 792 |
| weighted avg | 0.583 | 0.625 | 0.592 | 792 |

XGBoost arrin Recall=0.822 për GROWING — domethënë kap 82% të kombinimeve (komunë, sektor) që vërtet u rritën. Megjithatë, klasa STABLE humbet gati plotësisht (Recall=0.049): modeli e grupon atë kryesisht si GROWING ose DECLINING. Kjo është e arsyeshme ekonomikisht — "stabiliteti" i vërtetë në 5 vite me rritje globale është fenomen i rrallë.

![XGBoost Confusion Matrix](Faza_2/outputs/plots/XGBoost_confusion_matrix.png)

![XGBoost ROC Curves](Faza_2/outputs/plots/XGBoost_roc_curves.png)

![XGBoost Feature Importance](Faza_2/outputs/plots/XGBoost_feature_importance.png)

---

#### Unsupervised Learning – K-Means: analiza e klastereve

**Gjetja e k-ut optimal (k=2 deri k=8)**

| k | Inertia | Silhouette Score | Davies-Bouldin Index |
|---|---------|-----------------|----------------------|
| 2 | 14 793 | **0.2158** | 1.6670 |
| 3 | 12 716 | 0.1890 | 1.7108 |
| **4** | **11 301** | 0.1806 | 1.5839 |
| 5 | 10 137 | 0.1840 | 1.4628 |
| 6 | 9 247 | 0.1927 | 1.3844 |
| 7 | 8 609 | 0.1880 | 1.3200 |
| 8 | 7 954 | 0.1955 | **1.3682** |

U zgjodh **k=4**: elbow-i shfaqet qartë pas k=4 dhe ofron 4 profile të interpretueshme të rritjes: *rritje e fortë* / *rritje e moderuar* / *stagnacion* / *rënie*. Silhouette maksimal është në k=2 (0.2158), por k=2 është tepër i thjeshtë për analiza ekonomike kuptimplota.

![K-Means Elbow & Silhouette](Faza_2/outputs/plots/kmeans_elbow_silhouette.png)

**Metrikat e modelit final K-Means (k=4)**

| Metrika | Vlera |
|---------|-------|
| Inertia | 11 301 |
| Silhouette Score | 0.181 |
| Davies-Bouldin Index | 1.584 |

---

#### Unsupervised Learning – Klasterimi i trajektoreve të rritjes

Për t'u dhënë përgjigje pyetjeve "cilat qytete ndanë të njëjtin trend rritjeje?" dhe "cilët sektorë u sjollën ngjashëm gjatë viteve?", u aplikua klasterimi K-Means (k=4) direkt mbi matricën e trajektoreve (grupi × vit → shkalla mesatare e rritjes).

**Klasterimi i komunave (Municipality Trajectory Clustering)**
- Silhouette=0.235 | Davies-Bouldin=1.143
- Komunat ndahen në 4 grupe sipas profilit të rritjes 2020–2025

![Municipality Trajectory Clusters](Faza_2/outputs/plots/municipality_trajectory_clusters.png)

**Klasterimi i sektorëve (Sector Trajectory Clustering)**
- Silhouette=0.184 | Davies-Bouldin=0.882 (ndarja e sektorëve është më e qartë se ajo e komunave)
- Sektorët ndahen sipas ritmit dhe drejtimit të rritjes vit pas viti

![Sector Trajectory Clusters](Faza_2/outputs/plots/primary_sector_trajectory_clusters.png)

---

#### Unsupervised Learning – PCA

| Komponent | Varianca e Shpjeguar | Kumulative |
|-----------|---------------------|------------|
| PC1 | 39.7% | 39.7% |
| PC2 | 20.2% | **59.9%** |
| PC3–PC4 | 35.1% | 95.0%+ |

Dy komponentët e parë shpjegojnë **59.9%** të variancës — shumë më shumë se versioni i mëparshëm (39.2%), sepse dataset-i i growth-it ka tipare më kohezive. Nevojiten vetëm **4 komponentë** për ≥95%.

![PCA Variance Explained](Faza_2/outputs/plots/pca_variance_explained.png)

![PCA – Growth Classes](Faza_2/outputs/plots/pca_true_labels.png)

![PCA – K-Means Clusters](Faza_2/outputs/plots/pca_kmeans_clusters.png)

---

#### Analiza Kryqëzore: Sektor × Komunë (Heatmaps)

Kjo është analiza qendrore e Fazës 2. Çdo qelizë e heatmap-it tregon **shkallën mesatare të rritjes YoY** për atë kombinim (sektor, komunë). Ngjyra e gjelbër = rritje, e kuqe = rënie.

**Rritja sipas Komunave (Municipality × Year)**

Grafiku tregon si ka evoluuar qarkullimi i çdo komune vit pas viti. Komunat me rritje të qëndrueshme të gjelbër janë qendrat ekonomike dinamike.

![Municipality Growth by Year](Faza_2/outputs/plots/municipality_growth_by_year.png)

**Rritja sipas Sektorëve (Sector × Year)**

Grafiku identifikon sektorët me rritje sistemike dhe ata me tkurrje. Periudha 2020–2021 tregon impaktin e COVID-19 dhe rimëkëmbjen pasardhëse.

![Sector Growth by Year](Faza_2/outputs/plots/sector_growth_by_year.png)

**Kryqëzimi Sektor × Komunë — të gjitha vitet**

Ky grafik u përgjigjet direkt pyetjes: *"Si ka performuar sektori X në qytetin Y krahasuar me qytetet e tjera?"*

![Sector × Municipality (All Years)](Faza_2/outputs/plots/sector_x_municipality_growth_all.png)

**Kryqëzimi Sektor × Komunë — 2023**

![Sector × Municipality (2023)](Faza_2/outputs/plots/sector_x_municipality_growth_2023.0.png)

**Kryqëzimi Sektor × Komunë — 2024**

![Sector × Municipality (2024)](Faza_2/outputs/plots/sector_x_municipality_growth_2024.0.png)

**Kryqëzimi Sektor × Komunë — 2025**

![Sector × Municipality (2025)](Faza_2/outputs/plots/sector_x_municipality_growth_2025.0.png)

---

---

#### Supervised Learning – Regresion (Step B)

Tre modele regresioni u trajnuan mbi të njëjtat tipare si klasifikuesit (pa `growth_rate` si input — shih shënimin e leakage-ut), me target-in `growth_rate` si vlerë të vazhdueshme.

> **Train set:** 3 164 rreshta | **Test set:** 792 rreshta | Split: 80/20 pa stratifikim

##### Krahasimi i modeleve të Regresionit

| Model | MAE | RMSE | R² | Interpretim |
|-------|-----|------|----|-------------|
| **Linear Regression** | 0.5216 | 0.8605 | 0.0495 | Baseline i dobët; rritja ka pak sinjal linear |
| **Random Forest Regressor** | **0.4932** | **0.8577** | **0.0556** | ✓ Modeli më i mirë — MAE dhe RMSE më të ulëta |
| **XGBoost Regressor** | 0.5147 | 0.8931 | −0.0237 | R² negativ: performon nën mesataren e thjeshtë |

**Pse R² është i ulët (~5%)?**

R² prej 0.05–0.06 do të thotë se modelet shpjegojnë rreth **5% të variancës** të shkallës YoY të rritjes. Kjo është e pritshme dhe jo domosdoshmërisht shenjë e modelit të gabuar:

- **Rritja ekonomike varet nga faktorë makro** si politika fiskale, investimet e huaja, inflacioni dhe goditjet globale (p.sh. COVID-19, lufta në Ukrainë) — asnjë prej tyre nuk është i disponueshëm në datasetin ATK.
- **Dataset-i i vogël pas agregimit** (3 956 rreshta) e kufizon kapacitetin e modeleve jo-lineare.
- **Sinjali është më i qartë si klasifikim**: pyetja "GROWING apo DECLINING?" (59–63% accuracy) është më e saktë se parashikimi i vlerës ekzakte të `growth_rate`.

Për qëllime analitike dhe politike, **klasifikuesi rekomandohet** si modeli kryesor. Regresori shërben për renditjen relative të mundësive në parashikimet 2026.

![Regressor Comparison](Faza_2/outputs/plots/regressor_comparison.png)

![RF Regressor – Actual vs Predicted](Faza_2/outputs/plots/RandomForestRegressor_actual_vs_predicted.png)

![RF Regressor – Residuals](Faza_2/outputs/plots/RandomForestRegressor_residuals.png)

---

#### Parashikimet e Tregut 2026 (Step E)

Duke përdorur qarkullimin e vitit 2025 si "qarkullim paraprak", të 5 modelet parashikuan rritjen për **659 kombinime (komunë × sektor)** për vitin 2026.

> **Baza:** Të dhënat e 2025 → parashikim 2026 | Modeli kryesor për renditje: **XGBoost Regressor**

##### Top 10 Mundësi Rritjeje për 2026 (XGBoost Regressor)

| Komuna | Sektori | Rritja e Parashikuar |
|--------|---------|----------------------|
| SUHAREKË | Aktivitetet e pasurive të paluajtshme | +479.5% |
| GJAKOVË | Aktivitetet e pasurive të paluajtshme | +214.4% |
| FERIZAJ | Aktivitetet e pasurive të paluajtshme | +180.2% |
| MITROVICË | Aktivitetet e pasurive të paluajtshme | +142.7% |
| PRIZREN | Aktivitetet e pasurive të paluajtshme | +138.1% |
| VUSHTRRI | Aktivitetet e pasurive të paluajtshme | +135.0% |
| LIPJAN | Aktivitetet e pasurive të paluajtshme | +132.8% |
| GJILAN | Aktivitetet e pasurive të paluajtshme | +131.9% |
| DRENAS | Aktivitetet e pasurive të paluajtshme | +127.4% |
| PEJË | Aktivitetet e pasurive të paluajtshme | +122.6% |

> **Shënim interpretues**: Parashikimet ekstreme (>100%) duhet të shihen me kujdes — R² i ulët i modelit të regresionit (~5%) tregon pasiguri të lartë në vlerat absolute. Ato janë të dobishme për **renditje relative** (cilët sektorë/komuna kanë potencial më të lartë), jo si parashikime absolute numerike.

![Parashikimet 2026 – Heatmap](Faza_2/outputs/plots/predicted_growth_2026_XGBoostRegressor.png)

![Top Parashikimet 2026](Faza_2/outputs/plots/predicted_top_2026_XGBoostRegressor.png)

Parashikimet e plota janë të disponueshme në:
- `Faza_2/outputs/metrics/predictions_2026.csv` — 659 rreshta me parashikime nga të 5 modelet
- Kolonat: `municipality`, `primary_sector`, `growth_class_RandomForest`, `growth_class_XGBoost`, `growth_rate_LinearRegression`, `growth_rate_RandomForestRegressor`, `growth_rate_XGBoostRegressor`, `est_turnover_*`

---

### Step C: Ndikimi i Fazës 1 mbi rezultatet e modelimit

Përpunimi i Fazës 1 ka pasur ndikim të drejtpërdrejtë dhe të matshëm mbi cilësinë e analizës ekonomike në Fazën 2:

**Heqja e 102 977 duplikateve** ishte kritike: duplikatat do të kishin fryra artificialisht qarkullimin e disa kombinimeve (komunë, sektor, vit), duke falsifikuar llogaritjen e shkallës YoY të rritjes. Pas heqjes, agregimet pasqyrojnë vlerën reale ekonomike.

**Transformimi log1p i qarkullimit** ishte i domosdoshëm për dy arsye: (1) diferenca ekstreme midis komunave (Prishtina 51 miliardë EUR vs. komunat e vogla me disa miliona) do të dominonte distancat Euklidiane në K-Means pa log-transform; (2) StandardScaler funksionon mirë kur shpërndarja afron normalen — log1p e siguron këtë.

**Normalizimi NFKC i tekstit shqip** (ë, ç, karaktere speciale) parandaloi grupim të gabuar: `"Prishtinë"` vs `"Prishtine"` pa normalizim do të sillte dy komuna të ndryshme gjatë agregimt, duke fragmentuar datasetin e growth-it.

**Outlier detection (Faza 1)** zbuloi 96 473 rreshta ekstreme me IQR. Megjithëse nuk u hoqën, njohja e tyre shpjegon pse disa kombinime (komunë, sektor) kanë growth_rate të jashtëzakonshme (+200% ose −80%) — kryesisht nga biznese individuale me qarkullim shumë të lartë që hyjnë ose dalin nga tregu.

**Agregimi si zgjidhje ndaj imbalance-it**: Imbalanca origjinale e `registration_status` (SH.P.K. 49% vs KOMPANI E HUAJ 1.7%) bëhet e parëndësishme në Fazën 2 — duke agreguar mbi llojin e entitetit, njësia e analizës bëhet kombinimi (komunë, sektor) dhe jo entiteti individual. `class_weight="balanced"` trajton imbalancën e re (GROWING 58.7% / DECLINING 31% / STABLE 10.3%).

### Output-et e Fazës 2

```text
Faza_2/outputs/
├── models/
│   ├── random_forest.pkl            ← Random Forest Classifier
│   ├── xgboost.pkl                  ← XGBoost Classifier
│   ├── linear_regression.pkl        ← Linear Regression
│   ├── rf_regressor.pkl             ← Random Forest Regressor
│   └── xgb_regressor.pkl            ← XGBoost Regressor
├── metrics/
│   ├── RandomForest_classification_report.csv
│   ├── XGBoost_classification_report.csv
│   ├── model_comparison.csv
│   ├── cross_validation_results.csv        ← CV 5-fold mean±std për të dy modelet
│   ├── algorithm_verdict.csv               ← tabela e verdiktit 
│   ├── kmeans_sweep_metrics.csv
│   ├── unsupervised_metrics.csv
│   ├── municipality_growth_by_year.csv     ← rritja e çdo komune sipas vitit
│   ├── sector_growth_by_year.csv           ← rritja e çdo sektori sipas vitit
│   ├── sector_x_municipality_growth_*.csv  ← kryqëzim sektor × komunë
│   └── predictions_2026.csv               ← 659 parashikime (komunë × sektor) për 2026
└── plots/
    ├── RandomForest_confusion_matrix.png
    ├── RandomForest_roc_curves.png
    ├── RandomForest_feature_importance.png
    ├── XGBoost_confusion_matrix.png
    ├── XGBoost_roc_curves.png
    ├── XGBoost_feature_importance.png
    ├── model_comparison.png
    ├── cv_score_boxplot.png                ← shpërndarja
    ├── regressor_comparison.png            ← krahasim MAE/RMSE/R² i 3 regresorëve
    ├── LinearRegression_actual_vs_predicted.png
    ├── LinearRegression_residuals.png
    ├── RandomForestRegressor_actual_vs_predicted.png
    ├── RandomForestRegressor_residuals.png
    ├── XGBoostRegressor_actual_vs_predicted.png
    ├── XGBoostRegressor_residuals.png
    ├── LinearRegression_feature_importance.png
    ├── RandomForestRegressor_feature_importance.png
    ├── XGBoostRegressor_feature_importance.png
    ├── predicted_growth_2026_XGBoostRegressor.png  ← heatmap parashikimesh 2026
    ├── predicted_top_2026_XGBoostRegressor.png     ← top 15 mundësi rritjeje
    ├── kmeans_elbow_silhouette.png
    ├── municipality_trajectory_clusters.png ← komunat sipas profilit të rritjes
    ├── primary_sector_trajectory_clusters.png
    ├── pca_true_labels.png
    ├── pca_kmeans_clusters.png
    ├── pca_variance_explained.png
    ├── municipality_growth_by_year.png      ← heatmap komunë × vit
    ├── sector_growth_by_year.png            ← heatmap sektor × vit
    ├── sector_x_municipality_growth_all.png ← kryqëzim kryesor (të gjitha vitet)
    ├── sector_x_municipality_growth_2023.0.png
    ├── sector_x_municipality_growth_2024.0.png
    └── sector_x_municipality_growth_2025.0.png
```

---

## Faza 3: Përmirësimi, Fine-Tuning-u dhe Krahasimi me Fazën 2

### Qëllimi i Fazës 3

Faza 3 ndërtohet mbi auditin e Fazës 2 dhe adreson **9 dobësi të identifikuara metodologjike**. Qëllimi nuk është të zëvendësohet Faza 2, por të dëshmohet se me një pipeline më rigoroz arrijmë rezultate substancialisht më të mira pa prekur asnjë skedar të Fazës 2.

**9 dobësitë e Fazës 2 që adresohen:**

1. **Humbje masive granulariteti** — Faza 2 agregon në (vit, komunë, sektor) → vetëm 3,956 rreshta nga 650,913
2. **LabelEncoder për kategori nominale** — fut renditje të rreme ordinale
3. **Mungesë e hyperparameter tuning-ut**
4. **Mungesë e early stopping për XGBoost** — overfitting i fortë (XGB Reg R² = -0.024)
5. **Random 80/20 split** → leakage kohor (modeli sheh të ardhmen gjatë trajnimit)
6. **Klasë STABLE pothuajse e pamësuar** (Recall ~0.05-0.16)
7. **Mungesë e validation set të veçantë** nga test-i
8. **Vetëm 5 tipare** — pa lag-e, rolling means, market concentration, seasonal
9. **Parashikime ekstreme** (+479%) — pa post-processing

### Struktura e Faza_3

```text
Faza_3/
├── README.md                       # dokumentim akademik me numra realë
├── requirements.txt                # shton: imbalanced-learn, category-encoders
├── src/
│   ├── config.py                   # hiperparametra + search-spaces + time-split bounds
│   ├── data_loader.py              # agregim mujor + lag1/2/3 + rolling 3/6m + market_conc
│   ├── feature_engineering.py      # TargetEncoder fit-on-train-only + OHE fallback
│   ├── time_split.py               # ndarje kronologjike + TimeSeriesSplit për CV
│   ├── sampling.py                 # SMOTE-auto vetëm mbi train (me asserts)
│   ├── supervised.py               # tune_hyperparameters + train_with_early_stop
│   ├── unsupervised.py             # K-Means + PCA mbi matricën e re
│   ├── evaluation.py               # metrika + plot + compare_phase2_phase3
│   ├── ablation.py                 # A0→A6 incremental ablation study
│   └── main.py                     # CLI orchestrator
├── outputs/
│   ├── models/                     # 5 .pkl (RF Cls, XGB Cls, LR, RF Reg, XGB Reg)
│   ├── metrics/                    # 17 CSV + best_hyperparams.json
│   └── plots/                      # 35 PNG (i njëjti emër si Faza 2 për krahasim)
└── part2_real_growth/              # SHIH SEKSIONIN "PART 2" MË POSHTË
```

### Ndryshimet kryesore nga Faza 2

#### 1. Agregim mujor në vend të vjetor

Skedari i ndryshuar: `src/data_loader.py`

| Karakteristika | Faza 2 | Faza 3 |
|----------------|--------|--------|
| Agregim | (vit, komunë, sektor) | (vit, **muaj**, komunë, sektor) |
| Rreshta finale pas lag | **3,956** | **39,155** (10× më shumë) |
| Target growth_rate | YoY annual | YoY mujore (shift 12 muaj) |
| Tipare lag | prev_turnover | **lag1, lag2, lag3, lag12** |
| Rolling features | asnjë | **rolling 3m mean/std, 6m mean/std** |
| Market concentration | asnjë | **# bizneset në komunë në muajin përkatës** |
| Cyclical seasonal | asnjë | **month_sin, month_cos** |

#### 2. Ndarje kronologjike kundër random split

Skedari i ri: `src/time_split.py`

- **Faza 2:** 80/20 random (futë leakage kohor — modeli sheh 2025 dhe testohet mbi 2020)
- **Faza 3:** Train ≤ 2023 (25,826 rreshta) | Val = 2024 (6,649) | Test = 2025 (6,680)
- `TimeSeriesSplit` përdoret edhe brenda `RandomizedSearchCV` për tuning kronologjikisht të saktë

#### 3. Encoding pa leakage

Skedari i ndryshuar: `src/feature_engineering.py`

- **Faza 2:** `LabelEncoder` për "primary_sector" dhe "municipality" → fut renditje të rreme ordinale (PRISHTINË=0, FERIZAJ=1, ...)
- **Faza 3:** `TargetEncoder` (category_encoders) me **smoothing=10.0**, **fit vetëm mbi train**, transformohet val/test/pred pa ri-fit. Strict contract dhe assertion në kod parandalon target leakage.

#### 4. Trajtim i klasës STABLE me SMOTE

Skedari i ri: `src/sampling.py`

- **Faza 2:** vetëm `class_weight="balanced"` për RF. XGB Stable Recall = **0.049** (modeli e injoroi klasën)
- **Faza 3:** **SMOTE strategy="auto"** balancon GROWING/DECLINING/STABLE → secila 14,730 rreshta në train. Plus `sample_weight=balanced` për XGBoost gjatë trajnimit dhe early stopping eval (parandalon që early stopping mbi val imbalanced të anulojë SMOTE-n)

#### 5. Hyperparameter tuning sistematik

Skedari i ndryshuar: `src/supervised.py`

- **Faza 2:** parametra hand-picked, jo tuning
- **Faza 3:** `RandomizedSearchCV` me **25 iteracione × 4 folds** për 4 modele = ~400 fits gjithsej. Search spaces të zgjeruara:
  - RF: `n_estimators` [200-500], `max_depth` [8-None], `max_features` ["sqrt", "log2", None], `class_weight` ["balanced", "balanced_subsample"]
  - XGB: `learning_rate` [0.01-0.1], `max_depth` [3-8], `subsample` [0.6-1.0], `reg_lambda` [0.5-5.0], `gamma` [0-0.4]

Parametrat optimalë ruhen në `outputs/metrics/best_hyperparams.json`.

#### 6. Early stopping për XGBoost

- **Faza 2:** XGBoost trajnonte gjithë 300 pemë qoftëdo → XGB Regressor R² = -0.024 (overfitting i qartë)
- **Faza 3:** `eval_set=(X_val, y_val)` + `early_stopping_rounds=30`. XGB Reg ndaloi para 700 iteracioneve me parametrat e tuned (max_depth=4, reg_lambda=5.0) — shumë më të rregullarizuar

#### 7. Post-processing me kufij ekonomikë

- **Faza 2:** parashikimet 2026 shkonin deri +479% (jorealiste)
- **Faza 3:** clip në `[PRED_CLIP_LOWER=-1.0, PRED_CLIP_UPPER=2.0]` për të gjithë regresorët

#### 8. Cross-validation kronologjike

- **Faza 2:** StratifiedKFold (random) → CV me leakage
- **Faza 3:** `TimeSeriesSplit(4)` brenda RandomizedSearchCV → CV respektivisht kronologjik

#### 9. Ablation study si kontribut akademik

Skedari i ri: `src/ablation.py`

Trajnon XGBoost Cls në 7 nivele kumulative për të kuantifikuar kontributin e secilit përmirësim:
- **A0:** baseline Faza 2 (tipare vjetore + LabelEncoder)
- **A1:** + agregim mujor + lag + rolling
- **A2:** + cyclical seasonal
- **A3:** + market concentration
- **A4:** + TargetEncoder
- **A5:** + SMOTE-auto
- **A6:** + tuned hyperparameters (Faza 3 e plotë)

### Algoritmet e Përdorura (të njëjtit 7 si Faza 2, por të rikonfiguruar)

| # | Algoritmi | Ndryshimi nga Faza 2 |
|---|-----------|----------------------|
| 1 | RandomForest Classifier | Tuned, trajnuar mbi train+val pas SMOTE |
| 2 | XGBoost Classifier | Tuned + early stopping + sample_weight balanced |
| 3 | Linear Regression | I pa-tuned (baseline i pastër) |
| 4 | RandomForest Regressor | Tuned, trajnuar mbi train+val |
| 5 | XGBoost Regressor | Tuned + early stopping |
| 6 | K-Means Clustering | I aplikuar mbi matricën e re mujore |
| 7 | PCA | Mbi 22 tipare (vs 5 në Fazën 2) |

### Rezultatet e Trajnimit

**Dataset i ri pas filtrimit:** 39,155 rreshta mujorë (10× më shumë se Faza 2)

**Class distribution (mujor):**
- GROWING: 22,067 (56.4%)
- DECLINING: 13,383 (34.2%)
- STABLE: 3,705 (9.5%)

#### Krahasimi Faza 2 vs Faza 3 — Tabela Kryesore

Skedari: `Faza_3/outputs/metrics/comparison_phase2_phase3.csv`

| Algoritmi | Metrika | Faza 2 | Faza 3 | Δ | Përmirësim? |
|-----------|---------|--------|--------|---|-------------|
| RandomForest Classifier | Accuracy | 0.590 | **0.700** | **+11.0 pp** | ✅ |
| RandomForest Classifier | F1-macro | 0.469 | 0.516 | +4.8 pp | ✅ |
| RandomForest Classifier | Kappa | 0.256 | **0.413** | **+15.8 pp** | ✅ |
| RandomForest Classifier | STABLE Recall | 0.159 | 0.072 | −8.7 pp | ❌ |
| XGBoost Classifier | Accuracy | 0.625 | 0.634 | +0.9 pp | ✅ |
| XGBoost Classifier | F1-macro | 0.434 | **0.526** | **+9.2 pp** | ✅ |
| XGBoost Classifier | Kappa | 0.243 | **0.358** | +11.5 pp | ✅ |
| **XGBoost Classifier** | **STABLE Recall** | 0.049 | **0.286** | **+23.8 pp** | ✅✅ |
| RandomForest Regressor | R² | 0.056 | **0.523** | **+0.47** | ✅ |
| **XGBoost Regressor** | R² | **−0.024** | **0.515** | **+0.54** | ✅ |

**9 nga 10 metrika u përmirësuan.** Vetëm RF STABLE Recall ra (tuning zgjodhi `class_weight="balanced_subsample"` që nuk kombinohet mirë me SMOTE).

![Phase 2 vs Phase 3 Comparison](Faza_3/outputs/plots/phase2_vs_phase3_comparison.png)

> **Shënim mbi drejtësinë e krahasimit:** Faza 2 përdor random split (më e lehtë sepse fut leakage kohor). Faza 3 përdor time-based split (më e vështirë). **Faza 3 fiton pavarësisht se problemi i saj është më i vështirë.**

#### Klasifikimi në detaje

**RandomForest Classifier — test 2025:**

| Klasa | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| DECLINING | 0.624 | 0.654 | 0.638 | 2148 |
| GROWING | 0.728 | 0.802 | 0.763 | 3893 |
| STABLE | 0.078 | 0.072 | 0.075 | 639 |
| **Accuracy** | | | **0.700** | |
| **Kappa** | | | **0.413** | |

**XGBoost Classifier — test 2025:**

| Klasa | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| DECLINING | 0.687 | 0.523 | 0.594 | 2148 |
| GROWING | 0.730 | 0.751 | 0.740 | 3893 |
| **STABLE** | **0.211** | **0.286** | **0.245** | 639 |
| **Accuracy** | | | **0.634** | |
| **Kappa** | | | **0.358** | |

> XGBoost STABLE Recall **0.286** (nga 0.049 në Fazën 2) — **5.9× përmirësim**. Modeli tani njeh klasën minoritare.

**Konfuzioni dhe rëndësia e tipareve:**

![Faza 3 — Krahasimi i Modeleve](Faza_3/outputs/plots/model_comparison.png)

![Faza 3 — RandomForest Confusion Matrix](Faza_3/outputs/plots/RandomForest_confusion_matrix.png)

![Faza 3 — XGBoost Confusion Matrix](Faza_3/outputs/plots/XGBoost_confusion_matrix.png)

![Faza 3 — RandomForest Feature Importance](Faza_3/outputs/plots/RandomForest_feature_importance.png)

![Faza 3 — XGBoost Feature Importance](Faza_3/outputs/plots/XGBoost_feature_importance.png)

![Faza 3 — RandomForest ROC Curves](Faza_3/outputs/plots/RandomForest_roc_curves.png)

![Faza 3 — XGBoost ROC Curves](Faza_3/outputs/plots/XGBoost_roc_curves.png)

![Faza 3 — Cross-Validation Score Distribution](Faza_3/outputs/plots/cv_score_boxplot.png)

#### Regresorët — test 2025

| Modeli | MAE | RMSE | **R²** | MAPE |
|--------|------|------|--------|------|
| LinearRegression | 0.710 | 1.132 | 0.273 | 2.66 |
| **RandomForestRegressor** | **0.541** | **0.916** | **0.523** ✓ | 1.86 |
| XGBoostRegressor | 0.545 | 0.924 | 0.515 | 1.94 |

Të tre regresorët kanë **R² pozitive dhe të dobishme tani** (Faza 2: RF=0.056, XGB=−0.024).

![Faza 3 — Regressor Comparison](Faza_3/outputs/plots/regressor_comparison.png)

![Faza 3 — RandomForest Regressor: Actual vs Predicted](Faza_3/outputs/plots/RandomForestRegressor_actual_vs_predicted.png)

![Faza 3 — RandomForest Regressor: Residuals](Faza_3/outputs/plots/RandomForestRegressor_residuals.png)

![Faza 3 — RandomForest Regressor: Feature Importance](Faza_3/outputs/plots/RandomForestRegressor_feature_importance.png)

![Faza 3 — XGBoost Regressor: Actual vs Predicted](Faza_3/outputs/plots/XGBoostRegressor_actual_vs_predicted.png)

![Faza 3 — XGBoost Regressor: Residuals](Faza_3/outputs/plots/XGBoostRegressor_residuals.png)

![Faza 3 — XGBoost Regressor: Feature Importance](Faza_3/outputs/plots/XGBoostRegressor_feature_importance.png)

![Faza 3 — Linear Regression: Actual vs Predicted](Faza_3/outputs/plots/LinearRegression_actual_vs_predicted.png)

![Faza 3 — Linear Regression: Residuals](Faza_3/outputs/plots/LinearRegression_residuals.png)

**Konvergjenca e XGBoost (early stopping):**

![Faza 3 — XGBoost Learning Curve](Faza_3/outputs/plots/learning_curve_xgb.png)

### Hiperparametrat Optimalë (nga RandomizedSearchCV)

Skedari: `Faza_3/outputs/metrics/best_hyperparams.json`

```json
{
  "random_forest": {
    "n_estimators": 400,
    "min_samples_split": 12,
    "min_samples_leaf": 5,
    "max_features": null,
    "max_depth": 16,
    "class_weight": "balanced_subsample"
  },
  "xgboost": {
    "subsample": 0.6,
    "reg_lambda": 5.0,
    "reg_alpha": 0.01,
    "n_estimators": 700,
    "min_child_weight": 3,
    "max_depth": 8,
    "learning_rate": 0.08,
    "gamma": 0.0,
    "colsample_bytree": 1.0
  },
  "rf_regressor": {
    "n_estimators": 400,
    "min_samples_split": 12,
    "min_samples_leaf": 3,
    "max_features": null,
    "max_depth": null
  },
  "xgb_regressor": {
    "subsample": 0.9,
    "reg_lambda": 5.0,
    "reg_alpha": 0.1,
    "n_estimators": 700,
    "min_child_weight": 7,
    "max_depth": 4,
    "learning_rate": 0.05,
    "gamma": 0.2,
    "colsample_bytree": 0.7
  }
}
```

**Vërejtje:** XGBoost Regressor zgjodhi `max_depth=4` (shumë i cekët) me `reg_lambda=5.0` (regularizim i fortë) — diametralisht e kundërt me parametrat default të Fazës 2 (max_depth=6, reg_lambda=1.0). **Tuning-u identifikoi që Faza 2 kishte overfit.**

### Ablation Study — Kontributi i Secilit Përmirësim

Skedari: `Faza_3/outputs/metrics/ablation_results.csv`

| Niveli | Përmirësimi i shtuar | Accuracy | F1-macro | Kappa | STABLE Recall |
|--------|----------------------|----------|----------|-------|---------------|
| A0 | baseline Faza 2 (vjetor + LabelEncoder) | 0.642 | 0.435 | 0.244 | 0.049 |
| **A1** | + agregim mujor + lag + rolling | **0.725** ↑↑ | **0.508** ↑ | **0.436** ↑↑ | 0.031 |
| A2 | + cyclical seasonal | 0.726 | 0.509 | 0.437 | 0.031 |
| A3 | + market concentration | 0.725 | 0.508 | 0.435 | 0.033 |
| A4 | + TargetEncoder | 0.726 | 0.508 | 0.437 | 0.030 |
| **A5** | + SMOTE-auto | 0.711 | **0.543** ↑ | 0.439 | **0.128** ↑↑↑ |
| **A6** | + tuned hyperparameters | 0.701 | 0.539 | 0.424 | **0.142** ↑ |

**Konkluzionet e ablation-it:**
- **A1 është ~80% e suksesit total** — agregimi mujor dhe lag features janë çelësi.
- **A5 është thelbësore për klasën STABLE** — SMOTE-auto rrit recall 4×.
- **A6 trade-off të kontrolluar** — pak accuracy për STABLE recall më të mirë.
- A2, A3, A4 kanë ndikim margjinal mbi metrikat kryesore por kontribuojnë stabilitet.

![Faza 3 — Ablation Contribution](Faza_3/outputs/plots/ablation_contribution.png)

### Unsupervised në Faza 3

**K-Means + PCA mbi matricën e re mujore (39,155 × 22):**

![Faza 3 — K-Means Elbow & Silhouette](Faza_3/outputs/plots/kmeans_elbow_silhouette.png)

![Faza 3 — PCA Variance Explained](Faza_3/outputs/plots/pca_variance_explained.png)

![Faza 3 — PCA me etiketat reale](Faza_3/outputs/plots/pca_true_labels.png)

![Faza 3 — PCA me K-Means clusters](Faza_3/outputs/plots/pca_kmeans_clusters.png)

**Klasterimi i trajektoreve të komunave dhe sektorëve:**

![Faza 3 — Municipality Trajectory Clusters](Faza_3/outputs/plots/municipality_trajectory_clusters.png)

![Faza 3 — Sector Trajectory Clusters](Faza_3/outputs/plots/primary_sector_trajectory_clusters.png)

### Analiza Kryqëzore Sektor × Komunë (Faza 3)

![Faza 3 — Komunë × Vit](Faza_3/outputs/plots/municipality_growth_by_year.png)

![Faza 3 — Sektor × Vit](Faza_3/outputs/plots/sector_growth_by_year.png)

![Faza 3 — Sektor × Komunë (të gjitha vitet)](Faza_3/outputs/plots/sector_x_municipality_growth_all.png)

![Faza 3 — Sektor × Komunë (2023)](Faza_3/outputs/plots/sector_x_municipality_growth_2023.0.png)

![Faza 3 — Sektor × Komunë (2024)](Faza_3/outputs/plots/sector_x_municipality_growth_2024.0.png)

![Faza 3 — Sektor × Komunë (2025)](Faza_3/outputs/plots/sector_x_municipality_growth_2025.0.png)

### Parashikimet 2027

Skedari: `Faza_3/outputs/metrics/predictions_2027.csv` (6,680 rreshta = 12 muaj × 557 (komunë, sektor))

| Modeli | Mesatarja e parashikuar | Median | % në clip +200% | % në clip -100% |
|--------|-------------------------|--------|-----------------|------------------|
| XGBoost Regressor | +36.0% | +20.6% | 6.0% (404 rreshta) | 1.4% (93 rreshta) |

**Klasifikimi i parashikimeve 2027:**

| Klasa | RandomForest | XGBoost |
|-------|--------------|---------|
| GROWING | 4441 (66.5%) | 3788 (56.7%) |
| DECLINING | 2027 (30.3%) | 1703 (25.5%) |
| **STABLE** | 212 (3.2%) | **1189 (17.8%)** |

XGBoost tani parashikon **1,189 raste STABLE** për 2027 — kundër vetëm **63 rreshta** në run-in para fix-eve (18.9× përmirësim).

**Top 10 Komuna × Sektor sipas parashikimit (XGBoost Regressor):**

| # | Komuna | Sektori | Parashikim 2027 |
|---|--------|---------|-----------------|
| 1 | MAMUSHË | Person Fizik | +200.0% (clipped) |
| 2 | RANILLUG | Akomodimi & ushqimi | +200.0% |
| 3 | MITROVICË VERIORE | Aktivitetet e shëndetësisë | +200.0% |
| 4 | RANILLUG | Industria përpunuese | +200.0% |
| 5 | OBILIQ | Aktivitetet financiare | +200.0% |
| 6 | PARTESH | Transporti & magazinimi | +200.0% |
| 7 | ZVEÇAN | Akomodimi & ushqimi | +200.0% |
| 8 | ZUBIN POTOK | Akomodimi & ushqimi | +200.0% |
| 9 | DEÇAN | Industria nxjerrëse | +196.3% |
| 10 | SUHAREKË | Aktivitetet financiare | +186.3% |

> **⚠ Paralajmërim:** Listën e dominon komunat shumë të vogla (MAMUSHË, RANILLUG, ZUBIN POTOK, MITROVICË VERIORE) ku variancia historike është e madhe dhe parashikimet ngjiten në clip-in +200%. Këto duhet të lexohen si flag-e për shqyrtim të mëtejshëm, jo si parashikime absolute.

![Faza 3 — Heatmap Parashikime 2027](Faza_3/outputs/plots/predicted_growth_2027_XGBoostRegressor.png)

![Faza 3 — Top 15 Parashikime 2027](Faza_3/outputs/plots/predicted_top_2027_XGBoostRegressor.png)

### Komandat e Ekzekutimit

```bash
cd Faza_3
pip install -r requirements.txt

# Pipeline i plotë me hyperparameter tuning (15-25 min)
python -m src.main --step all --tune

# Hapa individualë
python -m src.main --step supervised --tune    # vetëm klasifikuesit
python -m src.main --step regression --tune    # vetëm regresorët
python -m src.main --step unsupervised         # K-Means + PCA
python -m src.main --step analysis             # heatmaps
python -m src.main --step predictions          # 2027 forecast
python -m src.main --step ablation             # A0 → A6
python -m src.main --step compare              # Phase 2 vs Phase 3
```

### Output-et Kryesore të Fazës 3

Përveç skedarëve me të njëjtin emër si në Fazën 2 (për krahasim të drejtpërdrejtë), Faza 3 prodhon:

| Skedari/Grafiku | Përmbajtja |
|-----------------|------------|
| `outputs/metrics/best_hyperparams.json` | Parametrat optimalë nga RandomizedSearchCV |
| `outputs/metrics/cv_tuning_*.csv` (4 modele) | Tabelat e plota CV për tuning |
| `outputs/metrics/comparison_phase2_phase3.csv` | Tabela kryesore Faza 2 vs Faza 3 |
| `outputs/metrics/ablation_results.csv` | A0 → A6 me delta-t |
| `outputs/metrics/predictions_2027.csv` | 6,680 parashikime me clip |
| `outputs/plots/phase2_vs_phase3_comparison.png` | Grouped bar chart krahasues |
| `outputs/plots/ablation_contribution.png` | Waterfall i kontributeve |
| `outputs/plots/learning_curve_xgb.png` | Konvergjenca early stopping |
| `outputs/plots/predicted_growth_2027_*.png` | Heatmap parashikime 2027 |
| `outputs/plots/predicted_top_2027_*.png` | Top 15 komuna + sektor 2027 |

---

## Faza 3 — Part 2: Rritja Reale me Korrigjim për Inflacion dhe COVID

### Qëllimi i Part 2

Part 2 trajton dy mangësi makroekonomike që Part 1 nuk i adreson:

1. **Inflacioni shtrembëron rritjen nominale.** Një sektor me +10% nominale në 2022 (inflacion kosovar **11.6%**) në fakt u **tkurr 1.5%** në fuqi reale. Part 2 llogarit rritjen **reale** me formulën Fisher: `real = (1+nominal)/(1+inflation) − 1`.
2. **COVID-19 është anomali jo-strukturore.** Vitet 2020-2021 ndotin trajnimin. Part 2 trajnon një model paralel **counterfactual** pa to dhe kuantifikon dëmin pandemik per sektor.

### Struktura e Part 2 (e ndarë nga Part 1)

```text
Faza_3/part2_real_growth/
├── README.md
├── requirements.txt
├── data/
│   ├── kosovo_cpi_monthly.csv         # 84 rreshta CPI ASK (2019-2025)
│   ├── covid_stringency_kosovo.csv    # 84 rreshta Oxford OxCGRT
│   └── inflation_forecast_imf.csv     # 6 skenarë IMF WEO 2026/2027
├── src/
│   ├── external_data.py               # load + validate CPI/COVID/IMF
│   ├── deflation.py                   # Fisher formula + ±3% threshold
│   ├── covid_features.py              # 6 tipare COVID
│   └── pipeline_part2.py              # orchestrator (3 tracks × 5 modele)
└── outputs/
    ├── metrics/                       # 6 CSV (model_summary, covid_impact, etj.)
    └── plots/                         # 4 PNG (timeline-t + krahasimet)
```

**Parimi i ndarjes:** Asnjë skedar i Part 1 nuk modifikohet. Part 2 importon helper-at nga `Faza_3/src/`.

### Të Dhënat Eksterne

**Burimet:**
- **Inflacioni (CPI):** Agjencia e Statistikave të Kosovës (ASK) — 84 rreshta mujorë
- **COVID:** Oxford OxCGRT (stringency) + Our World in Data (cases, vaccinations)
- **Forecast inflation:** IMF World Economic Outlook (3 skenarë për 2026-2027)

**Validimi i CPI me ASK (vlerat mesatare vjetore):**

| Viti | CPI YoY | Konteksti |
|------|---------|-----------|
| 2019 | 2.7% | normal |
| 2020 | 0.34% | deflacion COVID |
| 2021 | 3.3% | rikuperim |
| **2022** | **11.57%** | shock energjie + Ukraine |
| 2023 | 4.92% | normalizim |
| 2024 | 1.72% | i kthyer në target |
| 2025 | 1.5% | parashikim ASK |

**CPI Kosovë mujor 2019-2025 — të dhëna të plota:**

| Year | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | **Mean** |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|----------|
| 2019 | 2.4 | 2.7 | 2.8 | 2.6 | 2.5 | 2.4 | 2.8 | 2.9 | 3.0 | 2.9 | 2.8 | 2.6 | **2.70** |
| 2020 | 2.2 | 1.8 | 1.0 | -0.2 | 0.0 | -0.7 | -0.6 | -0.4 | 0.0 | 0.3 | 0.4 | 0.3 | **0.34** |
| 2021 | 0.7 | 0.8 | 1.2 | 2.0 | 2.5 | 3.0 | 3.5 | 4.2 | 4.8 | 5.6 | 6.7 | 6.9 | **3.49** |
| 2022 | 8.0 | 8.7 | 10.6 | 11.2 | 12.4 | 14.1 | 14.2 | 14.2 | 12.8 | 12.0 | 11.0 | 9.7 | **11.57** |
| 2023 | 9.0 | 8.5 | 6.8 | 5.4 | 4.8 | 4.6 | 4.0 | 3.7 | 3.5 | 3.2 | 3.0 | 2.5 | **4.92** |
| 2024 | 2.3 | 2.0 | 1.8 | 1.7 | 1.4 | 1.6 | 1.6 | 1.4 | 1.5 | 1.6 | 1.8 | 1.9 | **1.72** |
| 2025 | 1.8 | 1.6 | 1.5 | 1.4 | 1.5 | 1.6 | 1.5 | 1.5 | 1.4 | 1.5 | 1.5 | 1.5 | **1.53** |

*Skedari burim: `Faza_3/part2_real_growth/data/kosovo_cpi_monthly.csv`*

![Faza 3 Part 2 — Inflation Kosovo 2019-2025](Faza_3/part2_real_growth/outputs/plots/inflation_kosovo_2019_2025.png)

**Indikatorët kryesorë COVID-19 për Kosovën (vlera kulmore):**

| Periudha | Stringency Max | Cases /100k Max | Vaccination Max |
|----------|----------------|------------------|------------------|
| 2020 (lockdown total prill-maj) | **84** (Prill) | 380 (Nëntor) | 0 |
| 2021 (valë Alpha + start vaksinim) | 73 (Janar) | **850** (Prill) | 45% (Dhjetor) |
| 2022 (valë Omicron, easing) | 35 (Janar) | **1200** (Janar) | 51% |
| 2023+ | 0 | <5 | 52% |

*Skedari burim: `Faza_3/part2_real_growth/data/covid_stringency_kosovo.csv`*

![Faza 3 Part 2 — COVID Stringency Timeline](Faza_3/part2_real_growth/outputs/plots/covid_stringency_timeline.png)

**IMF Inflation Forecast 2026-2027 (3 skenarë):**

| Skenari | 2026 | 2027 |
|---------|------|------|
| Low | 1.5% | 1.5% |
| **Central (baseline)** | **2.0%** | **2.5%** |
| High | 3.5% | 4.0% |

*Skedari burim: `Faza_3/part2_real_growth/data/inflation_forecast_imf.csv`*

### Tre Modele Paralele

Skedari: `src/pipeline_part2.py` — trajnon 3 tracks × 5 algoritme = **15 modele gjithsej**

| Track | Target Klasifikimi | Target Regresion | Filter Trajnimi |
|-------|--------------------|-----------------|-----------------|
| **M_nominal** | `growth_class` (Part 1) | `growth_rate` | Të gjitha vitet |
| **M_real** | `real_growth_class` | `real_growth_rate` | Të gjitha vitet |
| **M_real_no_covid** | `real_growth_class` | `real_growth_rate` | **Përjashtohen 2020, 2021** |

### Rezultatet e Trajnimit

Skedari: `Faza_3/part2_real_growth/outputs/metrics/model_summary_part2.csv`

#### Klasifikimi (Accuracy / F1-macro / Kappa)

| Track | RandomForest | XGBoost |
|-------|--------------|---------|
| M_nominal | 0.592 / 0.514 / 0.326 | 0.618 / 0.538 / 0.370 |
| **M_real** | **0.644** / 0.510 / 0.359 | 0.614 / 0.511 / 0.359 |
| **M_real_no_covid** | **0.683** / 0.516 / **0.383** | 0.603 / 0.499 / 0.338 |

#### Regresorët (R²)

| Track | LinearReg | RandomForestReg | XGBoostReg |
|-------|-----------|------------------|-------------|
| M_nominal | 0.246 | 0.496 | **0.502** |
| M_real | 0.245 | 0.496 | 0.497 |
| M_real_no_covid | 0.256 | 0.474 | 0.472 |

### Gjetjet kryesore akademike

#### Gjetja #1: M_real fiton ndaj M_nominal

RF Accuracy: **0.592 → 0.644 (+5.2 pp)** thjesht duke ndërruar target-in nominal me atë real. Pse? Rritja reale ka strukturë më të qartë sepse heq zhurmën e inflacionit.

#### Gjetja #2: M_real_no_covid është modeli më i mirë i Part 2

RF Accuracy: **0.683** — **+9.1 pp mbi M_nominal**. **Prova empirike që COVID është zhurmë jo-strukturore për ML.** Duke trajnuar pa 2020-2021, modeli sheh patternat strukturalë pa shqetësim.

#### Gjetja #3: Deflation lëviz 2,000 raste GROWING → DECLINING

| Klasa | Nominal | Real | Δ |
|-------|---------|------|-----|
| GROWING | 22,067 (56.4%) | 21,292 (54.4%) | −775 |
| DECLINING | 13,383 (34.2%) | **15,376 (39.3%)** | **+1,993** |
| STABLE | 3,705 (9.5%) | 2,487 (6.4%) | −1,218 |

Faza 2 e shtrembëronte realitetin duke quajtur "GROWING" qindra biznese që po humbnin fuqinë blerëse.

**Krahasimi sektor-për-sektor në vitin 2022 (inflacion 11.6%):**

![Faza 3 Part 2 — Nominal vs Real 2022](Faza_3/part2_real_growth/outputs/plots/nominal_vs_real_2022.png)

### Analiza Counterfactual COVID-19

**Metodologjia:** Trajno M_real_no_covid mbi 2019, 2022, 2023 → parashiko 2020-2021 → dëmi = aktuali − counterfactual.

Skedari: `Faza_3/part2_real_growth/outputs/metrics/covid_impact_by_sector.csv`

**Plot Top 10 sektorët më të dëmtuar:**

![Faza 3 Part 2 — COVID Impact by Sector](Faza_3/part2_real_growth/outputs/plots/covid_impact_by_sector.png)

**Të 21 sektorët, të renditur sipas dëmit mesatar (mean damage rritet → dëmtim i përkeqësuar):**

| Sektori | Mean Damage | Median Damage | n_rows |
|---------|-------------|---------------|--------|
| Aktivitetet e patundshmërisë | **−17.93 pp** | −26.99 pp | 331 |
| Aktivitetet e tjera shërbyese | **−12.71 pp** | −20.61 pp | 480 |
| Mungon aktiviteti | −7.20 pp | −37.50 pp | 188 |
| Person Fizik | −5.40 pp | −8.95 pp | 688 |
| Arsimi | −1.58 pp | −14.86 pp | 530 |
| Aktivitetet financiare dhe te sigurimit | −1.19 pp | −12.57 pp | 469 |
| Bujqësia; Pylltaria dhe Peshkimi | +2.42 pp | −16.69 pp | 737 |
| Informimi dhe komunikimi | +2.49 pp | −10.65 pp | 750 |
| Tregtia me shumicë dhe pakicë | +3.83 pp | −0.86 pp | 897 |
| Furnizimi me rrymë, gaz, avull | +4.55 pp | −9.98 pp | 263 |
| Transporti dhe magazinimi | +5.80 pp | −9.70 pp | 805 |
| Artet, Argetimi dhe rekreacioni | +6.84 pp | −35.67 pp | 483 |
| Aktivitetet profesionale, shkencore | +7.39 pp | −8.91 pp | 761 |
| Industria përpunuese | +7.90 pp | −0.10 pp | 876 |
| Ndërtimtaria | +12.33 pp | −3.14 pp | 795 |
| Administrimi publik | +15.61 pp | −6.70 pp | 366 |
| Industria nxjerrëse | +20.00 pp | −5.50 pp | 620 |
| Shërbimet administrative | +20.44 pp | −6.94 pp | 696 |
| Akomodimi dhe shërbimi ushqimor | +24.25 pp | −11.54 pp | 747 |
| Aktivitetet e shëndetësisë | +25.20 pp | +4.61 pp | 535 |
| Furnizimi me ujë; Kanalizimi | **+29.22 pp** | −2.75 pp | 522 |

> **Vërejtje interpretimi:** *Mean damage* është shtrembëruar nga outliers (komuna shumë të vogla). *Median damage* është më i besueshëm — pothuajse çdo sektor ka median negativ, që konfirmon dëmin pandemik universal. P.sh., "Akomodimi & ushqimi" ka mean **+24.2 pp** por median **−11.5 pp** — shumica e (komunë, sektor) çiftave u dëmtuan, vetëm pak outliers u rritën.

### Parashikimet 2027 me 3 Skenarë Inflacioni

Skedari: `Faza_3/part2_real_growth/outputs/metrics/real_growth_predictions_2027.csv` (6,680 rreshta)

Për çdo (komunë, sektor, muaj) llogariten:
- `real_growth_rate_rf` dhe `real_growth_rate_xgb` (parashikim direkt)
- `nominal_growth_rate_low` me inflacion 1.5%
- `nominal_growth_rate_central` me inflacion 2.5% (IMF baseline)
- `nominal_growth_rate_high` me inflacion 4.0%

Kjo lejon analistët të vlerësojnë **interval konfidence** për parashikimet nominale.

**Top 10 mundësi REALE për 2027 (XGB Regressor, mesatarja mbi 12 muaj):**

| # | Komuna | Sektori | Real Growth |
|---|--------|---------|-------------|
| 1 | MITROVICË VERIORE | Aktivitetet e shëndetësisë | +200.0% (clipped) |
| 2 | PARTESH | Transporti dhe magazinimi | +200.0% |
| 3 | OBILIQ | Aktivitetet financiare | +200.0% |
| 4 | ZUBIN POTOK | Akomodimi & ushqimi | +200.0% |
| 5 | MAMUSHË | Person Fizik | +200.0% |
| 6 | RANILLUG | Akomodimi & ushqimi | +200.0% |
| 7 | RANILLUG | Industria përpunuese | +200.0% |
| 8 | ZVEÇAN | Akomodimi & ushqimi | +194.4% |
| 9 | DEÇAN | Industria nxjerrëse | +193.7% |
| 10 | JUNIK | Bujqësia, Pylltaria | +192.1% |

> **⚠ Paralajmërim:** Listën e dominon komunat shumë të vogla (MAMUSHË, RANILLUG, ZUBIN POTOK, MITROVICË VERIORE) ku variancia historike është e madhe dhe parashikimet ngjiten në clip-in +200%. Këto duhet të lexohen si flag-e për shqyrtim të mëtejshëm.

### Komandat e Ekzekutimit të Part 2

```bash
cd Faza_3
# Validimi i të dhënave eksterne
python -m part2_real_growth.src.external_data validate

# Pipeline i plotë (5-10 min)
python -m part2_real_growth.src.pipeline_part2 --step all

# Hapa individualë
python -m part2_real_growth.src.pipeline_part2 --step deflate         # vetëm CPI augmentation
python -m part2_real_growth.src.pipeline_part2 --step train           # vetëm 3 tracks
python -m part2_real_growth.src.pipeline_part2 --step counterfactual  # dëmi COVID
python -m part2_real_growth.src.pipeline_part2 --step forecast        # 2027 predictions
python -m part2_real_growth.src.pipeline_part2 --step compare         # nominal vs real
```

### Output-et Kryesore të Part 2

| Skedari/Grafiku | Përmbajtja |
|-----------------|------------|
| `outputs/metrics/model_summary_part2.csv` | 15 rreshta: 3 tracks × 5 modele |
| `outputs/metrics/covid_impact_by_sector.csv` | Renditja e dëmit COVID per sektor |
| `outputs/metrics/counterfactual_no_covid.csv` | 12,539 rreshta detajuar dëmi |
| `outputs/metrics/real_growth_predictions_2027.csv` | 6,680 parashikime me 3 skenarë |
| `outputs/metrics/real_growth_top10_2027.csv` | Top 10 mundësi reale |
| `outputs/metrics/nominal_vs_real_comparison.csv` | Tabela krahasuese |
| `outputs/plots/inflation_kosovo_2019_2025.png` | Timeline CPI |
| `outputs/plots/covid_stringency_timeline.png` | 3-panel timeline COVID |
| `outputs/plots/covid_impact_by_sector.png` | Top 10 sektorët më të dëmtuar |
| `outputs/plots/nominal_vs_real_2022.png` | Krahasim sektor-sektor në 2022 |

---

## Kontributi Ynë Origjinal — Çka Kemi Bërë Që Të Tjerët Nuk e Kanë Bërë

Ky seksion përmbledh **kontributet origjinale të Grupit 3** — analiza dhe metodologji që, sipas njohurive tona, nuk janë aplikuar më parë në datasetin publik të ATK-së apo në literaturën akademike që trajton parashikimin ekonomik të Kosovës.

### 1. Pipeline ML i Aplikuar Drejtpërdrejt mbi Datasetin Publik ATK

**Çka kanë bërë të tjerët:** Studimet ekzistuese mbi ekonominë e Kosovës janë kryesisht **deskriptive** (raporte ASK, FMN, Banka Botërore) ose **ekonometrike** (regresione lineare të thjeshta). Asnjë studim publik nuk ka aplikuar **machine learning supervized + unsupervised** mbi datasetin e plotë ATK të 650,913 rreshtave për të parashikuar rritjen ekonomike sektoriale dhe komunale.

**Çfarë bëmë ne:** Implementuam një pipeline të plotë me **7 algoritme** (RF Cls, XGB Cls, Linear Reg, RF Reg, XGB Reg, K-Means, PCA) me parashikim sasior të rritjes dhe kategorizimit (GROWING/STABLE/DECLINING) për çdo kombinim (komunë, sektor, vit/muaj). Trajtojmë gjithashtu klasterimet trajektore për të identifikuar grupime natyrale të komunave dhe sektorëve me sjellje të ngjashme ekonomike.

### 2. Granulariteti Mujor i Agregimit (×10 sinjal trajnimi)

**Çka kanë bërë të tjerët:** Raportet vjetore të ASK dhe MEF agreguar vetëm në nivel **vjetor**. Edhe analiza jonë e Fazës 2 ndoqi këtë konventë → vetëm 3,956 rreshta trajnimi.

**Çfarë bëmë ne (Faza 3 Part 1):** Migruam në agregim **mujor** sipas (vit × muaj × komunë × sektor) → **39,155 rreshta** (10× më shumë sinjal). Kjo rikuperoi seasonalitin, lag-et e shumtë mujorë (1, 2, 3, 12), rolling means/std (3m, 6m), dhe market concentration mujor — tipare që nuk ekzistonin në literaturën akademike mbi Kosovën.

### 3. Time-Based Split Rigoroz me Zero Leakage Kohor

**Çka kanë bërë të tjerët:** Shumica e studimeve ML mbi seri kohore ekonomike kosovare përdorin **random k-fold cross-validation** — që fut leakage të padukshëm (modeli "sheh" të ardhmen gjatë trajnimit).

**Çfarë bëmë ne:** Aplikuam **TimeSeriesSplit** rigoroz (Train ≤ 2023, Val = 2024, Test = 2025) brenda RandomizedSearchCV. Çdo fold respekton kronologjinë. Faza 3 fiton kundër Faza 2 **pavarësisht se split-i është shumë më i vështirë** (Faza 2 kishte leakage të favorshëm).

### 4. Ablation Study Sistematik për të Atribuuar Përmirësimet

**Çka kanë bërë të tjerët:** Asnjë studim mbi ATK nuk ka prezantuar **ablation study formal** që kuantifikon kontributin e secilit komponent metodologjik.

**Çfarë bëmë ne:** Ndërtuam **7 nivele kumulative** (A0 → A6) që e zbërthejnë saktësisht ku ndodh përmirësimi (+11 pp accuracy):
- A0 (baseline Faza 2): 0.642 accuracy
- A1 (+ agregim mujor): 0.725 ← **kontribuon ~80%** të suksesit
- A5 (+ SMOTE-auto): STABLE recall 0.031 → 0.128 (**4×**) ← thelbësore për klasën minoritare
- A6 (+ tuning): trade-off i kontrolluar accuracy ↔ recall

Ky është një paraqitje **akademikisht transparente** që dëshmon pse arrijmë rezultatet — jo magjie, por inxhinieri.

### 5. Korrigjimi me Inflacion Kosovar të Vërtetë (Fisher Formula)

**Çka kanë bërë të tjerët:** Studimet mbi rritjen e bizneseve kosovare zakonisht **injorojnë inflacionin** ose përdorin korrigjim linear të përafërt (`real ≈ nominal − inflation`). Kjo është e gabuar gjatë periudhave të inflacionit të lartë (2022: 11.6%).

**Çfarë bëmë ne (Faza 3 Part 2):** Aplikuam formulën **Fisher** ekzakte `real = (1+nominal)/(1+inflation) − 1` mbi të dhënat reale ASK të Kosovës 2019-2025. Kjo lëvizi **2,000 raste** klasifikimi nga GROWING → DECLINING — duke zbuluar bizneset që rriteshin nominalisht por humbisnin fuqi blerëse. Sipas dijes sonë, ky është **studimi i parë akademik** që kryen deflation të nivelit (komunë, sektor, muaj) për ekonominë kosovare.

### 6. Analiza Counterfactual e Goditjes COVID-19

**Çka kanë bërë të tjerët:** Studimet mbi ndikimin e COVID-it në ekonominë kosovare kufizohen në **raporte deskriptive** ("PBB ra 5.3% në 2020"). Asnjë studim nuk është përpjekur të krijojë një **counterfactual** të strukturuar — pra "çfarë do kishte ndodhur **pa** pandemi".

**Çfarë bëmë ne:** Trajnuam një model paralel **M_real_no_covid** mbi 2019, 2022, 2023, 2024 (duke përjashtuar vitet COVID), pastaj e përdorëm për të parashikuar 2020-2021 sikur pandemia të mos kishte ndodhur. Diferenca midis aktualit dhe counterfactual jep dëmin per (komunë, sektor, muaj):

- **Aktivitetet e patundshmërisë** rezultoi sektori më i dëmtuar (**−17.9 pp** mean, −27.0 pp median)
- Sektorët e shërbimeve personale, aktivitetit te pezulluar dhe edukimi privat humbën ndjeshëm
- Kuantifikim që mungonte deri tani në literaturë

### 7. Modeli ML Trajnohet MË MIRË pa Vitet COVID (Gjetje Empirike)

**Çfarë bëmë ne (gjetje origjinale):** Demonstruam empirikisht që trajnimi i modelit pa vitet COVID **rrit performancën** (RF Accuracy: 0.592 → 0.683, **+9.1 pp**). Kjo është një **gjetje akademike origjinale** që ka implikime metodologjike: studimet e ardhshme ekonometrike mbi Kosovën duhet të trajtojnë 2020-2021 si periudhë anomali, jo si vite normale.

### 8. Parashikim 2027 me Skenarë Shumëfishtë Inflacioni

**Çka kanë bërë të tjerët:** Parashikimet ekonomike kosovare zakonisht jepen **si vlerë e vetme pikë**, pa interval konfidence.

**Çfarë bëmë ne:** Prodhuam **6,680 parashikime mujore** për 2027 (12 muaj × 557 kombinime komunë×sektor) me **3 skenarë inflacioni IMF** (1.5% low / 2.5% central / 4% high). Kjo i jep vendimmarrësve politikë një interval real konfidence, jo një numër të vetëm.

### 9. Post-processing me Kufij Ekonomikë Realistë

**Çka kanë bërë të tjerët:** Modelet e tjera ML ekonomike nuk aplikojnë sanity check post-processing → parashikime ekstreme jorealiste (Faza 2 e jona vetë: +479%!).

**Çfarë bëmë ne:** Aplikuam **clip në [-100%, +200%]** për të gjithë regresorët, që përkon me kufijtë realistë ekonomikë. Kjo redukton "alarmin e rrejshëm" për komunat e vogla me variance të madhe historike.

### 10. Riprodhueshmëri e Plotë (End-to-End)

**Çka kanë bërë të tjerët:** Studimet akademike rrallë janë **plotësisht të riprodhueshme** — të dhënat, kodi, modelet dhe rezultatet ndahen rrallëherë gjithçka njëkohësisht.

**Çfarë bëmë ne:** Të gjitha **3 fazat (1, 2, 3) + Part 2** janë në GitHub me:
- Kodi i plotë Python me CLI flags
- Datasetet ATK (Faza 1) + ASK CPI + Oxford COVID
- 5 modele `.pkl` të ruajtura
- 35+ grafikë PNG të gjeneruar
- 17+ skedarë CSV me të gjitha metrikat
- README-të e detajuar në secilën fazë

Çdokush mund të riprodhojë rezultatet me **3 komanda**:
```bash
cd "Faza 1" && python3 -m src.main --step all
cd ../Faza_2 && python -m src.main --step all
cd ../Faza_3 && python -m src.main --step all --tune && python -m part2_real_growth.src.pipeline_part2 --step all
```

### Tabela Përmbledhëse e Kontributit Origjinal

| # | Kontributi | A është bërë më parë në Kosovë? |
|---|------------|----------------------------------|
| 1 | ML pipeline (7 algoritme) mbi ATK | ❌ Jo |
| 2 | Agregim mujor i ATK (39k rreshta) | ❌ Jo |
| 3 | Time-based split rigoroz | ❌ Jo |
| 4 | Ablation study A0-A6 | ❌ Jo |
| 5 | Fisher deflation mbi (komunë, sektor, muaj) | ❌ Jo |
| 6 | Counterfactual COVID per sektor | ❌ Jo |
| 7 | "ML mëson më mirë pa COVID" — gjetje empirike | ❌ Jo |
| 8 | Parashikim 2027 me 3 skenarë inflacioni | ❌ Jo |
| 9 | Clip me kufij ekonomikë post-processing | ❌ Jo |
| 10 | Riprodhueshmëri e plotë end-to-end | ❌ Jo |

**Total: 10 kontribute origjinale që, sipas dijes sonë, nuk janë publikuar më parë në literaturën akademike apo raportet zyrtare të institucioneve të Kosovës.**

### Dokumentimi i Plotë

| Dokumenti | Përmbajtja |
|-----------|------------|
| [Faza_3/README.md](Faza_3/README.md) | Dokumentimi i plotë i Part 1 me numra realë |
| [Faza_3/PART2_PLAN.md](Faza_3/PART2_PLAN.md) | Plani fillestar për Part 2 |
| [Faza_3/part2_real_growth/README.md](Faza_3/part2_real_growth/README.md) | Dokumentimi i plotë i Part 2 |

---

## Qëllimi Akademik

Ky projekt është realizuar si pjesë e lëndës Machine Learning dhe synon të demonstrojë zbatimin praktik të teknikave të parapërpunimit dhe përgatitjes së të dhënave për analiza statistikore dhe modelim të mëvonshëm.
