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
      <p><strong>Studentët:</strong></p>
      <ul>
        <li>Enis Hoxha</li>
        <li>Fisnik Hazrolli</li>
        <li>Endri Binaku</li>
      </ul>
    </td>
  </tr>
</table>

# Projekti i Përgatitjes dhe Pastrimit të të Dhënave të Qarkullimit

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

Karakteristikat kryesore të datasetit:

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

## Qëllimi Akademik

Ky projekt është realizuar si pjesë e lëndës Machine Learning dhe synon të demonstrojë zbatimin praktik të teknikave të parapërpunimit dhe përgatitjes së të dhënave për analiza statistikore dhe modelim të mëvonshëm.
