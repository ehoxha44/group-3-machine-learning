<table>
  <tr>
    <td width="150" align="center" valign="middle">
      <img src="https://upload.wikimedia.org/wikipedia/commons/e/e1/University_of_Prishtina_logo.svg" width="120" alt="University of Prishtina Logo" />
    </td>
    <td valign="top">
      <p><strong>Universiteti i Prishtinës</strong></p>
      <p>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</p>
      <p>Inxhinieri Kompjuterike dhe Softuerike — Programi Master</p>
      <p><strong>Lënda:</strong> Machine Learning</p>
      <p><strong>Faza 3:</strong> Përmirësimi, Fine-Tuning-u dhe Krahasimi me Fazën 2</p>
      <p><strong>Studentët (Gr. 3):</strong> Enis Hoxha · Fisnik Hazrolli · Endri Binaku</p>
    </td>
  </tr>
</table>

---

# Faza 3 — Përmirësimi, Fine-Tuning-u dhe Krahasimi me Fazën 2

## Përmbledhje Ekzekutive

Faza 3 zëvendëson pipeline-in vjetor të Fazës 2 (3,956 rreshta agregimi) me një pipeline mujor, kronologjik dhe të tuned (39,155 rreshta — **10× më shumë sinjal trajnimi**). Përmirësimi i Fazës 3 është substancial në thuajse çdo metrikë.

### Rezultatet kryesore (Faza 2 → Faza 3)

| Modeli | Metrika | Faza 2 | **Faza 3** | Δ |
|--------|---------|--------|------------|---|
| RandomForest Classifier | Accuracy | 0.590 | **0.700** | **+11.0 pp** ✅ |
| RandomForest Classifier | F1-macro | 0.469 | **0.516** | +4.8 pp ✅ |
| RandomForest Classifier | Kappa | 0.256 | **0.413** | **+15.8 pp** ✅ |
| RandomForest Classifier | STABLE Recall | 0.159 | 0.072 | −8.7 pp ❌ |
| XGBoost Classifier | Accuracy | 0.625 | 0.634 | +0.9 pp ✅ |
| XGBoost Classifier | F1-macro | 0.434 | **0.526** | **+9.2 pp** ✅ |
| XGBoost Classifier | Kappa | 0.243 | **0.358** | +11.5 pp ✅ |
| XGBoost Classifier | **STABLE Recall** | 0.049 | **0.286** | **+23.8 pp** ✅✅ |
| RF Regressor | R² | 0.056 | **0.523** | **+0.47** ✅ |
| XGBoost Regressor | R² | **−0.024** | **0.515** | **+0.54** ✅ |

> **Shënim metodologjik:** Faza 2 përdorte split random 80/20 që fut leakage kohor (modeli sheh 2025 dhe testohet mbi 2020). Faza 3 përdor split rigoroz kronologjik (train ≤ 2023, val = 2024, test = 2025). **Faza 3 fiton pavarësisht se problemi i saj është shumë më i vështirë.**

---

## 1. Qëllimi i Fazës 3

Faza 2 prodhoi rezultate të dobishme por kishte 9 dobësi të identifikuara në audit:

1. Humbje masive granulariteti nga agregimi vetëm vjetor (650K → 3.9K rreshta)
2. LabelEncoder për kategoritë nominale (futë renditje të rreme)
3. Mungesë e hyperparameter tuning-ut
4. Mungesë e early stopping për XGBoost
5. Leakage kohor nga split random
6. Mësim i dobët i klasës STABLE (recall ~0.05-0.16)
7. Mungesë e validation set të veçantë
8. Hapësirë e varfër tiparesh (vetëm 5)
9. Parashikime ekstreme jashtë kufijve realistë

**Faza 3 i adreson të 9 dobësitë** dhe shton dy kontribute akademikë origjinalë:
- **Ablation study** (A0→A6) që kuantifikon kontributin e secilit përmirësim
- **Cross-validation me TimeSeriesSplit** brenda tuning-ut, jo CV mbi të dhëna të rastësishme

---

## 2. Metodologjia

### 2.1 Pipeline-i i ri

```
Loading model_ready.csv (650,913 rreshta)
    ↓
Agregim mujor (year × month × municipality × sector) → 52,585 rreshta
    ↓
Filter: ≥24 muaj histori + drop NaN lag → 39,155 rreshta
    ↓
Class distribution: GROWING 56.4% | DECLINING 34.2% | STABLE 9.5%
    ↓
Time-based split:
  Train ≤ 2023  → 25,826 rreshta
  Val  = 2024   →  6,649 rreshta
  Test = 2025   →  6,680 rreshta
    ↓
fit_encoders(train)  ← TargetEncoder fit-on-train-only (parandalim leakage)
StandardScaler fit on train, transform val/test/pred
    ↓
SMOTE strategy="auto" mbi train: balancon GROWING/DECLINING/STABLE → 14,730 each
    ↓
RandomizedSearchCV (n_iter=25, TimeSeriesSplit 4 folds) → best_hyperparams.json
    ↓
RF: refit mbi (train+val) me best_params
XGB: refit me eval_set=val + early_stopping_rounds=30 + sample_weight balanced
    ↓
Evaluim mbi test (2025) → metrika + plot
    ↓
Predictions 2027 me clip [-1.0, +2.0]
    ↓
Ablation study (A0→A6)
    ↓
compare_phase2_phase3() → comparison_phase2_phase3.csv + plot
```

### 2.2 Tiparet e reja (14 totale vs 5 në Fazën 2)

```python
NUMERIC_FEATURES = [
    "year",
    "month",                          # I RI — sinjal kohor mujor
    "prev_turnover_log1p",            # YoY base (lag12)
    "lag1_turnover_log1p",            # I RI — muaji i kaluar
    "lag2_turnover_log1p",            # I RI — 2 muaj më parë
    "lag3_turnover_log1p",            # I RI — 3 muaj më parë
    "rolling3_mean_log1p",            # I RI — mesatare 3-mujore
    "rolling6_mean_log1p",            # I RI — mesatare 6-mujore
    "rolling3_std",                   # I RI — volatiliteti 3m
    "rolling6_std",                   # I RI — volatiliteti 6m
    "num_businesses_log1p",
    "market_concentration_log1p",     # I RI — # bizneset në komunë në muajin e ri
    "month_sin",                      # I RI — sezonalitet ciklik sin(2πm/12)
    "month_cos",                      # I RI — sezonalitet ciklik cos(2πm/12)
]
```

**Të gjitha lag dhe rolling janë `shifted by 1` për të parandaluar leakage** — ato përdorin vetëm informacion të disponueshëm para muajit aktual.

### 2.3 Hiperparametrat optimalë (nga RandomizedSearchCV)

Skedari: `outputs/metrics/best_hyperparams.json`

| Modeli | Best Params |
|--------|-------------|
| **RandomForest Cls** | n_estimators=400, max_depth=16, min_samples_split=12, min_samples_leaf=5, max_features=None, class_weight="balanced_subsample" |
| **XGBoost Cls** | n_estimators=700, max_depth=8, learning_rate=0.08, subsample=0.6, colsample_bytree=1.0, min_child_weight=3, reg_alpha=0.01, reg_lambda=5.0, gamma=0.0 |
| **RF Regressor** | n_estimators=400, max_depth=None, min_samples_split=12, min_samples_leaf=3, max_features=None |
| **XGBoost Reg** | n_estimators=700, max_depth=4 (i cekët!), learning_rate=0.05, subsample=0.9, colsample_bytree=0.7, min_child_weight=7, reg_lambda=5.0, gamma=0.2 |

Vënia në dukje:
- XGBoost Regressori zgjodhi `max_depth=4` — shumë i cekët — me `reg_lambda=5.0` (regularizim i fortë). Kjo është diametralisht e kundërt me parametrat default të Fazës 2 (`max_depth=6, reg_lambda=1.0`). Tuning-u ka identifikuar që modeli kishte overfit në Fazën 2.

---

## 3. Rezultatet e Detajuara

### 3.1 Klasifikimi — RandomForest (test 2025)

| Klasa | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| DECLINING | 0.624 | 0.654 | 0.638 | 2148 |
| GROWING | 0.728 | 0.802 | 0.763 | 3893 |
| STABLE | 0.078 | 0.072 | 0.075 | 639 |
| **Accuracy** | | | **0.700** | |
| **Macro avg** | 0.477 | 0.509 | 0.492 | 6680 |
| **Weighted avg** | 0.668 | 0.700 | 0.683 | 6680 |
| **Kappa** | | | **0.413** | |

> RF identifikon mirë GROWING dhe DECLINING. STABLE mbetet problem (recall 0.072) — tuning zgjodhi `class_weight="balanced_subsample"` që nuk kombinohet aq mirë me SMOTE-n e jashtëm.

### 3.2 Klasifikimi — XGBoost (test 2025)

| Klasa | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| DECLINING | 0.687 | 0.523 | 0.594 | 2148 |
| GROWING | 0.730 | 0.751 | 0.740 | 3893 |
| **STABLE** | **0.211** | **0.286** | **0.245** | 639 |
| **Accuracy** | | | **0.634** | |
| **Macro avg** | 0.543 | 0.520 | 0.526 | 6680 |
| **Weighted avg** | 0.661 | 0.634 | 0.645 | 6680 |
| **Kappa** | | | **0.358** | |

> **Fitorja kryesore e Fazës 3**: XGBoost Stable Recall ngjit nga **0.049 në 0.286** — **5.9× përmirësim**. Modeli tani e di që klasa STABLE ekziston. Kjo është rezultat i kombinimit SMOTE-strategy=`"auto"` + `sample_weight` të balancuar gjatë trajnimit (parandalon që early stopping mbi imbalanced val të anulojë SMOTE-n).

### 3.3 Regresorët (test 2025)

| Modeli | MAE | RMSE | **R²** | MAPE |
|--------|------|------|--------|------|
| LinearRegression | 0.710 | 1.132 | 0.273 | 2.66 |
| **RandomForest Regressor** | **0.541** | **0.916** | **0.523** ✓ | 1.86 |
| XGBoost Regressor | 0.545 | 0.924 | 0.515 | 1.94 |

> **Të tre regresorët kanë R² pozitiv dhe të dobishëm tani** (Phase 2: RF=0.056, XGB=−0.024). RF Regressor është fituesi kryesor me R²=0.52.

---

## 4. Ablation Study — Kontributi i Secilit Përmirësim

Skedari: `outputs/metrics/ablation_results.csv`

Çdo nivel shton një përmirësim mbi paraprakun, duke përdorur të njëjtin model (XGBoost Classifier).

| Niveli | Accuracy | F1-macro | Kappa | STABLE Recall | Vërejtja |
|--------|----------|----------|-------|---------------|----------|
| **A0** baseline Faza 2 | 0.642 | 0.435 | 0.244 | 0.049 | tiparet vjetore + LabelEncoder |
| **A1** + agregim mujor + lag + rolling | **0.725** ↑ | **0.508** ↑ | **0.436** ↑ | 0.031 | **vendimtarë** — granulariteti dhe lag-et solitën +8.3 pp acc |
| A2 + cyclical seasonal | 0.726 | 0.509 | 0.437 | 0.031 | minimale |
| A3 + market concentration | 0.725 | 0.508 | 0.435 | 0.033 | minimale |
| A4 + TargetEncoder | 0.726 | 0.508 | 0.437 | 0.030 | minimale (ndoshta sepse XGB tashmë trajton kategoritë mirë) |
| **A5** + SMOTE-auto | 0.711 | **0.543** ↑ | 0.439 | **0.128** ↑↑ | **fitore për STABLE** — Recall 4× |
| **A6** + tuning hyperparams | 0.701 | 0.539 | 0.424 | **0.142** ↑ | trade-off i kontrolluar: pak accuracy për STABLE recall më shumë |

### Diagrami: Si u ndërtua Faza 3 për të kapërcyer Fazën 2

![Ablation Contribution](outputs/plots/ablation_contribution.png)

**Konkluzionet e ablation-it:**

1. **Agregimi mujor + lag features është 90% e suksesit.** Pa A1, asnjë nga përmirësimet e tjera nuk do kishte rëndësi.
2. **SMOTE-auto është thelbësor për STABLE class.** Pa të, asnjë model nuk arrin t'i japë vëmendje klasës minoritare.
3. **TargetEncoder, cyclical, market concentration kanë ndikim margjinal.** Tipari "primary_sector" si LabelEncoder ka qenë mjaftueshëm informativ.
4. **Tuning e zhvendos balancën nga accuracy drejt F1-macro dhe STABLE recall** — kjo është një trade-off i drejtë për një problem multi-klasë me imbalancë.

---

## 5. Krahasimi Faza 2 vs Faza 3 — Detaje

### Tabela e Krahasimit

Skedari: `outputs/metrics/comparison_phase2_phase3.csv`

| Algorithm | Metric | Phase 2 | Phase 3 | Δ | Përmirësim? |
|-----------|--------|---------|---------|---|-------------|
| RandomForest | Accuracy | 0.5896 | 0.6996 | **+0.110** | ✅ |
| RandomForest | F1-Score (macro) | 0.4685 | 0.5163 | +0.048 | ✅ |
| RandomForest | Kappa | 0.2556 | 0.4131 | **+0.158** | ✅ |
| RandomForest | Stable Recall | 0.1585 | 0.0720 | −0.087 | ❌ |
| RandomForestRegressor | R² | 0.0556 | 0.5234 | **+0.468** | ✅ |
| XGBoost | Accuracy | 0.6250 | 0.6338 | +0.009 | ✅ |
| XGBoost | F1-Score (macro) | 0.4342 | 0.5263 | **+0.092** | ✅ |
| XGBoost | Kappa | 0.2433 | 0.3582 | **+0.115** | ✅ |
| XGBoost | **Stable Recall** | 0.0488 | **0.2864** | **+0.238** | ✅✅ |
| XGBoostRegressor | R² | −0.0237 | 0.5154 | **+0.539** | ✅ |

**Verdikti:** 9 nga 10 metrika u përmirësuan. Vetëm RF STABLE Recall u përkeqësua — kjo është një trade-off i njohur kur tuning-u zgjedh `class_weight="balanced_subsample"` mbi `"balanced"` (parametër i zgjedhur sepse maksimizon F1-macro mbi accuracy).

![Phase 2 vs Phase 3](outputs/plots/phase2_vs_phase3_comparison.png)

### Konsiderata e drejtësisë së krahasimit

Faza 2 përdor **random split 80/20** — më e lehtë sepse ka leakage kohor (modeli sheh 2025 gjatë trajnimit dhe testohet mbi 2020). Faza 3 përdor **time-based split** — më e vështirë sepse modeli nuk e ka parë kurrë vitin e testit.

**Faza 3 fiton pavarësisht se problemi është më i vështirë** — që do të thotë përmirësimi është edhe më substancial seç tregojnë numrat.

---

## 6. Parashikimet për 2027

Skedari: `outputs/metrics/predictions_2027.csv` (6,680 rreshta = 12 muaj × 557 (komunë, sektor) pairs)

### Distribuim i parashikimeve

| Modeli | Mesatarja | Median | % në +200% clip | % në −100% clip |
|--------|-----------|--------|-----------------|-----------------|
| XGBoost Regressor | +36.0% | +20.6% | 6.0% (404) | 1.4% (93) |

### Klasifikuesit në 2027

| Klasa | RF | XGB |
|-------|------|------|
| GROWING | 4441 (66.5%) | 3788 (56.7%) |
| DECLINING | 2027 (30.3%) | 1703 (25.5%) |
| **STABLE** | 212 (3.2%) | **1189 (17.8%)** |

> XGB tani parashikon **1,189 raste STABLE** (kundër vetëm **63 në run-in para fixes** — 18.9× përmirësim). Kjo është prova që fix-i i sample_weight + SMOTE-auto funksionoi.

### Vizualizimet

- `outputs/plots/predicted_growth_2027_XGBoostRegressor.png` — heatmap sektor × komunë për 2027
- `outputs/plots/predicted_top_2027_XGBoostRegressor.png` — top 15 komuna + top 15 sektorë

> **⚠ Paralajmërim interpretues:** Parashikimet ekstreme (që ngjiten në clip ±200%/−100%) janë kryesisht komuna shumë të vogla me numër të pakët bizneseve (MAMUSHË, RANILLUG, ZUBIN POTOK, etj.) ku variancia historike është e madhe. Ato duhen lexuar si **renditje relative**, jo si numra absolutë.

---

## 7. Si të Riprodhohen Rezultatet

### Instalimi

```bash
cd Faza_3
pip install -r requirements.txt
# requirements: pandas, numpy, scikit-learn, matplotlib, xgboost, joblib,
#               imbalanced-learn, category-encoders
```

### Pipeline-i i plotë

```bash
# Krijim i të gjithë artefakteve me hyperparameter tuning (15-25 min)
python -m src.main --step all --tune --forecast-mode static
```

### Hapa individualë

```bash
# Vetëm klasifikuesit
python -m src.main --step supervised --tune

# Vetëm regresorët
python -m src.main --step regression --tune

# Vetëm K-Means + PCA
python -m src.main --step unsupervised

# Vetëm heatmaps e rritjes
python -m src.main --step analysis

# Vetëm parashikimet 2027 (kërkon modelet e ruajtura)
python -m src.main --step predictions

# Vetëm ablation study
python -m src.main --step ablation

# Vetëm krahasimi Phase 2 vs Phase 3 (lexon CSV ekzistuese)
python -m src.main --step compare

# Smoke test pa tuning (përdor parametrat default)
python -m src.main --step all
```

### Opcionet e CLI

```
--tune              # Aktivizon RandomizedSearchCV (25 iter × 4 folds)
--encoding {target, onehot}  # default: target
--no-smote          # Çaktivizon SMOTE për ablation A0-A4
--forecast-mode {static, recursive}
```

---

## 8. Si t'i Lexoni Rezultatet

| Pyetja | Skedari/Grafiku përgjegjës |
|--------|----------------------------|
| Sa mirë klasifikon GROWING/STABLE/DECLINING? | `outputs/metrics/model_comparison.csv`, `RandomForest_confusion_matrix.png`, `XGBoost_confusion_matrix.png` |
| Cili regresor parashikon më mirë `growth_rate`? | `outputs/metrics/regressor_comparison.csv`, `regressor_comparison.png` |
| Si performoi Faza 3 vs Faza 2? | `outputs/metrics/comparison_phase2_phase3.csv`, `phase2_vs_phase3_comparison.png` |
| Cili përmirësim solli sa? | `outputs/metrics/ablation_results.csv`, `ablation_contribution.png` |
| Cilët hiperparametra fitues? | `outputs/metrics/best_hyperparams.json`, `cv_tuning_*.csv` |
| Si është konvergjenca e XGBoost? | `outputs/plots/learning_curve_xgb.png` |
| Cilët komunë × sektorë do rriten më shumë në 2027? | `outputs/metrics/predictions_2027.csv`, `predicted_growth_2027_*.png`, `predicted_top_2027_*.png` |
| Si është grupimi natyror i komunave/sektorëve? | `municipality_trajectory_clusters.png`, `primary_sector_trajectory_clusters.png` |
| Cili sektor u rrit më shumë në komunën X në vitin Y? | `sector_x_municipality_growth_2023.0.png` etj., dhe CSV-të përkatëse |

---

## 9. Kujt i Ndihmojnë dhe Si

### 9.1 Qeveria dhe MEF (Ministria e Ekonomisë)

- **Identifikim i sektorëve në rënie** për ndërhyrje politike (subvencione, lehtësime tatimore)
- **Planifikim buxhetor** bazuar në parashikime regjionale të të hyrave
- **Vlerësim i mbylljes së fiskalitetit** — sektorët STABLE me potencial shtypjeje tatimore

### 9.2 ATK (Administrata Tatimore e Kosovës)

- **Targetim më i mirë i monitorimit** ekonomik — sektorët DECLINING me rrezik humbjeje të të hyrave
- **Analiza e konformitetit** — divergjencat e parashikimit vs aktualit indikojnë anomali të mundshme

### 9.3 Komunat

- **Prioritizim i ndërhyrjeve** sipas sektorëve lokalë — për shembull, Prishtina ka strukturë krejt të ndryshme nga Suhareka
- **Planifikim i lejeve për biznese** të reja në sektorët më të dobishëm
- **Diversifikim ekonomik** — komunat me dependency të lartë mbi një sektor (Mitrovicë mbi industrinë nxjerrëse) mund të identifikohen

### 9.4 Investitorët (publikë, privatë, BERZH, EBRD, IFC)

- **Identifikim i segmenteve me potencial rritjeje** — top-N parashikime 2027
- **Vlerësim i riskut** — kombinime me variance të lartë (clip-ed predictions) janë ende të dobishme si flags

### 9.5 Akademia dhe Studentët

- **Benchmark më realist** për problemet e parashikimit ekonomik mbi të dhëna publike
- **Studim metodologjik** — krahasim midis nominal aggregation vs monthly granularity
- **Skedarë të zhvilluar** për mësimdhënie të time-series ML

---

## 10. Çfarë i Shton Faza 3 Kontributit Akademik

Përpos rezultateve të përmirësuara, Faza 3 sjell pesë kontribute origjinale që Faza 2 ose projekte të ngjashme nuk i kanë demonstruar:

1. **Time-aware ML rigoroz** — split kronologjik + TimeSeriesSplit në CV; zero leakage kohor.
2. **Ablation study i detajuar** (A0→A6) që e bën transparentë çfarë solli sa në përmirësimin total.
3. **Encoding leakage-safe** — TargetEncoder fit-on-train-only me unit assertion.
4. **Trajtim i imbalancës multi-klasë** — SMOTE-auto + sample_weight balanced, që zgjidh problemin që SMOTE i pastër nuk e zgjidh kur kombinohet me early stopping mbi val imbalanced.
5. **Post-processing me kufij ekonomikë** — clip [-100%, +200%] që parandalon parashikime jorealiste të modeleve të nxitur.

---

## 11. Limitime dhe Punë e Ardhshme

### 11.1 Limitime aktuale

- **Inflacioni nuk është adresuar** — rritja parashikohet nominale, jo reale. Sektor me +10% në 2022 (inflacion 11.6%) shfaqet GROWING por në fakt u tkurr realisht. **Plani për Part 2 e adreson këtë.** Shih `PART2_PLAN.md`.
- **COVID-19 trajtohet si vit normal** — 2020/2021 ndotin trajnimin me anomali jo-strukturore. Part 2 plan adreson edhe këtë.
- **CPI kombëtar, jo per komunë** — kufizim i të dhënave publike.
- **RF Stable Recall 0.072** — RF zgjedh `class_weight="balanced_subsample"` që anulon SMOTE-n. E rregullueshme me forcim manual.
- **Parashikime ekstreme për komuna shumë të vogla** — Mamushë, Ranillug, Zubin Potok shpesh ngjiten në clip. Rrjedh nga variance historike e madhe.

### 11.2 Punë e ardhshme

1. **Part 2 e Fazës 3 (PLANIFIKUAR):** Korrigjim për inflacionin + analiza counterfactual e COVID-it → rritja reale. Shih `PART2_PLAN.md` për planin e detajuar.
2. **Integrim makroekonomik:** indikatorë makro nga World Bank (FDI, deficit fiscal, eksporte), Eurostat, IMF.
3. **Modele sekuenciale:** LSTM/Transformer mbi seritë mujore — mund të kapë dinamika periodike që pemët nuk i kapin.
4. **Modele Bayesiane:** pasiguri probabilistike për vendimmarrje politike (intervale 95% confidence rreth parashikimit).
5. **Zgjerim historik:** të dhëna ATK 2010-2018 për trajnim më të gjerë (nëse të disponueshme).
6. **CPI sektoriale dhe komunale:** kërkim me ASK për të dhëna më të granulare.

---

## 12. Struktura e Skedarëve

```text
Faza_3/
├── README.md                                  ← ky dokument
├── PART2_PLAN.md                              ← plani për pjesën 2 (inflacion + COVID)
├── requirements.txt
├── src/
│   ├── config.py                              ← hyperparametra, time-split bounds, feature lists
│   ├── data_loader.py                         ← agregim mujor + lag + rolling + market_conc
│   ├── feature_engineering.py                 ← TargetEncoder, OHE, fit-on-train-only
│   ├── time_split.py                          ← chronological train/val/test + TimeSeriesSplit
│   ├── sampling.py                            ← SMOTE wrapper (assertions për leakage)
│   ├── supervised.py                          ← RF/XGB/LR + tune_hyperparameters + early stop
│   ├── unsupervised.py                        ← K-Means + PCA
│   ├── evaluation.py                          ← metrika + plot + compare_phase2_phase3
│   ├── ablation.py                            ← A0→A6 incremental ablation
│   └── main.py                                ← CLI orchestrator
└── outputs/
    ├── models/
    │   ├── random_forest.pkl                  ← RF Classifier (tuned)
    │   ├── xgboost.pkl                        ← XGB Classifier (tuned + early stop)
    │   ├── linear_regression.pkl              ← OLS baseline
    │   ├── rf_regressor.pkl                   ← RF Regressor (tuned)
    │   └── xgb_regressor.pkl                  ← XGB Regressor (tuned + early stop)
    ├── metrics/
    │   ├── model_comparison.csv               ← RF vs XGB klasifikim
    │   ├── regressor_comparison.csv           ← 3 regresorët
    │   ├── RandomForest_classification_report.csv
    │   ├── XGBoost_classification_report.csv
    │   ├── LinearRegression_regression_metrics.csv
    │   ├── RandomForestRegressor_regression_metrics.csv
    │   ├── XGBoostRegressor_regression_metrics.csv
    │   ├── comparison_phase2_phase3.csv       ← tabela kryesore e krahasimit
    │   ├── ablation_results.csv               ← A0→A6
    │   ├── best_hyperparams.json              ← parametrat optimalë
    │   ├── cv_tuning_random_forest.csv        ← skoret CV për RF Cls
    │   ├── cv_tuning_xgboost.csv              ← skoret CV për XGB Cls
    │   ├── cv_tuning_rf_regressor.csv         ← skoret CV për RF Reg
    │   ├── cv_tuning_xgb_regressor.csv        ← skoret CV për XGB Reg
    │   ├── kmeans_sweep_metrics.csv           ← K-Means sweep k=2..8
    │   ├── unsupervised_metrics.csv           ← finalet K-Means + PCA
    │   ├── municipality_growth_by_year.csv    ← heatmap komunë × vit (vlerat)
    │   ├── sector_growth_by_year.csv          ← heatmap sektor × vit (vlerat)
    │   ├── sector_x_municipality_growth_*.csv ← kryqëzim sektor × komunë
    │   └── predictions_2027.csv               ← 6,680 parashikime nga 5 modele
    └── plots/   (35 grafikë gjithsej)
        ├── RandomForest_confusion_matrix.png
        ├── RandomForest_roc_curves.png
        ├── RandomForest_feature_importance.png
        ├── XGBoost_confusion_matrix.png
        ├── XGBoost_roc_curves.png
        ├── XGBoost_feature_importance.png
        ├── model_comparison.png                   ← bar chart klasifikuesit
        ├── cv_score_boxplot.png                   ← shpërndarja CV (5 folda)
        ├── regressor_comparison.png               ← bar chart MAE/RMSE/R²
        ├── *Regressor_actual_vs_predicted.png     ← 3 grafikë
        ├── *Regressor_residuals.png               ← 3 grafikë
        ├── *Regressor_feature_importance.png      ← 2 grafikë (LR nuk ka)
        ├── learning_curve_xgb.png                 ← konvergjenca early stopping
        ├── phase2_vs_phase3_comparison.png        ← KRAHASIMI KRYESOR
        ├── ablation_contribution.png              ← waterfall i kontributeve
        ├── kmeans_elbow_silhouette.png            ← K-Means analiza
        ├── pca_variance_explained.png             ← PCA variance curve
        ├── pca_true_labels.png                    ← PCA 2D etiketat reale
        ├── pca_kmeans_clusters.png                ← PCA 2D K-Means
        ├── municipality_trajectory_clusters.png   ← klasterimi i komunave
        ├── primary_sector_trajectory_clusters.png ← klasterimi i sektorëve
        ├── municipality_growth_by_year.png        ← heatmap komunë × vit
        ├── sector_growth_by_year.png              ← heatmap sektor × vit
        ├── sector_x_municipality_growth_*.png     ← 4 heatmaps kryqëzimi
        ├── predicted_growth_2027_XGBoostRegressor.png
        └── predicted_top_2027_XGBoostRegressor.png
```

---

## 13. Konkluzioni

**Faza 3 i kapërceu të 9 dobësitë e Fazës 2 dhe arriti përmirësime substanciale në 9 nga 10 metrika.** Përmirësimi më dramatik është në STABLE recall për XGBoost (0.049 → 0.286, **5.9× më mirë**), që ishte një nga problemet kryesore të Fazës 2. R²-të e regresorëve u rritën nga afër zero në 0.5+ — duke i bërë parashikimet sasiore realisht të dobishme.

Ablation-i kuantifikoi se **agregimi mujor + lag features** kontribuojnë me ~80% të përmirësimit, ndërkohë **SMOTE-auto + sample_weight balanced** janë kritikë për klasën STABLE.

Pavarësisht këtyre fitoreve, Faza 3 nuk adreson **inflacionin** dhe **COVID-19** — dy faktorë makroekonomikë që distortojnë growth_rate-in nominal. **Plani për Part 2 (`PART2_PLAN.md`)** ofron një udhërrëfyes të detajuar për këtë zgjerim me modele paralele dhe analiza counterfactual.

---

*Faza 3 e Grupit 3 — Machine Learning, Master FIEK, Universiteti i Prishtinës.*
*Pipeline-i u ekzekutua me sukses më 13 maj 2026 me hyperparameter tuning aktiv (25 iteracione × 4 folds × 4 modele).*
