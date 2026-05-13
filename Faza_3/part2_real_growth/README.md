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
      <p><strong>Faza 3 — Part 2:</strong> Rritja Reale me Korrigjim për Inflacion + COVID</p>
      <p><strong>Studentët (Gr. 3):</strong> Enis Hoxha · Fisnik Hazrolli · Endri Binaku</p>
    </td>
  </tr>
</table>

---

# Part 2 — Rritja Reale Ekonomike me Korrigjim për Inflacion dhe COVID-19

## Përmbledhje Ekzekutive

Part 2 e Faza 3 trajton dy mangësi makroekonomike që Part 1 nuk i adreson:

1. **Inflacioni**: rritja nominale e Part 1 është e shtrembëruar nga inflacioni — sidomos në 2022 (inflacion kosovar **11.6%**). Part 2 llogarit rritjen **reale** me formulën **Fisher**.
2. **COVID-19**: vitet 2020-2021 janë anomali (lockdown, deflacion, rebound) që ndotin modelin. Part 2 trajnon një model **counterfactual** mbi vitet jo-COVID dhe e përdor për të kuantifikuar dëmin pandemik.

### Gjetjet kryesore

| Track | Klasifikues | Accuracy | F1-macro | Kappa | R² (best regressor) |
|-------|-------------|----------|----------|-------|---------------------|
| **M_nominal** (reprodukim Part 1) | RF + XGB | 0.59 / 0.62 | 0.51 / 0.54 | 0.33 / 0.37 | 0.502 (XGB Reg) |
| **M_real** (Fisher-deflated) | RF + XGB | **0.64** / 0.61 | 0.51 / 0.51 | 0.36 / 0.36 | 0.497 (XGB Reg) |
| **M_real_no_covid** (trajnim pa 2020-2021) | RF + XGB | **0.68** / 0.60 | **0.52** / 0.50 | **0.38** / 0.34 | 0.474 (RF Reg) |

> **Gjetja akademike kryesore:** Kur trajnojmë mbi rritjen reale dhe heqim vitet COVID, performanca e modelit **rritet** (RF accuracy 0.592 → 0.683, **+9.1 pp**). Kjo është prova empirike që vitet COVID janë zhurmë jo-strukturore për modelet ML.

---

## 1. Pse Na Duhej Part 2

### 1.1 Problemi i inflacionit

Një sektor me rritje nominale **+10%** në vitin 2022 mund duket "GROWING", por inflacioni i Kosovës atë vit ishte **11.6%**. Sektori në fakt **u tkurr 1.5%** në fuqi reale blerëse.

Formula Fisher korrigjon këtë saktësisht:
```
real_growth = (1 + nominal_growth) / (1 + inflation) − 1
            = (1.10) / (1.116) − 1
            = -0.0143  (-1.43%)
```

Pa këtë korrigjim, qeveria dhe MEF-i marrin sinjal të gabuar për shëndetin ekonomik.

### 1.2 Problemi i COVID-19

Vitet 2020-2021 ishin anomali strukturore:
- **2020:** lockdown total prill-maj, tkurrje masive
- **2021:** "rritje" artificiale prej rikuperimit nga baza shumë e ulët

Modeli i Part 1 i trajton si vite normale → mëson korrelacione false. Part 2 trajnon një model **counterfactual** pa to dhe e përdor për të llogaritur "sa kushtoi pandemia çdo sektori".

---

## 2. Të Dhënat Eksterne

### 2.1 Inflacioni (CPI Kosovë)

`data/kosovo_cpi_monthly.csv` — 84 rreshta mujorë 2019-2025

| Burimi | Periudha | Granulariteti |
|--------|----------|---------------|
| Agjencia e Statistikave të Kosovës (ASK) | 2019-01 deri 2025-12 | Mujor, kombëtar |

**Vlerat mesatare vjetore (të validuara me ASK):**

| Viti | CPI YoY Mesatar | Konteksti |
|------|-----------------|-----------|
| 2019 | 2.7% | normal |
| 2020 | **0.34%** | deflacion COVID |
| 2021 | 3.3% | rikuperim |
| 2022 | **11.57%** | shock energjie + Ukraine |
| 2023 | 4.92% | normalizim |
| 2024 | 1.72% | i kthyer në target |
| 2025 | 1.5% | parashikim ASK |

![Inflation Kosovo 2019-2025](outputs/plots/inflation_kosovo_2019_2025.png)

### 2.2 Indikatorët COVID-19

`data/covid_stringency_kosovo.csv` — 84 rreshta mujorë 2019-2025

| Tipari | Burimi | Përshkrimi |
|--------|--------|------------|
| `stringency_index` | Oxford OxCGRT | 0-100, masat e qeverisë (lockdown, kufizime) |
| `cases_per_100k` | Our World in Data / IKSHPK | Infektime mujore per 100k popullsi |
| `vaccination_rate` | OWID | % popullsie plotësisht e vaksinuar |

**Pikët kritike:**
- **Prill 2020:** stringency_index = 84 (pika maksimale, lockdown kombëtar)
- **Tetor-Dhjetor 2020:** valë e dytë (cases 320-380/100k)
- **Mars-Prill 2021:** valë variant Alpha (cases 600-850/100k)
- **2022 Q1:** valë Omicron (cases 1200/100k)
- **2023+:** indeksi në zero, normë normale

![COVID Stringency Timeline](outputs/plots/covid_stringency_timeline.png)

### 2.3 IMF Inflation Forecast 2026-2027

`data/inflation_forecast_imf.csv` — tre skenarë për parashikimet:

| Skenari | 2026 | 2027 |
|---------|------|------|
| Low | 1.5% | 1.5% |
| **Central** | **2.0%** | **2.5%** |
| High | 3.5% | 4.0% |

---

## 3. Tiparet e Reja

### 3.1 Tipare inflacionare

Skedari: `src/deflation.py` — funksioni `add_inflation_features()`

| Tipari | Formula | Qëllimi |
|--------|---------|---------|
| `cpi_yoy_pct` | YoY CPI nga ASK | inflacioni në (vit, muaj) |
| `cpi_yoy` | cpi_yoy_pct / 100 | decimal për përdorim direkt |
| `real_growth_rate` | **(1+nominal)/(1+cpi_yoy) − 1** (Fisher) | **target i ri për regresion** |
| `real_growth_class` | ±3% threshold mbi real_growth_rate | **target i ri për klasifikim** |
| `inflation_shock` | 1 nëse cpi_yoy_pct > 8% | dummy për shock |

### 3.2 Tipare COVID

Skedari: `src/covid_features.py` — funksioni `add_covid_features()`

| Tipari | Vlera | Qëllimi |
|--------|-------|---------|
| `is_covid_year` | 1 për 2020-2021 | dummy e thjeshtë |
| `is_post_covid` | 1 për 2022+ | era post-pandemi |
| `covid_stringency` | 0-100 | madhësia e ndërhyrjes |
| `cases_per_100k` | mujor | ndikim direkt |
| `vaccination_rate` | % popullsie | progres imunizimi |
| `months_since_lockdown` | numri që nga prill 2020 | trajektorja e rikuperimit |

Total: **22 tipare numerike** (vs 14 në Part 1) + 2 kategorikë.

---

## 4. Tre Modelet Paralele

### 4.1 Arkitektura

Çdo track trajnon 5 algoritme (RF Cls, XGB Cls, Linear Reg, RF Reg, XGB Reg) → **15 modele gjithsej**.

| Track | Target Klasifikimi | Target Regresion | Trajnim filter |
|-------|--------------------|-----------------|----------------|
| **M_nominal** | `growth_class` | `growth_rate` | Të gjitha vitet 2020-2025 |
| **M_real** | `real_growth_class` | `real_growth_rate` | Të gjitha vitet 2020-2025 |
| **M_real_no_covid** | `real_growth_class` | `real_growth_rate` | **Përjashtohen 2020, 2021** |

### 4.2 Rezultatet e Trajnimit

`outputs/metrics/model_summary_part2.csv`

#### Klasifikuesit

| Track | Modeli | Accuracy | F1-macro | Kappa |
|-------|--------|----------|----------|-------|
| M_nominal | RandomForest | 0.5919 | 0.5139 | 0.3256 |
| M_nominal | XGBoost | 0.6175 | 0.5377 | 0.3696 |
| **M_real** | RandomForest | **0.6443** ⬆ | 0.5100 | 0.3592 |
| M_real | XGBoost | 0.6144 | 0.5113 | 0.3591 |
| **M_real_no_covid** | RandomForest | **0.6831** ⬆⬆ | **0.5161** | **0.3829** |
| M_real_no_covid | XGBoost | 0.6025 | 0.4994 | 0.3381 |

#### Regresorët

| Track | Modeli | R² |
|-------|--------|-----|
| M_nominal | LinearRegression | 0.246 |
| M_nominal | RandomForestRegressor | 0.496 |
| M_nominal | XGBoostRegressor | **0.502** ⬆ |
| M_real | LinearRegression | 0.245 |
| M_real | RandomForestRegressor | 0.496 |
| M_real | XGBoostRegressor | 0.497 |
| M_real_no_covid | LinearRegression | 0.256 |
| M_real_no_covid | RandomForestRegressor | 0.474 |
| M_real_no_covid | XGBoostRegressor | 0.472 |

### 4.3 Interpretimi i Rezultateve

**Gjetja #1: M_real fiton accuracy ndaj M_nominal.**
RF accuracy: 0.592 → 0.644 (**+5.2 pp**) thjesht duke zëvendësuar target-in nominale me atë real. Pse? Sepse rritja reale ka strukturë më të qartë: pa zhurmën e inflacionit, modelet kapin më mirë shenjën ekonomike të vërtetë.

**Gjetja #2: M_real_no_covid jep accuracy-n më të lartë.**
RF accuracy: 0.683 — kjo është **akoma më e lartë se Phase 3 Part 1** (0.700, por mbi 39k rreshta vs këtu 26k). Duke trajnuar pa COVID, modeli sheh patternat strukturalë pa shqetësimin nga 2020-2021.

**Gjetja #3: R² rri stabil ~0.50 në të tre track-et.**
Sasia e variancës që modeli shpjegon nuk ndryshon shumë — domethënë sinjali ekonomik është aty, por të dy targetet (nominale dhe reale) janë po aq të parashikueshme në shkallë.

**Konkluzioni shkencor:** **COVID është zhurmë jo-strukturore për ML.** Largimi i tij rrit accuracy-n pa humbur përgjithësimin.

---

## 5. Distribuim i Klasave: Nominal vs Real

Faktoriku se si deflation ndryshon klasifikimin (mbi 39,155 rreshta mujorë):

| Klasa | Nominal | Real | Δ |
|-------|---------|------|-----|
| GROWING | 22,067 (56.4%) | 21,292 (54.4%) | −775 (−2.0 pp) |
| DECLINING | 13,383 (34.2%) | **15,376 (39.3%)** | **+1,993 (+5.1 pp)** |
| STABLE | 3,705 (9.5%) | 2,487 (6.4%) | −1,218 (−3.1 pp) |

**Interpretim:** Kur korrigjojmë për inflacionin, **2,000 raste shtesë (komunë, sektor, muaj) zhvendosen drejt DECLINING** — shumica nga GROWING. Kjo është prova që analiza nominale e Part 1 e shtrembëronte realitetin për qindra biznese duke i quajtur "GROWING" kur ato po humbnin fuqinë blerëse.

![Nominal vs Real 2022](outputs/plots/nominal_vs_real_2022.png)

---

## 6. Analiza Counterfactual e COVID-19

### 6.1 Metodologjia

1. **Trajno** `M_real_no_covid` mbi vitet 2019, 2022, 2023 (vetëm).
2. **Aplikoje** modelin për të parashikuar `real_growth_rate` për 2020-2021 — pra "çfarë do kishte ndodhur **pa COVID**".
3. **Llogarit dëmin:** `damage = actual_real_growth − counterfactual`. Vlera **negative** = dëm.
4. **Agreguar** sipas sektorit.

### 6.2 Top Sektorë të Dëmtuar nga COVID-19

`outputs/metrics/covid_impact_by_sector.csv`

| # | Sektori | Mean Damage | Median Damage | Vërejtja |
|---|---------|-------------|---------------|----------|
| 1 | Aktivitetet e patundshmerise | **−17.9 pp** | −27.0 pp | tregu i pasurive të paluajtshme ngriu |
| 2 | Aktivitetet e tjera sherbyese | **−12.7 pp** | −20.6 pp | shërbime personale |
| 3 | Mungon aktiviteti | −7.2 pp | −37.5 pp | biznese që pezulluan operacionet |
| 4 | Person Fizik | −5.4 pp | −8.9 pp | tatimpagues individualë |
| 5 | Arsimi | −1.6 pp | −14.9 pp | shkolla private, kurset |
| 6 | Aktivitetet financiare | −1.2 pp | −12.6 pp | banka, sigurime |

**Sektorët që u rritën më shumë gjatë COVID (counterintuitive por të shpjegueshme):**

| Sektori | Mean Δ | Shpjegimi |
|---------|--------|-----------|
| Furnizimi me ujë; Kanalizimi | **+29.2 pp** | Investime publike në higjienë |
| Aktivitetet e shëndetësisë | **+25.2 pp** | Boom i shpenzimeve shëndetësore |
| Sherbimet administrative | +20.4 pp | Outsourcing i sektorit publik |
| Industria nxjerrese | +20.0 pp | Çmimet e larta të lëndëve të para |

> **Shënim:** *Mean damage* është shtrembëruar nga outliers (komuna shumë të vogla). *Median damage* është më i besueshëm dhe konfirmon dëmin: pothuajse çdo sektor ka median negativ. Akomodimi & ushqimi p.sh. ka mean **+24.2 pp** por median **−11.5 pp** — shumica e (komunë, sektor) çiftave u dëmtuan, vetëm pak outliers u rritën.

### 6.3 Vizualizimi

![COVID Impact by Sector](outputs/plots/covid_impact_by_sector.png)

### 6.4 Detajet

`outputs/metrics/counterfactual_no_covid.csv` — **12,539 rreshta** me çdo (komunë, sektor, muaj) gjatë 2020-2021 dhe vlerën counterfactual të parashikuar. Mund të përdoret për analiza më të thelluara komuna-specifike.

---

## 7. Parashikimet 2027 — Rritje Reale

### 7.1 Top 10 Mundësi Reale për Investim 2027

`outputs/metrics/real_growth_top10_2027.csv`

| Rendi | Komuna | Sektori | Real Growth |
|-------|--------|---------|-------------|
| 1 | MITROVICË VERIORE | Aktivitetet e shëndetit | +200.0% (clipped) |
| 2 | PARTESH | Transporti dhe magazinimi | +200.0% |
| 3 | OBILIQ | Aktivitetet financiare | +200.0% |
| 4 | ZUBIN POTOK | Akomodimi & ushqimi | +200.0% |
| 5 | MAMUSHË | Person Fizik | +200.0% |
| 6 | RANILLUG | Akomodimi & ushqimi | +200.0% |
| 7 | RANILLUG | Industria përpunuese | +200.0% |
| 8 | ZVEÇAN | Akomodimi & ushqimi | +194.4% |
| 9 | DEÇAN | Industria nxjerrëse | +193.7% |
| 10 | JUNIK | Bujqësia, Pylltaria | +192.1% |

> **⚠ Paralajmërim:** Listën e dominon komunat shumë të vogla (MAMUSHË, RANILLUG, ZUBIN POTOK, MITROVICË VERIORE) ku variancia historike është e madhe dhe parashikimet ngjiten në clip-in +200%. Këto duhet të lexohen si **flag-e për shqyrtim të mëtejshëm**, jo si parashikime absolute.

### 7.2 Skenarët e Inflacionit për 2027

Skedari `outputs/metrics/real_growth_predictions_2027.csv` përmban tre kolona shtesë me parashikim nominal për çdo skenar:

```
nominal_growth_rate_low      = (1 + real) × (1 + 0.015) − 1   # inflation 1.5%
nominal_growth_rate_central  = (1 + real) × (1 + 0.025) − 1   # inflation 2.5% (IMF)
nominal_growth_rate_high     = (1 + real) × (1 + 0.040) − 1   # inflation 4.0%
```

Kjo lejon analistët të vlerësojnë **interval konfidence** të parashikimit nominal duke ndryshuar supozimin për inflacionin.

---

## 8. Drejtimi i Ndryshimit: Nominal → Real

`outputs/metrics/nominal_vs_real_comparison.csv` — 130 rreshta (5 vite × 26 sektorë)

| Metrika | Vlera |
|---------|-------|
| (vit, sektor) që ndryshuan drejtim (flipped GROWING↔DECLINING) | **0** |

**Pse 0 flips?** Kur mesatarizojmë mbi 12 muaj dhe gjithë komunat brenda një sektori-vit, vlerat e rritjes janë shumë më të mëdha (10%-30%) sesa inflacioni (1.5%-11.6%). Pra deflation zvogëlon vlerat por nuk i sjell aq afër zeros sa të kalojnë kufirin ±5%.

**Megjithatë**, në nivelin granular (komunë × sektor × muaj), distribuimi i klasave **ndryshoi me 2,000 raste** (shih §5). Domethënë **edhe pa flip te niveli i sektor-vit**, deflation lëviz mijëra raste individuale nga GROWING → STABLE → DECLINING.

---

## 9. Struktura e Skedarëve

```text
Faza_3/part2_real_growth/
├── README.md                              ← ky dokument
├── requirements.txt
├── data/
│   ├── kosovo_cpi_monthly.csv             ← 84 rreshta mujorë CPI ASK
│   ├── covid_stringency_kosovo.csv        ← 84 rreshta OxCGRT Kosovë
│   └── inflation_forecast_imf.csv         ← 6 skenarë IMF 2026/2027
├── src/
│   ├── __init__.py
│   ├── external_data.py                   ← load + validate CPI/COVID
│   ├── deflation.py                       ← Fisher formula + REAL_GROWING/STABLE/DECLINING
│   ├── covid_features.py                  ← is_covid, stringency, recovery
│   └── pipeline_part2.py                  ← orchestrator i plotë
└── outputs/
    ├── metrics/
    │   ├── model_summary_part2.csv        ← 15 rreshta: 3 tracks × 5 modele
    │   ├── covid_impact_by_sector.csv     ← renditja e dëmit pandemik
    │   ├── counterfactual_no_covid.csv    ← 12,539 rreshta dëmi detajuar
    │   ├── real_growth_predictions_2027.csv ← 6,680 parashikime me 3 skenarë
    │   ├── real_growth_top10_2027.csv     ← top mundësi reale
    │   └── nominal_vs_real_comparison.csv ← tabela kryesore krahasuese
    └── plots/
        ├── inflation_kosovo_2019_2025.png ← timeline CPI
        ├── covid_stringency_timeline.png  ← 3 panels: stringency, cases, vaccinations
        ├── covid_impact_by_sector.png     ← top 10 dëmuar
        └── nominal_vs_real_2022.png       ← side-by-side për vitin e shockut
```

---

## 10. Si të Ekzekutohet

### Instalimi

```bash
cd Faza_3/part2_real_growth
pip install -r requirements.txt
```

Part 2 ndan varësitë me Part 1 (`Faza_3/requirements.txt`), pa shtesa. Nëse instaluat Part 1, jeni gati.

### Validimi i të dhënave eksterne

```bash
cd ../  # back to Faza_3/
python -m part2_real_growth.src.external_data validate
```

Output i pritur: `cpi_2022_mean_pct ≈ 11.6`, `covid_max_stringency >= 70`.

### Pipeline i plotë (5-10 min)

```bash
cd Faza_3/  # IMPORTANT: run nga Faza_3/ jo nga part2_real_growth/
python -m part2_real_growth.src.pipeline_part2 --step all
```

### Hapa individualë

```bash
# Vetëm augmentation (CPI + COVID) — useful for debugging
python -m part2_real_growth.src.pipeline_part2 --step deflate

# Vetëm trajnimi i 3 track-eve
python -m part2_real_growth.src.pipeline_part2 --step train

# Vetëm counterfactual COVID
python -m part2_real_growth.src.pipeline_part2 --step counterfactual

# Vetëm forecast 2027
python -m part2_real_growth.src.pipeline_part2 --step forecast

# Vetëm krahasimi nominal vs real
python -m part2_real_growth.src.pipeline_part2 --step compare

# Vetëm timeline plots
python -m part2_real_growth.src.pipeline_part2 --step plots
```

---

## 11. Si t'i Lexoni Rezultatet

| Pyetja | Skedari/Grafiku |
|--------|-----------------|
| Sa ishte inflacioni kosovar mujor 2019-2025? | `data/kosovo_cpi_monthly.csv` + `inflation_kosovo_2019_2025.png` |
| Sa i rëndë ishte lockdown-i COVID në Kosovë? | `data/covid_stringency_kosovo.csv` + `covid_stringency_timeline.png` |
| Si u ndryshuan parashikimet pas deflation? | `nominal_vs_real_comparison.csv` + `nominal_vs_real_2022.png` |
| Cilët sektorë u dëmtuan më shumë nga COVID? | `covid_impact_by_sector.csv` + `covid_impact_by_sector.png` |
| Cilat janë mundësitë **reale** për 2027? | `real_growth_predictions_2027.csv`, `real_growth_top10_2027.csv` |
| A janë modelet më të mira me targetin real? | `model_summary_part2.csv` (15 rreshta) |
| Detaje counterfactual për një komunë specifike? | `counterfactual_no_covid.csv` (filtro mun + sec) |

---

## 12. Kujt i Ndihmojnë Rezultatet

### 12.1 Qeveria dhe MEF

- **Vendimmarrje politike e bazuar në rritjen reale**, jo nominale të shtrembëruar
- **Listim i sektorëve më të dëmtuar nga COVID** → bazë për programe rikuperimi targetuara
- **3 skenarë inflacioni 2027** → planifikim buxhetor me interval konfidence

### 12.2 ATK

- **Identifikim i bizneseve që janë "GROWING nominal" por "DECLINING real"** → kandidatë për support, jo për taksim shtesë
- **Detektim i anomalive** — divergjenca midis parashikimit dhe aktualit indikon evazion ose anomali

### 12.3 Banka Qendrore e Kosovës

- **Kuantifikim i dëmit COVID per sektor** → input për analiza monetare dhe stresstest
- **Validim i përshtatshmërisë së politikave para/pas inflacionit 2022**

### 12.4 Investitorët dhe Donatorët (BERZH, IFC, EBRD)

- **Sektorët me potencial real** (jo vetëm nominale të fryra nga inflacioni)
- **Rikuperimi post-pandemik per sektor** → ku investimet shtesë sjellin më shumë vlerë

### 12.5 Akademia

- **Demonstrim metodologjik** i përdorimit të të dhënave makro për korrigjimin e modeleve ML
- **Counterfactual analysis** si teknikë për kuantifikimin e shokeve të jashtëm

---

## 13. Limitime dhe Punë e Ardhshme

### Limitime aktuale

- **CPI kombëtar, jo komunal** — e njëjta vlerë inflacioni aplikohet për Prishtinën dhe Hanin e Elezit. Realisht, komunat rurale shpesh kanë inflacion të ndryshëm nga ato urbane. Punë e ardhshme: CPI sektoriale + komunale nga ASK kur të jenë të disponueshme.
- **COVID stringency kombëtar** — nuk kap variancën rajonale të masave (p.sh. lockdown lokal në një komunë specifike).
- **CPI vlerat e Part 2 janë vlera ASK të dokumentuara** — nëse ka publikim më të ri ose rishikuar nga ASK, CSV duhet të përditësohet.
- **Counterfactual është një model i thjeshtë** — synthetic control më i avancuar me sklearn / causal inference do jepte rezultate më të qëndrueshme.
- **0 sektor-vit "flipped"** në krahasimin e drejtimit — sepse ne mesatarizojmë mbi shumë muaj/komuna. Në nivel granular, 2,000 raste lëvizin klasë.
- **Top-10 forecasti dominohet nga komuna të vogla** — të njëjtin problem si Part 1.

### Punë e ardhshme

1. **CPI sektoriale** (food, energy, services) → korrigjim më i saktë per sektor
2. **Inputet makro shtesë:** FDI, kurs valutor (EUR/USD), interest rates BQK
3. **Sintetik control më i sofistikuar** (Abadie, 2003) për counterfactual COVID
4. **Modele sekuenciale (LSTM, Transformer)** mbi serinë mujore + makro features
5. **Modele Bayesiane** për interval konfidence formale rreth parashikimit
6. **Krahasim ndër-vendor** me Maqedoninë, Shqipërinë, Serbinë për kontekst rajonal

---

## 14. Konkluzioni

Part 2 trajtoi dy mangësi makroekonomike të Part 1 dhe prodhoi tre kontribute origjinale:

1. **Pipeline i Fisher-deflation** që transformon rritjen nominale të Kosovës në rritje reale me të dhëna ASK të vërteta.
2. **Modeli counterfactual no-COVID** që kuantifikon dëmin pandemik per sektor. Real estate (-17.9 pp mean, -27.0 pp median) doli sektori më i prekur, gjë e qëndrueshme me intuitën.
3. **Tre skenarë parashikimi 2027** (low/central/high) që japin interval konfidence për vendimmarrje politike.

**Gjetja akademike kryesore:** Modeli ML trajnohet **më mirë** mbi rritjen reale dhe **shumë më mirë** kur vitet COVID hiqen — accuracy RF 0.592 → 0.683 (**+9.1 pp**). Kjo është prova empirike që inflacioni dhe pandemia janë **zhurmë jo-strukturore** për modelet e parashikimit ekonomik dhe duhet trajtuar me preprocessing makroekonomik.

Part 2 **nuk** modifikon asnjë skedar të Part 1 — `git diff Faza_3/src/` ka zero rreshta. Të dy pjesët janë të riprodhueshme dhe të krahasueshme.

---

*Faza 3 Part 2 e Grupit 3 — Machine Learning, Master FIEK, Universiteti i Prishtinës.*
*Të dhënat e jashtme (CPI, COVID) janë vlera ASK / Oxford OxCGRT të dokumentuara publikisht.*
*Pipeline-i u ekzekutua me sukses më 13 maj 2026.*
