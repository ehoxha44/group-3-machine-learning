# Qarkullimi Data Cleaning Report

## Data Types
| column | dtype | semantic_type | null_count | null_rate_decimal | null_rate_percent | unique_count | sample_values |
| --- | --- | --- | --- | --- | --- | --- | --- |
| year | float64 | numeric | 0 | 0.0 | 0.000000% | 7 | 2025.0 | 2025.0 | 2025.0 |
| month | float64 | numeric | 0 | 0.0 | 0.000000% | 12 | 1.0 | 1.0 | 1.0 |
| num_taxpayers | float64 | numeric | 0 | 0.0 | 0.000000% | 477 | 2.0 | 2.0 | 1.0 |
| turnover_eur | float64 | numeric | 0 | 0.0 | 0.000000% | 442973 | 0.0 | 13436.77 | 0.0 |
| primary_sector | object | categorical | 0 | 0.0 | 0.000000% | 23 | Bujqesia;Pylltaria dhe Peshkimi | Bujqesia;Pylltaria dhe Peshkimi | Bujqesia;Pylltaria dhe Peshkimi |
| municipality | object | categorical | 0 | 0.0 | 0.000000% | 38 | KLINË | PODUJEVË | MITROVICË VERIORE |
| registration_status | object | categorical | 0 | 0.0 | 0.000000% | 30 | SH.P.K. | SH.P.K. | SH.P.K. |

## Quality Status
| metric | value |
| --- | --- |
| total_rows | 650913 |
| total_columns | 7 |
| complete_rows | 650913 |
| rows_with_any_null | 0 |
| duplicate_rows | 0 |
| invalid_year_rows | 0 |
| invalid_month_rows | 0 |
| invalid_num_taxpayers_rows | 0 |
| invalid_turnover_rows | 176 |

Readable overall state:
Cleaned dataset is usable after cleaning; blocked fields were not detected.

## Complete vs Null Data
| column | null_count | null_rate_decimal | null_rate_percent | non_null_count |
| --- | --- | --- | --- | --- |
| year | 0 | 0.0 | 0.000000% | 650913 |
| month | 0 | 0.0 | 0.000000% | 650913 |
| num_taxpayers | 0 | 0.0 | 0.000000% | 650913 |
| turnover_eur | 0 | 0.0 | 0.000000% | 650913 |
| primary_sector | 0 | 0.0 | 0.000000% | 650913 |
| municipality | 0 | 0.0 | 0.000000% | 650913 |
| registration_status | 0 | 0.0 | 0.000000% | 650913 |

## Missing Value Strategy
| column_role | strategy |
| --- | --- |
| numeric | preserve nulls in strict dataset, median-impute only in model-ready dataset |
| categorical | preserve nulls in strict dataset, fill with Unknown only in model-ready dataset |
| row_filtering | drop only rows with all critical business fields missing or repeated header rows |

## Outlier Findings
| metric | count |
| --- | --- |
| turnover_eur_outlier_iqr | 96473 |
| turnover_eur_outlier_zscore | 5455 |
| num_taxpayers_outlier_iqr | 79451 |
| num_taxpayers_outlier_zscore | 8101 |
| turnover_eur_outlier_log_iqr | 104535 |

## Class Distribution Summary
| target_column | class_label | count | share_decimal | share_percent |
| --- | --- | --- | --- | --- |
| registration_status | SH.P.K. | 321710 | 0.49424424 | 49.4244% |
| registration_status | INDIVIDUAL | 270904 | 0.4161908 | 41.6191% |
| registration_status | ORTAKËRIA E PËRGJ. | 17491 | 0.02687149 | 2.6871% |
| registration_status | SHOQËRI AKCIONARE | 11642 | 0.01788565 | 1.7886% |
| registration_status | KOMPANI E HUAJ | 11218 | 0.01723425 | 1.7234% |
| registration_status | NDËRMARRJE SHOQ. | 4871 | 0.00748333 | 0.7483% |
| registration_status | OJQ | 3729 | 0.00572888 | 0.5729% |
| registration_status | PERSON FIZIK | 2579 | 0.00396213 | 0.3962% |
| registration_status | KONSORCIUM | 984 | 0.00151172 | 0.1512% |
| registration_status | TJETER | 826 | 0.00126899 | 0.1269% |
| registration_status | KOOPERATIVA BUJQ. | 782 | 0.00120139 | 0.1201% |
| registration_status | NDËRMARRJE NËN MENAXHIM TË AKP-SË | 711 | 0.00109231 | 0.1092% |
| registration_status | PËRFAQËSUES FISKAL | 697 | 0.0010708 | 0.1071% |
| registration_status | NDËRMARRJE PUB. | 654 | 0.00100474 | 0.1005% |
| registration_status | PROJEKT | 497 | 0.00076354 | 0.0764% |
| registration_status | DEGA E SHOQËRISË SË HUAJ | 276 | 0.00042402 | 0.0424% |
| registration_status | KOOPERATIVA | 260 | 0.00039944 | 0.0399% |
| registration_status | ORGANIZATA BUXHETORE | 222 | 0.00034106 | 0.0341% |
| registration_status | SHOQËRI KOMANDITE | 157 | 0.0002412 | 0.0241% |
| registration_status | ORTAKËRI E KUFIZUAR | 148 | 0.00022737 | 0.0227% |
| registration_status | BASHKËSI FETARE | 107 | 0.00016438 | 0.0164% |
| registration_status | KOMP .PUB. E KUF. | 84 | 0.00012905 | 0.0129% |
| registration_status | KOMPANI SIGURIMI | 84 | 0.00012905 | 0.0129% |
| registration_status | AGJENSION I HUAJ QEVERITAR | 83 | 0.00012751 | 0.0128% |
| registration_status | ZYRË NDËRLIDHËSE | 83 | 0.00012751 | 0.0128% |
| registration_status | ZYRA E PËRFAQËSISË NË KOSOVË | 74 | 0.00011369 | 0.0114% |
| registration_status | SHOQATË | 16 | 2.458e-05 | 0.0025% |
| registration_status | PARTI POLITIKE | 11 | 1.69e-05 | 0.0017% |
| registration_status | SINDIKATAT | 10 | 1.536e-05 | 0.0015% |
| registration_status | AMBASADA | 3 | 4.61e-06 | 0.0005% |
| municipality | PRISHTINË | 74143 | 0.11390616 | 11.3906% |
| municipality | PRIZREN | 43438 | 0.06673396 | 6.6734% |
| municipality | FERIZAJ | 41708 | 0.06407615 | 6.4076% |
| municipality | PEJË | 34826 | 0.05350331 | 5.3503% |
| municipality | GJAKOVË | 32560 | 0.05002205 | 5.0022% |
| municipality | GJILAN | 32541 | 0.04999286 | 4.9993% |
| municipality | FUSHË KOSOVË | 28284 | 0.04345281 | 4.3453% |
| municipality | MITROVICË | 27066 | 0.04158159 | 4.1582% |
| municipality | SUHAREKË | 22829 | 0.03507228 | 3.5072% |
| municipality | PODUJEVË | 22592 | 0.03470817 | 3.4708% |
| municipality | VUSHTRRI | 21312 | 0.0327417 | 3.2742% |
| municipality | LIPJAN | 21120 | 0.03244673 | 3.2447% |
| municipality | RAHOVEC | 19600 | 0.03011155 | 3.0112% |
| municipality | GLLOGOC | 19121 | 0.02937566 | 2.9376% |
| municipality | GRAÇANICË | 18460 | 0.02836016 | 2.8360% |
| municipality | VITI | 17114 | 0.0262923 | 2.6292% |
| municipality | ISTOG | 16538 | 0.02540739 | 2.5407% |
| municipality | MALISHEVË | 16116 | 0.02475907 | 2.4759% |
| municipality | KLINË | 15725 | 0.02415837 | 2.4158% |
| municipality | DEÇAN | 13856 | 0.02128702 | 2.1287% |
| municipality | OBILIQ | 13643 | 0.02095979 | 2.0960% |
| municipality | SKENDERAJ | 13419 | 0.02061566 | 2.0616% |
| municipality | KAÇANIK | 12803 | 0.0196693 | 1.9669% |
| municipality | SHTIME | 12464 | 0.01914849 | 1.9148% |
| municipality | KAMENICË | 11966 | 0.01838341 | 1.8383% |
| municipality | DRAGASH | 7806 | 0.01199239 | 1.1992% |
| municipality | HANI I ELEZIT | 5733 | 0.00880763 | 0.8808% |
| municipality | MITROVICË VERIORE | 5528 | 0.00849269 | 0.8493% |
| municipality | LEPOSAVIQ | 4782 | 0.0073466 | 0.7347% |
| municipality | SHTËRPCË | 4690 | 0.00720526 | 0.7205% |
| municipality | MAMUSHË | 3556 | 0.0054631 | 0.5463% |
| municipality | ZVEÇAN | 3009 | 0.00462274 | 0.4623% |
| municipality | NOVOBËRDË | 2585 | 0.00397134 | 0.3971% |
| municipality | KLLOKOT | 2578 | 0.00396059 | 0.3961% |
| municipality | JUNIK | 2373 | 0.00364565 | 0.3646% |
| municipality | ZUBIN POTOK | 2329 | 0.00357805 | 0.3578% |
| municipality | PARTESH | 2105 | 0.00323392 | 0.3234% |
| municipality | RANILLUG | 595 | 0.0009141 | 0.0914% |
| primary_sector | Tregtia me shumice dhe pakice; Riparimi i mjeteve motorike dhe motoeikletave | 186172 | 0.28601672 | 28.6017% |
| primary_sector | Industria perpunuese | 141299 | 0.21707817 | 21.7078% |
| primary_sector | Ndertimtaria | 49924 | 0.07669842 | 7.6698% |
| primary_sector | Aktivitetet profesionale, shkencore dhe teknike | 35033 | 0.05382132 | 5.3821% |
| primary_sector | Bujqesia;Pylltaria dhe Peshkimi | 29130 | 0.04475252 | 4.4753% |
| primary_sector | Informimi dhe komunikimi | 28094 | 0.04316091 | 4.3161% |
| primary_sector | Sherbimet administrative dhe mbeshtetese | 26146 | 0.04016819 | 4.0168% |
| primary_sector | Transporti dhe magazinimi | 24500 | 0.03763944 | 3.7639% |
| primary_sector | Akomodimi dhe sherbimi ushqimor | 24056 | 0.03695732 | 3.6957% |
| primary_sector | Aktivitetet e tjera sherbyese | 18869 | 0.02898851 | 2.8989% |
| primary_sector | Aktivitetet e shendetit te njeriut dhe te punes sociale | 16393 | 0.02518463 | 2.5185% |
| primary_sector | Artet, Argetimi dhe rekreacioni | 13078 | 0.02009178 | 2.0092% |
| primary_sector | Arsimi | 10461 | 0.01607127 | 1.6071% |
| primary_sector | Aktivitetet financiare dhe te sigurimit | 10189 | 0.0156534 | 1.5653% |
| primary_sector | Furnizimi me uje; Kanalizimi; Aktivitetet e menaxhimit dhe te trajtimit te mbeturinave | 8505 | 0.01306626 | 1.3066% |
| primary_sector | Industria nxjerrese | 7885 | 0.01211375 | 1.2114% |
| primary_sector | Aktivitetet e patundshmerise | 6298 | 0.00967564 | 0.9676% |
| primary_sector | Administrimi publik dhe mbrojtja; Sigurimi social i detyrueshem | 4969 | 0.00763389 | 0.7634% |
| primary_sector | Furnizimi me rryme, gaz, avull dhe ajer te kondicionuar | 4313 | 0.00662608 | 0.6626% |
| primary_sector | Mungon aktiviteti | 2895 | 0.0044476 | 0.4448% |
| primary_sector | Person Fizik | 2579 | 0.00396213 | 0.3962% |
| primary_sector | Aktivitetet e ekonomive familjare si punedhenes; Mallrat dhe sherbimet e padiferencuara, Aktivitetet e ekonomive familjare per perdorim vetanak | 95 | 0.00014595 | 0.0146% |
| primary_sector | Aktivitetet e trupave dhe organizatave nderkombetare | 30 | 4.609e-05 | 0.0046% |

## Readiness for Modeling
The strict cleaned dataset preserves nulls and flags invalid rows. The model-ready dataset imputes numeric columns with medians, fills categorical columns with Unknown, and adds imputation flags plus derived features such as year_month and turnover_eur_log1p.

## Generated Graphs
- turnover_by_year.png
- turnover_by_month.png
- turnover_by_municipality_top20.png
- turnover_by_sector_top15.png
- nulls_by_column.png
- outlier_summary.png
- registration_status_distribution_top15.png
- smote_before_top15_counts.png
- smote_before_top15_percent.png
- smote_after_top15_counts.png
- smote_after_top15_percent.png