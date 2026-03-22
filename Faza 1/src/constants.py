RAW_TO_CANONICAL = {
    "Viti\nGodina\nYear": "year",
    "Muaji\nMesec\nMonth": "month",
    "Përshkrimi i Sektorit (Kryesor)\nOpis (Glavnog) Sektora\nDescription of (Primary) Sector": "primary_sector",
    "Komuna\nOpština\nMunicipality ": "municipality",
    "Statusi i Regjistrimit\nStatus Registracije\nRegistration Status": "registration_status",
    "Numri i Tatimpaguesve\nBroj Poreskih Obveznika \nNumber of Taxpayers": "num_taxpayers",
    "Qarkullimi në Euro\nPromet u evrima\nTurnover in Euro ": "turnover_eur",
}

IGNORED_RAW_COLUMNS = ["Unnamed: 0", "Unnamed: 4", "Unnamed: 5", "Unnamed: 8"]
NUMERIC_COLUMNS = ["year", "month", "num_taxpayers", "turnover_eur"]
CATEGORICAL_COLUMNS = ["primary_sector", "municipality", "registration_status"]
BUSINESS_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
CANDIDATE_TARGET_COLUMNS = ["registration_status", "municipality", "primary_sector"]
