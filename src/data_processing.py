from pathlib import Path
import pandas as pd
import numpy as np

# Dynamické určenie koreňového adresára projektu a priečinka Data
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

def process_scouting_data():
    # 1. Načítanie surových dát
    input_file = DATA_DIR / "All_Leagues.csv"
    data = pd.read_csv(input_file, sep=';', low_memory=False)
    
    # Zabezpečenie štatistickej významnosti (min. 900 minút)
    representative_data = data[data["Minutes played"] >= 900]
    representative_data.to_csv(DATA_DIR / "Representative_players.csv", sep=';', index=False)

    # Načítanie dát s doplneným vekom (ak existuje, inak fallback na vyfiltrované)
    added_age_file = DATA_DIR / "Representative_players_added_age.csv"
    if added_age_file.exists():
        data_with_age = pd.read_csv(added_age_file, sep=';', low_memory=False)
    else:
        data_with_age = representative_data

    # 2. Filtrácia cieľovej vzorky (U23)
    young_players = data_with_age[data_with_age["Age"] <= 23].copy()

    # 3. Odstránenie pozície brankára a špecifických brankárskych metrík
    gk_cols_to_drop = [
        'Aerial duels per 90', 'Clean sheets', 'Exits per 90', 
        'Conceded goals per 90', 'Shots against per 90', 'Prevented goals', 
        'Prevented goals per 90', 'Conceded goals', 'Shots against',
        'Save rate, %', 'xG against', 'xG against per 90' 
    ]
    young_players = young_players.drop(columns=[c for c in gk_cols_to_drop if c in young_players.columns])

    # 4. Technická normalizácia a oprava dátových typov
    text_cols = ['Player', 'Team', 'Team within selected timeframe', 'Position', 'Birth country', 
                 'Passport country', 'Foot', 'On loan', 'Season', 'Main Position', 'Primary Position', 
                 'Date', 'Contract expires']

    for col in young_players.columns:
        if col not in text_cols and young_players[col].dtype == 'object':
            young_players[col] = young_players[col].astype(str).str.replace(',', '.')
            young_players[col] = pd.to_numeric(young_players[col], errors='coerce')

    physical_cols = ['Height', 'Weight', 'Market value']
    for col in physical_cols:
        if col in young_players.columns:
            young_players[col] = young_players[col].replace(np.nan, 0)

    # Štandardizácia názvov stĺpcov
    young_players.columns = [
        c.replace(' ', '_').replace(',', '').replace('%', 'pct').replace('/', 'per').replace('.', '') 
        for c in young_players.columns
    ]
    
    # Export vyčisteného datasetu
    output_file = DATA_DIR / "Cleaned_young_representative_players.csv"
    young_players.to_csv(output_file, sep=';', index=False)
    print(f"✅ Hotovo: {output_file}")

if __name__ == "__main__":
    process_scouting_data()