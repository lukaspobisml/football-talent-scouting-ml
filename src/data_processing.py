import pandas as pd
import numpy as np

def process_scouting_data():
    # Načítanie surových dát zo všetkých líg
    data = pd.read_csv('All_Leagues.csv', sep=';', low_memory=False)
    
    # 1. Zabezpečenie štatistickej významnosti (min. 900 minút podľa metodiky pre mladých hráčov)
    representative_data = data[data["Minutes played"] >= 900]
    representative_data.to_csv('Representative_players.csv', sep=';', index=False)

    # Následné načítanie dát po manuálnom doplnení chýbajúceho veku z externých zdrojov
    data_with_age = pd.read_csv('Representative_players_added_age.csv', sep=';', low_memory=False)

    # 2. Filtrácia cieľovej vzorky (hráči U23 a mladší)
    young_players = data_with_age[data_with_age["Age"] <= 23].copy()

    # 3. Odstránenie pozície brankára a špecifických brankárskych metrík (Redukcia šumu)
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

    # Nahradenie chýbajúcich hodnôt pri fyzických a ekonomických parametroch nulami
    physical_cols = ['Height', 'Weight', 'Market value']
    for col in physical_cols:
        if col in young_players.columns:
            young_players[col] = young_players[col].replace(np.nan, 0)

    # Štandardizácia názvov stĺpcov pre bezproblémové spracovanie v Pythone
    young_players.columns = [c.replace(' ', '_').replace(',', '').replace('%', 'pct').replace('/', 'per').replace('.', '') for c in young_players.columns]
    
    # Export finálneho čistého datasetu
    young_players.to_csv('Cleaned_young_representative_players.csv', sep=';', index=False)

if __name__ == "__main__":
    process_scouting_data()