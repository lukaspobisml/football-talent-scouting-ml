import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def clean_market_value(value):
    """Pomocná funkcia na konverziu trhovej hodnoty na numerický formát."""
    if pd.isna(value) or value == 0: 
        return 0.0
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

def evaluate_business_kpis():
    """
    Kvantifikácia ekonomického prínosu modelu a validácia voči trhovým štandardom.
    Generuje výstupy pre reportovanie biznisových KPI.
    """
    # 1. Načítanie vstupných dát
    try:
        df_shortlist = pd.read_csv('business_evaluation.csv', sep=';')
        df_full = pd.read_csv('Cleaned_young_representative_players.csv', sep=';')
    except FileNotFoundError:
        return

    # 2. Príprava a čistenie dátových štruktúr
    df_shortlist = df_shortlist.dropna(subset=['Hráč', 'Prospect Rating'])
    df_shortlist['MV_jun_2025_num'] = df_shortlist['MV jun 2025'].apply(clean_market_value)
    df_shortlist['MV_teraz_num'] = df_shortlist['MV teraz'].apply(clean_market_value)
    
    df_full['Market_value_num'] = df_full['Market_value'].apply(clean_market_value)

    # 3. Kvantifikácia KPI: Skautingová arbitráž (Pomer cena/výkon)
    df_shortlist['Arbitrage_Index'] = df_shortlist['Prospect Rating'] / ((df_shortlist['MV_jun_2025_num'] / 100000) + 1)
    
    arbitrage_top = df_shortlist.sort_values(by='Arbitrage_Index', ascending=False).head(15)
    export_arbitrage = arbitrage_top[['Hráč', 'Prospect Rating', 'MV jun 2025', 'MV teraz', 'Arbitrage_Index']]
    export_arbitrage.to_csv('KPI_Arbitrage_Top_Targets.csv', index=False, sep=';', decimal='.')

    # 4. Kvantifikácia KPI: Ochrana kapitálu (Eliminácia nadhodnotených hráčov / Expensive Flops)
    shortlist_names = df_shortlist['Hráč'].unique()
    expensive_non_shortlisted = df_full[~df_full['Player'].isin(shortlist_names)].sort_values(by='Market_value_num', ascending=False)
    
    flops_export = expensive_non_shortlisted[['Player', 'Team', 'Market_value_num', 'Primary_Position']].head(10)
    flops_export.to_csv('KPI_Expensive_Flops_Avoided.csv', index=False, sep=';', decimal='.')

    # 5. Kvantifikácia KPI: Celkové ekonomické zhodnotenie portfólia (ROI)
    initial_value = df_shortlist['MV_jun_2025_num'].sum()
    final_value = df_shortlist['MV_teraz_num'].sum()
    roi_percentage = ((final_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0

    roi_summary = pd.DataFrame({
        'Metric': ['Initial Portfolio Value (June 2025)', 'Current Portfolio Value (April 2026)', 'ROI (%)'],
        'Value': [initial_value, final_value, round(roi_percentage, 2)]
    })
    roi_summary.to_csv('KPI_ROI_Summary.csv', index=False, sep=';', decimal='.')

if __name__ == "__main__":
    evaluate_business_kpis()