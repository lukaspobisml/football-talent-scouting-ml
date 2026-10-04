from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

def clean_market_value(value):
    if pd.isna(value) or value == 0: 
        return 0.0
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)

def evaluate_business_kpis():
    shortlist_file = DATA_DIR / 'business_evaluation.csv'
    full_file = DATA_DIR / 'Cleaned_young_representative_players.csv'
    
    if not shortlist_file.exists() or not full_file.exists():
        print("❌ Chýbajú vstupné súbory pre biznisovú evaluáciu.")
        return

    df_shortlist = pd.read_csv(shortlist_file, sep=';')
    df_full = pd.read_csv(full_file, sep=';')

    # Podpora oboch názvov stĺpca skóre
    score_col = 'Prospect Rating' if 'Prospect Rating' in df_shortlist.columns else ('Prospect_Rating' if 'Prospect_Rating' in df_shortlist.columns else 'Widzew Score')

    df_shortlist = df_shortlist.dropna(subset=['Hráč', score_col])
    df_shortlist['MV_jun_2025_num'] = df_shortlist['MV jun 2025'].apply(clean_market_value)
    df_shortlist['MV_teraz_num'] = df_shortlist['MV teraz'].apply(clean_market_value)
    
    df_full['Market_value_num'] = df_full['Market_value'].apply(clean_market_value)

    # 1. Skautingová arbitráž
    df_shortlist['Arbitrage_Index'] = df_shortlist[score_col] / ((df_shortlist['MV_jun_2025_num'] / 100000) + 1)
    
    arbitrage_top = df_shortlist.sort_values(by='Arbitrage_Index', ascending=False).head(15)
    export_arbitrage = arbitrage_top[['Hráč', score_col, 'MV jun 2025', 'MV teraz', 'Arbitrage_Index']]
    export_arbitrage.to_csv(DATA_DIR / 'KPI_Arbitrage_Top_Targets.csv', index=False, sep=';', decimal='.')

    # 2. Ochrana kapitálu (Flops)
    shortlist_names = df_shortlist['Hráč'].unique()
    expensive_non_shortlisted = df_full[~df_full['Player'].isin(shortlist_names)].sort_values(by='Market_value_num', ascending=False)
    
    flops_export = expensive_non_shortlisted[['Player', 'Team', 'Market_value_num', 'Primary_Position']].head(10)
    flops_export.to_csv(DATA_DIR / 'KPI_Expensive_Flops_Avoided.csv', index=False, sep=';', decimal='.')

    # 3. ROI portfólia
    initial_value = df_shortlist['MV_jun_2025_num'].sum()
    final_value = df_shortlist['MV_teraz_num'].sum()
    roi_percentage = ((final_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0

    roi_summary = pd.DataFrame({
        'Metric': ['Initial Portfolio Value (June 2025)', 'Current Portfolio Value (April 2026)', 'ROI (%)'],
        'Value': [initial_value, final_value, round(roi_percentage, 2)]
    })
    roi_summary.to_csv(DATA_DIR / 'KPI_ROI_Summary.csv', index=False, sep=';', decimal='.')
    print("✅ Biznisové KPI úspešne vygenerované.")

if __name__ == "__main__":
    evaluate_business_kpis()