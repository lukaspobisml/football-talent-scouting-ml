import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
import re

warnings.filterwarnings('ignore')

# =====================================================================
# 1. LIGOVÉ KOEFICIENTY A METRIKY
# =====================================================================
def get_league_weights():
    data = {
        'League': [
            'Polish Ekstraklasa', 'Ekstraklasa', 'Azerbaijan Premyer Liga', 'Premyer Liga',
            'Germany 3. Liga', '3. Liga', 'Spain Primera Division', 'Primera Division', 'Primera División',
            'Bosnia Premijer Liga', 'Premijer Liga', 'Albania Kategoria Superiore', 'Kategoria Superiore',
            'Kazakhstan Premier League', 'Premier League', 'France National 1', 'National 1',
            'Armenia Premier League', 'Turkey 1. Lig', '1. Lig', 'Switzerland Challenge League', 'Challenge League',
            'Finland Veikkausliiga', 'Veikkausliiga', 'Czech FNL', 'FNL', 'Iceland Besta-deild Karla', 'Besta-deild Karla',
            'Italy Serie C', 'Serie C', 'Lithuania A Lyga', 'A Lyga', 'Georgia Erovnuli Liga', 'Erovnuli Liga',
            'Latvia Virsliga', 'Virsliga', 'Moldova Super Liga', 'Super Liga', 'Austria 2. Liga', '2. Liga',
            'Hungary NB 2', 'NB 2', 'NB II', 'Israel Liga Leumit', 'Liga Leumit', 'Slovakia 2. Liga', '2. liga'
        ],
        'Opta_Rank': [
            11, 11, 48, 48, 63, 63, 57, 57, 57, 62, 62, 66, 66, 71, 71, 73, 73, 81,
            125, 125, 91, 91, 85, 85, 108, 108, 94, 94, 129, 129, 119, 119, 109, 109,
            140, 140, 135, 135, 153, 153, 134, 134, 134, 154, 154, 223, 223
        ],
        'Market_Value': [
            21.22, 21.22, 10.35, 10.35, 7.77, 7.77, 5.05, 5.05, 5.05, 5.59, 5.59,
            4.92, 4.92, 4.85, 4.85, 3.74, 3.74, 4.72, 8.18, 8.18, 5.00, 5.00,
            3.02, 3.02, 4.11, 4.11, 2.43, 2.43, 5.44, 5.44, 4.07, 4.07, 2.77, 2.77,
            4.24, 4.24, 3.22, 3.22, 4.23, 4.23, 2.36, 2.36, 2.36, 2.73, 2.73, 1.09, 1.09
        ]
    }
    df = pd.DataFrame(data)
    df['Norm_Rank'] = 11.0 / df['Opta_Rank']
    df['Norm_MV'] = df['Market_Value'] / 21.22
    df['W_league'] = ((df['Norm_Rank'] * 0.75) + (df['Norm_MV'] * 0.25)) ** 0.5
    df['Final_Weight'] = 0.5 + (0.5 * df['W_league'])
    
    clean_dict = {}
    for k, v in zip(df['League'], df['Final_Weight']):
        clean_key = ' '.join(k.lower().replace('_', ' ').split())
        clean_dict[clean_key] = v
    return clean_dict

bucket_configs = {
    'forward': {
        'scoring': ['xg_per_90', 'goal_conversion_pct', 'shots_on_target_pct'],
        'positioning': ['successful_attacking_actions_per_90', 'touches_in_box_per_90', 'accelerations_per_90'],
        'physicality': ['aerial_duels_won_pct', 'offensive_duels_won_pct']
    },
    'attacking midfielder': {
        'threat': ['non_penalty_goals_per_90', 'xg_per_90', 'shots_on_target_pct'],
        'playmaking': ['xa_per_90', 'smart_passes_per_90', 'key_passes_per_90', 'deep_completions_per_90', 'through_passes_per_90'],
        'carrying': ['successful_dribbles_pct', 'progressive_runs_per_90', 'fouls_suffered_per_90'],
        'positioning': ['received_passes_per_90', 'touches_in_box_per_90']
    },
    'defensive midfielder (6)': {
        'defense': ['defensive_duels_won_pct', 'padj_interceptions', 'padj_sliding_tackles', 'aerial_duels_won_pct'],
        'distribution': ['accurate_passes_pct', 'accurate_long_passes_pct', 'progressive_passes_per_90'],
        'carrying': ['successful_dribbles_pct', 'progressive_runs_per_90', 'fouls_suffered_per_90']
    },
    'central midfielder (8)': {
        'creativity': ['xa_per_90', 'smart_passes_per_90', 'key_passes_per_90', 'deep_completions_per_90'],
        'distribution': ['accurate_passes_pct', 'accurate_long_passes_pct', 'progressive_passes_per_90'],
        'carrying': ['successful_dribbles_pct', 'progressive_runs_per_90', 'fouls_suffered_per_90']
    },
    'central defender': {
        'defense': ['successful_defensive_actions_per_90', 'defensive_duels_won_pct', 'aerial_duels_won_pct', 'padj_sliding_tackles', 'padj_interceptions', 'shots_blocked_per_90'],
        'build_up': ['accurate_passes_pct', 'accurate_forward_passes_pct', 'accurate_long_passes_pct', 'progressive_passes_per_90']
    },
    'winger': {
        'threat': ['non_penalty_goals_per_90', 'xg_per_90'],
        'creativity': ['xa_per_90', 'accurate_crosses_pct', 'key_passes_per_90', 'deep_completed_crosses_per_90'],
        'carrying': ['successful_dribbles_pct', 'progressive_runs_per_90', 'fouls_suffered_per_90'],
        'defense': ['padj_interceptions', 'defensive_duels_won_pct']
    },
    'fullback': {
        'defense': ['padj_interceptions', 'defensive_duels_won_pct', 'padj_sliding_tackles', 'aerial_duels_won_pct'],
        'progression': ['progressive_runs_per_90', 'progressive_passes_per_90', 'accurate_passes_pct'],
        'attacking_support': ['xa_per_90', 'deep_completed_crosses_per_90', 'accurate_crosses_pct']
    }
}

def clean_data(df):
    df.columns = [c.lower().replace(', %', '_pct').replace(' / ', '_per_').replace(' ', '_').replace('-', '_').replace('.', '').strip() for c in df.columns]
    id_cols = ['player', 'team', 'position', 'league', 'league_clean', 'competition', 'season', 'team_within_selected_timeframe'] 
    for col in df.columns:
        if col not in id_cols:
            try:
                val = df[col].astype(str).str.replace(',', '.').str.strip()
                df[col] = pd.to_numeric(val, errors='coerce').fillna(0)
            except: pass
    return df

# =====================================================================
# 2. HLAVNÁ ČASŤ - DIAGNOSTIKA A VÝPOČET
# =====================================================================
def run_discovery_pipeline_with_diagnostics():
    print("="*125)
    print(f"{'PROSPECT AI: DISCOVERY PIPELINE (S DIAGNOSTIKOU DÁT)':^125}")
    print("="*125)

    try:
        df_eks_raw = pd.concat([pd.read_csv('Ekstraklasa_24_25.csv', sep=';'), 
                                pd.read_csv('Ekstraklasa_25_26.csv', sep=';')], ignore_index=True)
        df_eks = clean_data(df_eks_raw)
        df_eks = df_eks.sort_values('minutes_played', ascending=False).drop_duplicates('player').reset_index(drop=True)
        
        df_all_raw = pd.read_csv('All_Leagues.csv', sep=';', decimal=',')
        df_all = clean_data(df_all_raw)
        
        if 'season' in df_all.columns:
            df_all['league_clean'] = df_all['season'].astype(str).apply(lambda x: " ".join(x.split('_')[:2]) if '_' in x else x)
        else:
            df_all['league_clean'] = 'Unknown'
            
    except Exception as e: 
        return print(f"❌ CHYBA NAČÍTANIA DÁT: {e}")

    # =====================================================================
    # --- DIAGNOSTIKA 1: Surové dáta a filtre ---
    print("\n--- 📊 DIAGNOSTIKA ČISTENIA DÁT (DATA FUNNEL) ---")
    print(f"1. Pôvodný počet riadkov v 'All_Leagues.csv': {len(df_all)}")
    
    df_young = df_all[(df_all['age'] <= 23) & (df_all['minutes_played'] >= 900)].copy()
    
    vyradeni_vek_minuty = len(df_all) - len(df_young)
    print(f"2. Počet riadkov po filtri (U23 & Minúty >= 900): {len(df_young)} (Vyradených: {vyradeni_vek_minuty} hráčov pre nízku minutáž/vek)")
    # =====================================================================

    league_weights_dict = get_league_weights()

    def assign_role(pos_str, prim_pos_str=''):
        pos = str(pos_str).lower()
        prim = str(prim_pos_str).lower().strip()
        
        # 1. Priradenie primárne podľa 'primary_position'
        if 'forward' in prim: return 'forward'
        if 'winger' in prim: return 'winger'
        if 'fullback' in prim or 'wing back' in prim: return 'fullback'
        if 'central defender' in prim: return 'central defender'
        if 'attacking midfielder' in prim: return 'attacking midfielder'
        
        # 2. Špeciálne delenie IBA pre Central Midfielders
        if 'central midfielder' in prim:
            # Ak má v detailnej pozícii 'DM' (Defensive Midfielder)
            if 'dm' in pos: 
                return 'defensive midfielder (6)'
            # Ak nemá DM, berieme ho ako klasickú osmičku (CMF)
            else:
                return 'central midfielder (8)'
                
        # 3. Fallback (Záloha) pre Ekstraklasu alebo staršie exporty bez primary_position
        fallback = str(pos_str).replace(',', ' ').split()[0].lower() if pd.notna(pos_str) else ""
        mapping = {
            'dmf': 'defensive midfielder (6)', 'ldmf': 'defensive midfielder (6)', 'rdmf': 'defensive midfielder (6)',
            'cmf': 'central midfielder (8)', 'lcmf': 'central midfielder (8)', 'rcmf': 'central midfielder (8)',
            'amf': 'attacking midfielder', 'lamf': 'attacking midfielder', 'ramf': 'attacking midfielder',
            'cb': 'central defender', 'lcb': 'central defender', 'rcb': 'central defender',
            'rb': 'fullback', 'lb': 'fullback', 'rwb': 'fullback', 'lwb': 'fullback',
            'rw': 'winger', 'lw': 'winger', 'rwf': 'winger', 'lwf': 'winger', 'lm': 'winger', 'rm': 'winger',
            'cf': 'forward', 'st': 'forward'
        }
        return mapping.get(fallback, "unmapped")

    df_eks['scouting_role'] = df_eks['position'].apply(lambda x: assign_role(x))
    primary_col = 'primary_position' if 'primary_position' in df_young.columns else 'position'
    df_young['scouting_role'] = df_young.apply(lambda row: assign_role(row.get('position', ''), row.get(primary_col, '')), axis=1)
    
    # =====================================================================
    # --- DIAGNOSTIKA 2: Brankári a iní nezaradení ---
    unmapped_count = len(df_young[df_young['scouting_role'] == 'unmapped'])
    df_young = df_young[df_young['scouting_role'] != 'unmapped']
    print(f"3. Počet riadkov po priradení pozícií: {len(df_young)} (Vyradených: {unmapped_count} hráčov bez podporovanej pozície / Brankári)")
    # =====================================================================

    if 'market_value' in df_eks.columns:
        df_eks['market_value'] = pd.to_numeric(df_eks['market_value'], errors='coerce').fillna(0)
    else:
        return print("❌ Ekstraklasa dataset nemá stĺpec Market Value!")
        
    df_eks_train = df_eks[df_eks['market_value'] > 0]

    all_predictions = []

    print("\n--- 🤖 SPÚŠŤAM ML PREDIKCIE (UČENIE NA EKSTRAKLASE) ---")
    for role, buckets in bucket_configs.items():
        all_metrics = [m for sublist in buckets.values() for m in sublist]
        found = [m for m in all_metrics if m in df_eks_train.columns]
        
        d_eks = df_eks_train[df_eks_train['scouting_role'] == role].fillna(0)
        d_you = df_young[df_young['scouting_role'] == role].fillna(0)
        
        if d_eks.empty or d_you.empty or not found: continue
            
        rf_reg = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)
        rf_reg.fit(d_eks[found], d_eks['market_value'])
        
        d_you['Raw_Expected_MV'] = rf_reg.predict(d_you[found])
        
        def get_coeff(row_str):
            if not isinstance(row_str, str): return 0.70
            clean_target = ' '.join(row_str.lower().replace('_', ' ').split())
            clean_target = re.sub(r'\b\d{2}(-\d{2})?\b', '', clean_target).strip()
            for dict_league, weight in league_weights_dict.items():
                if dict_league in clean_target:
                    return weight
            return 0.70

        l_col = next((c for c in ['season', 'league', 'competition', 'league_clean'] if c in d_you.columns), None)
        d_you['l_coeff'] = d_you[l_col].apply(get_coeff) if l_col else 0.70
        d_you['Expected_Market_Value'] = d_you['Raw_Expected_MV'] * d_you['l_coeff']
        
        all_predictions.append(d_you)

    if not all_predictions:
        return print("⚠️ Neboli spracovaní žiadni hráči.")

    df_final = pd.concat(all_predictions, ignore_index=True)
    
    # =====================================================================
    # --- DIAGNOSTIKA 3: Odstránenie duplicít ---
    pred_duplikatmi = len(df_final)
    df_final = df_final.sort_values('Expected_Market_Value', ascending=False).drop_duplicates('player')
    odstranene_duplikaty = pred_duplikatmi - len(df_final)
    print(f"4. Finálny počet UNIKÁTNYCH hráčov v rebríčku: {len(df_final)} (Odstránených: {odstranene_duplikaty} duplicitných riadkov u tých istých hráčov)")
    print("-" * 125)
    # =====================================================================

    print("\n>>> TOP 20 HRÁČOV S NAJLEPŠÍM DÁTOVÝM PROFILOM (ČISTÁ HERNÁ KVALITA)")
    print("-" * 125)
    
    cols_to_print = ['player', 'scouting_role', 'team', 'league_clean', 'Expected_Market_Value']
    if 'market_value' in df_final.columns:
        cols_to_print.append('market_value')

    top_20 = df_final.head(20)
    print(top_20[cols_to_print].to_string(index=False, float_format="%.0f"))
    print("-" * 125)

    export_filename = 'Top_Discovery_Expected_Value.csv'
    df_final.to_csv(export_filename, index=False, sep=';', decimal=',', float_format='%.0f')
    print(f"\n✅ ÚSPECH: Kompletný dataset s vypočítanou kvalitou uložený do: '{export_filename}'")

if __name__ == "__main__":
    run_discovery_pipeline_with_diagnostics()