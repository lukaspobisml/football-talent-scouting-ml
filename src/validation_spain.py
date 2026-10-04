import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

# 1. KONFIGURÁCIA BUCKETOV (Zostáva zachovaná pre konzistenciu modelu Prospect)
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
        'defense': ['defensive_duels_won_pct', 'padj_interceptions', 'padj_sliding_tackles'],
        'distribution': ['accurate_passes_pct', 'accurate_long_passes_pct', 'progressive_passes_per_90'],
        'physicality': ['aerial_duels_won_pct']
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
    id_cols = ['player', 'team', 'position', 'league', 'competition'] 
    for col in df.columns:
        if col not in id_cols:
            try:
                val = df[col].astype(str).str.replace(',', '.').str.strip()
                df[col] = pd.to_numeric(val, errors='coerce').fillna(0)
            except: pass
    return df

def assign_role_from_shortcut(pos_str):
    if pd.isna(pos_str): return "other"
    primary = str(pos_str).replace(',', ' ').split()[0].upper()
    mapping = {
        'DMF': 'defensive midfielder (6)', 'LDMF': 'defensive midfielder (6)', 'RDMF': 'defensive midfielder (6)',
        'CMF': 'central midfielder (8)', 'LCMF': 'central midfielder (8)', 'RCMF': 'central midfielder (8)',
        'AMF': 'attacking midfielder', 'LAMF': 'attacking midfielder', 'RAMF': 'attacking midfielder',
        'CB': 'central defender', 'LCB': 'central defender', 'RCB': 'central defender',
        'RB': 'fullback', 'LB': 'fullback', 'RWB': 'fullback', 'LWB': 'fullback',
        'RW': 'winger', 'LW': 'winger', 'RWF': 'winger', 'LWF': 'winger', 'LM': 'winger', 'RM': 'winger',
        'CF': 'forward', 'ST': 'forward', 'SS': 'forward'
    }
    return mapping.get(primary, "other")

def run_spanish_validation_master():
    # Načítanie dát - predpokladáme súbory za posledné dve sezóny La Ligy
    try:
        # Možno použiť tvoj formát Spain_La Liga_24-25.csv
        df_24 = pd.read_csv('Spain_La Liga_24-25.csv', sep=';') 
        df_25 = pd.read_csv('Spain_La Liga_25-26.csv', sep=';')
        df = pd.concat([df_24, df_25], ignore_index=True)
    except Exception as e:
        return print(f"❌ CHYBA NAČÍTANIA ŠPANIELSKYCH DÁT: {e}")

    df = clean_data(df)
    # Odstránenie brankárov a priradenie rolí
    df = df[~df['position'].astype(str).str.contains('GK', na=False)].copy()
    df['scouting_role'] = df['position'].apply(assign_role_from_shortcut)
    
    # Agregácia na hráča (ak hral v oboch sezónach)
    all_metrics = list(set([m for b in bucket_configs.values() for sub in b.values() for m in sub]))
    pool = [c for c in all_metrics if c in df.columns]
    
    df_agg = df.groupby(['player', 'scouting_role']).agg({
        'minutes_played': 'sum',
        'team': 'last',
        **{m: 'mean' for m in pool}
    }).reset_index()

    print("\n" + "="*125)
    print(f"{'PROSPECT SCOUTING: EXTERNÁ VALIDÁCIA MODELU NA LA LIGE (MIN 1500+ min)':^125}")
    print("="*125)

    for role, buckets in bucket_configs.items():
        metrics_in_role = [m for sublist in buckets.values() for m in sublist if m in df_agg.columns]
        
        # Filtrujeme hráčov danej role s dostatočnou minutážou
        d_role = df_agg[(df_agg['scouting_role'] == role) & (df_agg['minutes_played'] >= 1800)].copy()
        
        if len(d_role) < 10: continue

        # 1. Identifikácia lokálnej elity pre učenie váh atribútov
        scaler = StandardScaler()
        d_scaled = pd.DataFrame(scaler.fit_transform(d_role[metrics_in_role]), 
                                columns=metrics_in_role, index=d_role.index)
        
        # Predbežné skóre pre určenie benchmarku (Top 15% ligy)
        d_role['temp_perf'] = d_scaled.mean(axis=1)
        threshold = d_role['temp_perf'].quantile(0.85)
        d_role['is_elite'] = (d_role['temp_perf'] >= threshold).astype(int)

        # 2. Trénovanie Prospect AI na rozpoznanie dôležitosti bucketov
        model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        model.fit(d_role[metrics_in_role], d_role['is_elite'])
        
        feat_imp = pd.Series(model.feature_importances_, index=metrics_in_role)
        
        # Prepočet váh pre buckety
        b_weights = {}
        for b_name, m_list in buckets.items():
            valid_m = [m for m in m_list if m in metrics_in_role]
            b_weights[b_name] = feat_imp[valid_m].sum()
        
        total_w = sum(b_weights.values()) if sum(b_weights.values()) > 0 else 1
        norm_weights = {k: v/total_w for k, v in b_weights.items()}

        # 3. Finálny výpočet Prospect Indexu pre La Ligu
        d_role['final_raw'] = 0
        for b_name, weight in norm_weights.items():
            b_metrics = [m for m in buckets[b_name] if m in metrics_in_role]
            if b_metrics:
                d_role['final_raw'] += d_scaled[b_metrics].mean(axis=1) * weight

        # CDF Normalizácia na 0-100 (ako v tvojom pôvodnom Prospect modeli)
        sigma = d_role['final_raw'].std() if d_role['final_raw'].std() > 0 else 1.0
        d_role['Prospect_Rating'] = norm.cdf(d_role['final_raw'], loc=0, scale=sigma) * 100

        # Výpis výsledkov
        print(f"\n>>> POZÍCIA: {role.upper()} | Počet testovaných: {len(d_role)}")
        top_10 = d_role.sort_values('Prospect_Rating', ascending=False).head(10)
        print(top_10[['player', 'team', 'Prospect_Rating']].to_string(index=False, float_format="%.1f"))
        print("-" * 60)

if __name__ == "__main__":
    run_spanish_validation_master()