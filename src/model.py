from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from scipy.stats import norm
import warnings
import re

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

def get_league_weights():
    data = {
        'League': [
            'Polish Ekstraklasa', 'Azerbaijan Premyer Liga', 'Premyer Liga',
            'Germany 3. Liga', '3. Liga', 'Spain Primera Division', 'Bosnia Premijer Liga',
            'Albania Kategoria Superiore', 'Kazakhstan Premier League', 'France National 1',
            'Armenia Premier League', 'Turkey 1. Lig', 'Switzerland Challenge League',
            'Finland Veikkausliiga', 'Czech FNL', 'Iceland Besta-deild Karla', 'Italy Serie C',
            'Lithuania A Lyga', 'Georgia Erovnuli Liga', 'Latvia Virsliga', 'Moldova Super Liga',
            'Austria 2. Liga', 'Hungary NB 2', 'Israel Liga Leumit', 'Slovakia 2. Liga'
        ],
        'Opta_Rank': [
            11, 48, 48, 63, 63, 57, 62, 66, 71, 73, 81, 125, 91, 85, 108, 94, 
            129, 119, 109, 140, 135, 153, 134, 154, 223
        ],
        'Market_Value': [
            21.22, 10.35, 10.35, 7.77, 7.77, 5.05, 5.59, 4.92, 4.85, 3.74, 4.72, 8.18, 
            5.00, 3.02, 4.11, 2.43, 5.44, 4.07, 2.77, 4.24, 3.22, 4.23, 2.36, 2.73, 1.09
        ]
    }
    df = pd.DataFrame(data)
    df['Norm_Rank'] = 11.0 / df['Opta_Rank']
    df['Norm_MV'] = df['Market_Value'] / 21.22
    df['W_league'] = ((df['Norm_Rank'] * 0.75) + (df['Norm_MV'] * 0.25)) ** 0.5
    df['Final_Weight'] = 0.5 + (0.5 * df['W_league'])
    
    return {' '.join(k.lower().replace('_', ' ').split()): v for k, v in zip(df['League'], df['Final_Weight'])}

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

expert_benchmarks = {
    'forward': ['M. Ishak', 'T. Bobček', 'E. Koulouris'], 
    'winger': ['B. Nowak', 'K. Grosicki', 'C. Mena'],
    'central midfielder (8)': ['R. Kapič', 'B. Wolski', 'V. Kochergin'],
    'defensive midfielder (6)': ['I. Zhelizko', 'T. Romanczuk','P. Hellebrand'], 
    'attacking midfielder': ['Ivi López', 'Jesús Imaz', 'Afonso Sousa'],
    'fullback': ['B. Wdowik', 'Joel Pereira', 'E. Janža'],
    'central defender': ['S. Svarnas', 'A. Milić', 'O. Wójcik']
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

def run_prospect_model():
    df_eks_raw = pd.concat([
        pd.read_csv(DATA_DIR / 'Ekstraklasa_24_25.csv', sep=';'), 
        pd.read_csv(DATA_DIR / 'Ekstraklasa_25_26.csv', sep=';')
    ], ignore_index=True)
    
    df_eks = clean_data(df_eks_raw).sort_values('minutes_played', ascending=False).drop_duplicates('player').reset_index(drop=True)
    df_young = clean_data(pd.read_csv(DATA_DIR / 'Cleaned_young_representative_players.csv', sep=';'))
    
    league_weights_dict = get_league_weights()

    def assign_role(pos_str):
        if pd.isna(pos_str): return "unknown"
        primary = str(pos_str).replace(',', ' ').split()[0].upper()
        mapping = {'RW': 'winger', 'LW': 'winger', 'RWF': 'winger', 'LWF': 'winger'}
        return mapping.get(primary, "other")

    df_eks['scouting_role'] = df_eks['position'].apply(assign_role)
    df_young['scouting_role'] = df_young['position'].apply(assign_role)

    all_export_data = []

    for role, buckets in bucket_configs.items():
        all_metrics = [m for sublist in buckets.values() for m in sublist]
        found = [m for m in all_metrics if m in df_eks_raw.columns]
        d_eks = df_eks[(df_eks['scouting_role'] == role) & (df_eks['minutes_played'] >= 1200)].copy()
        d_you = df_young[df_young['scouting_role'] == role].copy()
        
        if d_eks.empty or not found: continue
        
        target_names = expert_benchmarks.get(role, [])
        d_eks['is_elite'] = d_eks['player'].apply(lambda x: 1 if any(str(t).lower() in str(x).lower() for t in target_names) else 0)

        param_grid = {'n_estimators': [100, 300, 500], 'max_depth': [3, 4, 5], 'min_samples_leaf': [1, 2]}
        rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
        cv_folds = StratifiedKFold(n_splits=min(3, max(1, d_eks['is_elite'].sum())), shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=cv_folds, scoring='balanced_accuracy', n_jobs=-1)
        grid_search.fit(d_eks[found], d_eks['is_elite'])
        
        feat_imp = pd.Series(grid_search.best_estimator_.feature_importances_, index=found)
        b_imp = {n: feat_imp[[m for m in m_list if m in found]].sum() for n, m_list in buckets.items()}
        
        total_weight = sum(b_imp.values()) if sum(b_imp.values()) > 0 else 1
        obj_weights = {k: v/total_weight for k, v in b_imp.items()}

        if not d_you.empty:
            scaler = StandardScaler()
            scaler.fit(d_eks[found])
            d_you_scaled = pd.DataFrame(scaler.transform(d_you[found]), columns=found, index=d_you.index)
            d_you['final_raw'] = 0
            
            for b_name, w in obj_weights.items():
                b_metrics = [m for m in buckets[b_name] if m in found]
                if b_metrics: 
                    b_raw = d_you_scaled[b_metrics].mean(axis=1)
                    d_you['final_raw'] += b_raw * w
                    d_you[f'Radar_{b_name.capitalize()}'] = norm.cdf(b_raw) * 100
                    
            def get_coeff(row_str):
                clean_target = ' '.join(str(row_str).lower().replace('_', ' ').split())
                for dict_league, weight in league_weights_dict.items():
                    if dict_league in clean_target: return weight
                return 0.70
                
            l_col = next((c for c in ['season', 'league', 'competition'] if c in d_you.columns), None)
            d_you['l_coeff'] = d_you[l_col].apply(get_coeff) if l_col else 0.70
            
            adj_perf = d_you['final_raw'] * d_you['l_coeff']
            d_you['Prospect_Rating'] = norm.cdf(adj_perf, loc=0, scale=1.0) * 100

            for m in found:
                d_you[f'Raw_{m}'] = d_you[m]
                d_you[f'Adj_{m}'] = d_you[m] * d_you['l_coeff']
                d_you[f'Pctl_{m}'] = norm.cdf(d_you_scaled[m]) * 100
                
            all_export_data.append(d_you)

    if all_export_data:
        df_export = pd.concat(all_export_data, ignore_index=True)
        threshold_mapping = {
            'central defender': 71.1,
            'fullback': 71.4,
            'defensive midfielder (6)': 61.3,
            'central midfielder (8)': 75.1,
            'attacking midfielder': 80.7,
            'winger': 67.8,
            'forward': 73.6
        }
        
        df_filtered = df_export[df_export.apply(lambda r: r['Prospect_Rating'] >= threshold_mapping.get(r['scouting_role'], 100), axis=1)]
        df_final = df_filtered.sort_values('Prospect_Rating', ascending=False).drop_duplicates('player')
        
        # Export do priečinka Data
        output_file = DATA_DIR / 'Prospect_Tableau_Export.csv'
        df_final.to_csv(output_file, index=False, sep=';', decimal='.', float_format='%.3f')
        print(f"✅ Hotovo: {output_file}")

if __name__ == "__main__":
    run_prospect_model()