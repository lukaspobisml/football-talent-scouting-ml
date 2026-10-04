import pandas as pd
import warnings

warnings.filterwarnings('ignore')

BUCKET_CONFIGS = {
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

def get_bucket_name(metric_name, player_role):
    if pd.isna(player_role):
        return 'Other'
        
    role_lower = str(player_role).lower()
    metric_lower = metric_name.lower()
    
    target_role = next((role for role in BUCKET_CONFIGS if role in role_lower), None)
    if not target_role:
        return 'Other'
        
    for bucket, metrics_list in BUCKET_CONFIGS[target_role].items():
        if any(m.lower() in metric_lower or metric_lower in m.lower() for m in metrics_list):
            return bucket.replace('_', ' ').title()
            
    return 'Other'

def prepare_tableau_export():
    try:
        df = pd.read_csv('Prospect_Tableau_Export.csv', sep=';', decimal='.')
    except FileNotFoundError:
        return

    pctl_cols = [c for c in df.columns if c.startswith('Pctl_')]
    base_metrics = [c.replace('Pctl_', '') for c in pctl_cols]
    
    id_vars = [c for c in ['player', 'team', 'position', 'scouting_role', 'season', 'league', 'competition', 'Prospect_Rating', 'l_coeff'] if c in df.columns]
    
    rows = []
    
    for _, row in df.iterrows():
        l_coeff = row.get('l_coeff', 1)
        player_role = row.get('scouting_role', row.get('position', ''))
        
        # Fáza 1: Agregácia do funkčných kategórií (Buckets)
        player_buckets = {}
        for metric in base_metrics:
            pctl_val = row.get(f'Pctl_{metric}')
            if pd.notna(pctl_val):
                bucket_name = get_bucket_name(metric, player_role)
                if bucket_name != 'Other':
                    if bucket_name not in player_buckets:
                        player_buckets[bucket_name] = {'raw_sum': 0, 'adj_sum': 0, 'count': 0}
                    player_buckets[bucket_name]['raw_sum'] += pctl_val
                    player_buckets[bucket_name]['adj_sum'] += (pctl_val * l_coeff)
                    player_buckets[bucket_name]['count'] += 1
                
        # Fáza 2: Transformácia do dlhého formátu (Long format) a tvorba štítkov
        for metric in base_metrics:
            pctl_val = row.get(f'Pctl_{metric}')
            if pd.notna(pctl_val):
                bucket_name = get_bucket_name(metric, player_role)
                if bucket_name == 'Other':
                    continue 

                new_row = {col: row[col] for col in id_vars}
                clean_name = metric.replace('_per_90', ' /90').replace('_pct', ' %').replace('_', ' ').title()
                
                real_raw = row.get(f'Raw_{metric}', None)
                real_adj = row.get(f'Adj_{metric}', None)
                str_raw = f"{real_raw:.2f}" if pd.notna(real_raw) else "N/A"
                str_adj = f"{real_adj:.2f}" if pd.notna(real_adj) else "N/A"
                
                # Zápis do kategórií pre Tableau kalkulácie
                new_row['Bucket_Name'] = bucket_name
                new_row['Bucket_Score_Raw'] = player_buckets[bucket_name]['raw_sum'] / player_buckets[bucket_name]['count']
                new_row['Bucket_Score_Adj'] = player_buckets[bucket_name]['adj_sum'] / player_buckets[bucket_name]['count']
                
                # Dynamické štítky spájajúce percentily s absolútnymi produkciami
                new_row['Metric Name'] = clean_name
                new_row['Metric Name Raw'] = f"{clean_name} ({str_raw})"
                new_row['Metric Name Adjusted'] = f"{clean_name} ({str_adj})"
                
                new_row['Shape_Value_Raw'] = pctl_val
                new_row['Shape_Value_Adj'] = pctl_val * l_coeff
                new_row['Real_Value_Raw'] = real_raw
                new_row['Real_Value_Adj'] = real_adj
                
                rows.append(new_row)
                
    df_radar = pd.DataFrame(rows)
    
    # Preusporiadanie stĺpcov pre lepšiu čitateľnosť exportu
    first_cols = id_vars + ['Bucket_Name', 'Metric Name', 'Metric Name Raw', 'Metric Name Adjusted']
    df_radar = df_radar[first_cols + [c for c in df_radar.columns if c not in first_cols]]

    df_radar.to_csv('Prospect_Detailed_Radar_Export.csv', index=False, sep=';', decimal='.', float_format='%.3f')

if __name__ == "__main__":
    prepare_tableau_export()