from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

def clean_col(c):
    return str(c).replace(', %', '_pct').replace(' %', '_pct').replace('%', 'pct')\
                .replace(' ', '_').replace('/', '_').replace('.', '')\
                .replace('-', '_').replace('__', '_').lower().strip()

def plot_global_high_correlations(df, valid_metrics, threshold=0.75):
    numeric_df = df[valid_metrics].select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    mask_high_corr = (np.abs(corr_matrix) >= threshold) & (corr_matrix != 1.0)
    cols_to_keep = corr_matrix.columns[mask_high_corr.any()].tolist()

    if len(cols_to_keep) < 2:
        return

    filtered_corr = corr_matrix.loc[cols_to_keep, cols_to_keep]

    plt.figure(figsize=(10, 8))
    mask_upper = np.triu(np.ones_like(filtered_corr, dtype=bool))

    sns.heatmap(filtered_corr, 
                mask=mask_upper,
                annot=False, 
                fmt=".2f", 
                cmap='coolwarm', 
                vmin=-1, vmax=1, 
                center=0,
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .7})

    plt.title('Matica metrík vykazujúcich multikolinearitu', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def analyze_global_correlations():
    print("--- 🌍 GLOBÁLNA ANALÝZA MULTIKOLINEARITY ---")

    try:
        df1 = pd.read_csv(DATA_DIR / 'Ekstraklasa_24_25.csv', sep=';', decimal=',')
        df2 = pd.read_csv(DATA_DIR / 'Ekstraklasa_25_26.csv', sep=';', decimal=',')
        df = pd.concat([df1, df2]).drop_duplicates(subset=['Player', 'Team', 'Minutes played']).reset_index(drop=True)
    except Exception as e:
        print(f"❌ Chyba pri načítaní dát: {e}")
        return

    df.columns = [clean_col(c) for c in df.columns]
    df = df[pd.to_numeric(df['minutes_played'], errors='coerce') >= 600].copy()

    configs = {
        'Forward': {
            'Scoring': ['non_penalty_goals_per_90', 'xg_per_90', 'goal_conversion_pct', 'shots_per_90'],
            'Positioning': ['received_passes_per_90', 'touches_in_box_per_90', 'accelerations_per_90'],
            'Physicality': ['aerial_duels_won_pct', 'aerial_duels_per_90', 'offensive_duels_won_pct']
        },
        'Winger': {
            'Threat': ['non_penalty_goals_per_90', 'xg_per_90', 'shots_per_90'],
            'Creativity': ['xa_per_90', 'accurate_crosses_pct', 'key_passes_per_90', 'passes_to_penalty_area_per_90'],
            'Carrying': ['dribbles_per_90', 'progressive_runs_per_90', 'fouls_suffered_per_90'],
            'Defense': ['padj_interceptions', 'successful_defensive_actions_per_90']
        },
        'Attacking_Midfielder': {
            'Threat': ['non_penalty_goals_per_90', 'xg_per_90', 'shots_per_90'],
            'Playmaking': ['xa_per_90', 'smart_passes_per_90', 'key_passes_per_90', 'passes_to_penalty_area_per_90'],
            'Carrying': ['dribbles_per_90', 'progressive_runs_per_90', 'accelerations_per_90'],
            'Positioning': ['received_passes_per_90', 'touches_in_box_per_90']
        },
        'Central_Midfielder': {
            'Creativity': ['xa_per_90', 'smart_passes_per_90', 'key_passes_per_90'],
            'Distribution': ['accurate_passes_pct', 'received_passes_per_90', 'passes_to_final_third_per_90', 'progressive_passes_per_90'],
            'Defense': ['defensive_duels_won_pct', 'successful_defensive_actions_per_90', 'padj_interceptions'],
            'Carrying': ['progressive_runs_per_90', 'accelerations_per_90']
        },
        'Central_Defender': {
            'Defense': ['successful_defensive_actions_per_90', 'defensive_duels_won_pct', 'aerial_duels_won_pct', 'padj_interceptions', 'fouls_per_90'],
            'Build-up': ['accurate_passes_pct', 'accurate_long_passes_pct', 'progressive_passes_per_90', 'average_pass_length_m']
        },
        'Fullback': {
            'Defense': ['padj_interceptions', 'defensive_duels_won_pct', 'successful_defensive_actions_per_90', 'aerial_duels_won_pct'],
            'Progression': ['progressive_runs_per_90', 'accelerations_per_90', 'progressive_passes_per_90', 'accurate_passes_pct'],
            'Attacking_support': ['xa_per_90', 'touches_in_box_per_90', 'accurate_crosses_pct']
        }
    }

    high_corr_threshold = 0.75

    all_metrics_set = set()
    for pos, buckets in configs.items():
        for b_name, metrics in buckets.items():
            all_metrics_set.update(metrics)
            
    valid_metrics = [m for m in all_metrics_set if m in df.columns]
    
    for m in valid_metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce').fillna(0)

    corr_matrix = df[valid_metrics].corr()
    problematic_pairs = []
    for i in range(len(valid_metrics)):
        for j in range(i+1, len(valid_metrics)):
            m1 = valid_metrics[i]
            m2 = valid_metrics[j]
            corr_val = corr_matrix.loc[m1, m2]
            if abs(corr_val) >= high_corr_threshold:
                problematic_pairs.append((m1, m2, corr_val, abs(corr_val)))

    if problematic_pairs:
        problematic_pairs.sort(key=lambda x: x[3], reverse=True)
        print(f"⚠️ NÁJDENÉ VYSOKÉ ZHODY (r >= {high_corr_threshold}):")
        for m1, m2, val, _ in problematic_pairs:
            print(f"  🔴 ({val:.2f}): '{m1}' a '{m2}'")
        plot_global_high_correlations(df, valid_metrics, threshold=high_corr_threshold)
    else:
        print(f"✅ Žiadne metriky nepresahujú prah {high_corr_threshold}.")

if __name__ == "__main__":
    analyze_global_correlations()