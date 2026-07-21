"""
Spearman Rank Correlation Analysis: POI Features vs Building Problem Types
==========================================================================
Approach:
  1. Load building-level data from ground_truth Excel, aggregate to community-level problem rates
  2. Load POI category distributions from JSON
  3. Merge into community-level data matrix
  4. Compute Spearman rank correlation coefficients and p-values
  5. FDR multiple testing correction
  6. Heatmap + scatter plot (LOESS) visualization
  7. Compare with Pearson results
"""

import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns

# ===== Paths =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  # complete/
GT_EXCEL = os.path.join(PROJECT_DIR, 'ground_truth_final_neighborhood_filtered.xlsx')
POI_JSON = os.path.join(PROJECT_DIR, '04_Correction_and_Fusion', '小区周边POI数据_高德_mapped_v2_fixed.json')
OUTPUT_DIR = BASE_DIR  # output in same folder as script
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to Pearson results for comparison
PEARSON_EXCEL = os.path.join(PROJECT_DIR, '04_Correction_and_Fusion', 'Pearson_Correlation_Analysis', 'poi_correlation_results.xlsx')

# 7 building problem types
CLASS_NAMES = [
    'pipeline_damage',
    'addition_balconies_windows',
    'public_spaces_illegal_occupation',
    'no_aging_place_modifications',
    'wall_damage',
    'illegal_renovation_expansion',
    'elevator_addition',
]

CLASS_NAMES_EN = {
    'pipeline_damage': 'Pipeline Damage',
    'addition_balconies_windows': 'Addition/Balcony',
    'public_spaces_illegal_occupation': 'Public Space Occ.',
    'no_aging_place_modifications': 'Aging Modification',
    'wall_damage': 'Wall Damage',
    'illegal_renovation_expansion': 'Illegal Renovation',
    'elevator_addition': 'Elevator Addition',
}

POI_CN_TO_EN = {
    '住宿服务': 'Accommodation',
    '公司企业': 'Corporate',
    '医疗保健': 'Healthcare',
    '生活服务': 'Daily Service',
    '科教文化': 'Education/Culture',
    '购物服务': 'Shopping',
    '餐饮服务': 'Dining',
}
POI_EN_TO_CN = {v: k for k, v in POI_CN_TO_EN.items()}


# ============================================================
# 1. Data Loading
# ============================================================
def load_ground_truth():
    df = pd.read_excel(GT_EXCEL)
    print(f'Ground Truth: {len(df)} buildings, {df["社区"].nunique()} communities')
    return df


def load_poi():
    with open(POI_JSON, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    TARGET_POI_CATS = {'餐饮服务', '生活服务', '购物服务', '公司企业', '医疗保健', '科教文化', '住宿服务'}
    poi = {}
    for name, info in raw.items():
        raw_cats = info.get('category_distribution', {})
        filtered = {k: v for k, v in raw_cats.items() if k in TARGET_POI_CATS}
        poi[name] = filtered
    print(f'POI: {len(poi)} communities (7 target categories only)')
    return poi, sorted(TARGET_POI_CATS)


# ============================================================
# 2. Community-Level Matrix
# ============================================================
def build_community_matrix(df_gt, poi_dict, poi_categories):
    community_groups = df_gt.groupby('社区')
    rows = []
    for community, group in community_groups:
        n_buildings = len(group)
        row = {'Community': community, 'N_buildings': n_buildings}
        for cls in CLASS_NAMES:
            col = cls
            if col not in group.columns:
                col = f'{cls}_gt'
            if col not in group.columns:
                possible = [c for c in group.columns if cls in c]
                if possible:
                    col = possible[0]
                else:
                    row[f'{cls}_rate'] = 0
                    continue
            detected = (group[col].astype(float) > 0).sum()
            row[f'{cls}_rate'] = detected / n_buildings
        if community in poi_dict:
            cat_dist = poi_dict[community]
            total_poi = sum(cat_dist.values())
            row['poi_total'] = total_poi
            for cat in poi_categories:
                count = cat_dist.get(cat, 0)
                row[f'poi_{cat}'] = count
                row[f'poi_{cat}_ratio'] = count / total_poi if total_poi > 0 else 0
        else:
            row['poi_total'] = 0
            for cat in poi_categories:
                row[f'poi_{cat}'] = 0
                row[f'poi_{cat}_ratio'] = 0
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f'\nCommunity-level matrix: {len(df)} communities')
    return df


# ============================================================
# 3. Spearman Correlation
# ============================================================
def compute_correlation(df, poi_categories):
    problem_cols = [f'{c}_rate' for c in CLASS_NAMES]
    poi_cols = [f'poi_{c}_ratio' for c in poi_categories]

    mask = df['poi_total'] > 0
    df_valid = df[mask].copy()
    print(f'Valid communities: {len(df_valid)} / {len(df)}')

    n_problems = len(problem_cols)
    n_poi = len(poi_cols)
    corr_matrix = np.zeros((n_problems, n_poi))
    pval_matrix = np.zeros((n_problems, n_poi))

    for i, pcol in enumerate(problem_cols):
        for j, ocol in enumerate(poi_cols):
            rho, p = stats.spearmanr(df_valid[pcol], df_valid[ocol])
            corr_matrix[i, j] = rho
            pval_matrix[i, j] = p

    # FDR correction
    pvals_flat = pval_matrix.flatten()
    reject, pvals_corrected, _, _ = multipletests(pvals_flat, method='fdr_bh', alpha=0.05)
    pval_corrected_matrix = pvals_corrected.reshape(n_problems, n_poi)
    reject_matrix = reject.reshape(n_problems, n_poi)

    problem_labels = [CLASS_NAMES_EN.get(c, c) for c in CLASS_NAMES]
    poi_labels_en = [POI_CN_TO_EN.get(c, c) for c in poi_categories]

    corr_df = pd.DataFrame(corr_matrix, index=problem_labels, columns=poi_labels_en)
    pval_df = pd.DataFrame(pval_matrix, index=problem_labels, columns=poi_labels_en)
    pval_corr_df = pd.DataFrame(pval_corrected_matrix, index=problem_labels, columns=poi_labels_en)
    reject_df = pd.DataFrame(reject_matrix, index=problem_labels, columns=poi_labels_en)

    return corr_df, pval_df, pval_corr_df, reject_df, df_valid


# ============================================================
# 4. Print Results
# ============================================================
def print_results(corr_df, pval_df, pval_corr_df, reject_df):
    print('\n' + '=' * 80)
    print('Spearman Rank Correlation Matrix (rho)')
    print('=' * 80)
    print(corr_df.round(3).to_string())

    print('\n' + '=' * 80)
    print('p-value Matrix (Raw)')
    print('=' * 80)
    print(pval_df.round(4).to_string())

    print('\n' + '=' * 80)
    print('p-value Matrix (FDR Corrected)')
    print('=' * 80)
    print(pval_corr_df.round(4).to_string())

    print('\n' + '=' * 80)
    print('Significance Summary (FDR < 0.05)')
    print('=' * 80)
    sig_pairs = []
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            if reject_df.iloc[i, j]:
                r = corr_df.iloc[i, j]
                p = pval_corr_df.iloc[i, j]
                sig_pairs.append((corr_df.index[i], corr_df.columns[j], r, p))
    if sig_pairs:
        for problem, poi, r, p in sorted(sig_pairs, key=lambda x: abs(x[2]), reverse=True):
            direction = 'Positive' if r > 0 else 'Negative'
            print(f'  {problem} x {poi}: rho={r:.3f}, p(FDR)={p:.4f} [{direction}]')
    else:
        print('  No statistically significant pairs (FDR < 0.05)')
        print('\n  Uncorrected p < 0.05:')
        for i in range(corr_df.shape[0]):
            for j in range(corr_df.shape[1]):
                if pval_df.iloc[i, j] < 0.05:
                    r = corr_df.iloc[i, j]
                    p = pval_df.iloc[i, j]
                    print(f'    {corr_df.index[i]} x {corr_df.columns[j]}: rho={r:.3f}, p={p:.4f}')


# ============================================================
# 5. Heatmap
# ============================================================
def plot_heatmap(corr_df):
    fig, ax = plt.subplots(figsize=(12, 7))
    vmax = max(abs(corr_df.values.min()), abs(corr_df.values.max()), 0.5)

    sns.heatmap(
        corr_df, annot=False, cmap='RdBu_r', center=0,
        vmin=-vmax, vmax=vmax, linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Spearman rho', 'shrink': 0.8}, ax=ax,
    )

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            rho = corr_df.iloc[i, j]
            text = f'{rho:.2f}'
            color = 'white' if abs(rho) > vmax * 0.6 else 'black'
            ax.text(j + 0.5, i + 0.5, text,
                    ha='center', va='center', fontsize=9, color=color,
                    fontweight='bold', fontfamily='DejaVu Sans')

    ax.set_xlabel('POI Category', fontsize=12)
    ax.set_ylabel('Building Problem Type', fontsize=12)
    ax.set_title('Spearman Correlation: Problem Rate vs POI Features',
                 fontsize=14, pad=15)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'spearman_correlation_heatmap.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'\nHeatmap saved: {out_path}')
    plt.close()


# ============================================================
# 6. Scatter Plot with LOESS
# ============================================================
def plot_scatter(corr_df, pval_df, pval_corr_df, df_valid, poi_categories):
    sig_pairs = []
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            if pval_corr_df.iloc[i, j] < 0.05:
                sig_pairs.append((corr_df.index[i], corr_df.columns[j],
                                  corr_df.iloc[i, j], pval_corr_df.iloc[i, j]))

    if not sig_pairs:
        print('\nNo FDR-significant pairs; showing top 6 by uncorrected p-value')
        all_pairs = []
        for i in range(corr_df.shape[0]):
            for j in range(corr_df.shape[1]):
                all_pairs.append((corr_df.index[i], corr_df.columns[j],
                                  corr_df.iloc[i, j], pval_df.iloc[i, j]))
        sig_pairs = sorted(all_pairs, key=lambda x: x[3])[:6]
        subtitle = '(Top 6 by uncorrected p-value)'
    else:
        subtitle = '(FDR < 0.05)'
        sig_pairs = sorted(sig_pairs, key=lambda x: abs(x[2]), reverse=True)

    if not sig_pairs:
        print('No plottable correlation pairs')
        return

    n_pairs = len(sig_pairs)
    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    rate_col_map = {CLASS_NAMES_EN.get(c, c): f'{c}_rate' for c in CLASS_NAMES}
    ratio_col_map = {en: f'poi_{cn}_ratio' for cn, en in POI_CN_TO_EN.items()}

    for idx, (problem, poi_cat, rho, p) in enumerate(sig_pairs):
        ax = axes[idx]
        x_col = ratio_col_map.get(poi_cat, f'poi_{poi_cat}_ratio')
        y_col = rate_col_map.get(problem, f'{problem}_rate')

        if x_col not in df_valid.columns or y_col not in df_valid.columns:
            ax.text(0.5, 0.5, f'Missing column:\n{x_col}\n{y_col}',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        x = df_valid[x_col].values
        y = df_valid[y_col].values

        ax.scatter(x, y, alpha=0.5, s=30, edgecolors='white', linewidth=0.5)

        # LOESS smooth curve
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]
        try:
            smoothed = lowess(y_sorted, x_sorted, frac=0.6, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=1.5, alpha=0.8)
        except Exception:
            # fallback to linear fit
            z = np.polyfit(x, y, 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p_line(x_line), 'r--', linewidth=1.5, alpha=0.8)

        ax.set_xlabel(f'{poi_cat}', fontsize=10)
        ax.set_ylabel(f'{problem} Rate', fontsize=10)
        ax.set_title(f'rho={rho:.3f}, p={p:.4f}', fontsize=10)
        ax.grid(True, alpha=0.3)

    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Spearman: Problem Rate vs POI Features {subtitle}', fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'spearman_correlation_scatter.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Scatter plot saved: {out_path}')
    plt.close()


# ============================================================
# 7. Pearson vs Spearman Comparison
# ============================================================
def compare_with_pearson(spearman_corr, spearman_pval_corr, poi_categories):
    """Load Pearson results and generate a side-by-side comparison table."""
    if not os.path.exists(PEARSON_EXCEL):
        print(f'\n[Pearson results not found at {PEARSON_EXCEL}, skipping comparison]')
        return None

    pearson_corr = pd.read_excel(PEARSON_EXCEL, sheet_name='Correlation_r', index_col=0)
    pearson_pval_corr = pd.read_excel(PEARSON_EXCEL, sheet_name='P_FDR_corrected', index_col=0)

    # Build comparison table: for each pair, show Pearson r/p and Spearman rho/p
    rows = []
    for problem in spearman_corr.index:
        for poi in spearman_corr.columns:
            pearson_r = pearson_corr.loc[problem, poi]
            pearson_p = pearson_pval_corr.loc[problem, poi]
            spearman_r = spearman_corr.loc[problem, poi]
            spearman_p = spearman_pval_corr.loc[problem, poi]

            # Check consistency
            pearson_sig = pearson_p < 0.05
            spearman_sig = spearman_p < 0.05
            if pearson_sig and spearman_sig:
                consistency = 'Both sig.'
            elif pearson_sig:
                consistency = 'Pearson only'
            elif spearman_sig:
                consistency = 'Spearman only'
            else:
                consistency = 'Both ns.'

            rows.append({
                'Problem': problem,
                'POI Category': poi,
                'Pearson r': round(pearson_r, 3),
                'Pearson p(FDR)': round(pearson_p, 4),
                'Spearman rho': round(spearman_r, 3),
                'Spearman p(FDR)': round(spearman_p, 4),
                'Consistency': consistency,
            })

    df_compare = pd.DataFrame(rows)

    # Print summary
    print('\n' + '=' * 80)
    print('Pearson vs Spearman Comparison')
    print('=' * 80)

    sig_compare = df_compare[df_compare['Consistency'] != 'Both ns.']
    if len(sig_compare) > 0:
        print(sig_compare.to_string(index=False))
    else:
        print('  No significant pairs in either method.')

    both_sig = df_compare[df_compare['Consistency'] == 'Both sig.']
    print(f'\n  Both significant: {len(both_sig)} pairs')
    print(f'  Pearson only: {len(df_compare[df_compare["Consistency"] == "Pearson only"])} pairs')
    print(f'  Spearman only: {len(df_compare[df_compare["Consistency"] == "Spearman only"])} pairs')

    return df_compare


# ============================================================
# 8. Save Results
# ============================================================
def save_results(corr_df, pval_df, pval_corr_df, reject_df, df_community, df_compare):
    out_path = os.path.join(OUTPUT_DIR, 'spearman_correlation_results.xlsx')
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        corr_df.to_excel(writer, sheet_name='Spearman_rho')
        pval_df.to_excel(writer, sheet_name='P_raw')
        pval_corr_df.to_excel(writer, sheet_name='P_FDR_corrected')
        reject_df.to_excel(writer, sheet_name='Significant_FDR05')
        df_community.to_excel(writer, sheet_name='Community_Data', index=False)
        if df_compare is not None:
            df_compare.to_excel(writer, sheet_name='Pearson_vs_Spearman', index=False)
    print(f'\nResults saved: {out_path}')


# ============================================================
# Main
# ============================================================
def main():
    print('=' * 60)
    print('Spearman Rank Correlation: POI Features vs Problem Types')
    print('=' * 60)

    df_gt = load_ground_truth()
    poi_dict, poi_categories = load_poi()
    df_community = build_community_matrix(df_gt, poi_dict, poi_categories)

    corr_df, pval_df, pval_corr_df, reject_df, df_valid = compute_correlation(
        df_community, poi_categories
    )

    print_results(corr_df, pval_df, pval_corr_df, reject_df)
    plot_heatmap(corr_df)
    plot_scatter(corr_df, pval_df, pval_corr_df, df_valid, poi_categories)
    df_compare = compare_with_pearson(corr_df, pval_corr_df, poi_categories)
    save_results(corr_df, pval_df, pval_corr_df, reject_df, df_community, df_compare)

    print('\nDone!')


if __name__ == '__main__':
    main()

