"""
Correlation Analysis: POI Features vs Building Problem Types
============================================================
Approach:
  1. Load building-level data from ground_truth Excel, aggregate to community-level problem rates
  2. Load POI category distributions from JSON
  3. Merge into community-level data matrix
  4. Compute Pearson correlation coefficients and p-values
  5. FDR multiple testing correction
  6. Heatmap + scatter plot visualization
"""

import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns

# ===== 路径 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
GT_EXCEL = os.path.join(PROJECT_DIR, 'ground_truth_final_neighborhood_filtered.xlsx')
POI_JSON = os.path.join(PROJECT_DIR, '04_Correction_and_Fusion', '小区周边POI数据_高德_mapped_v2_fixed.json')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', '实验结果_Reports')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 7类建筑问题
CLASS_NAMES = [
    'pipeline_damage',
    'addition_balconies_windows',
    'public_spaces_illegal_occupation',
    'no_aging_place_modifications',
    'wall_damage',
    'illegal_renovation_expansion',
    'elevator_addition',
]

CLASS_NAMES_CN = {
    'pipeline_damage': 'Pipeline Damage',
    'addition_balconies_windows': 'Addition/Balcony',
    'public_spaces_illegal_occupation': 'Public Space Occ.',
    'no_aging_place_modifications': 'Aging Modification',
    'wall_damage': 'Wall Damage',
    'illegal_renovation_expansion': 'Illegal Renovation',
    'elevator_addition': 'Elevator Addition',
}


# ============================================================
# 1. 数据加载
# ============================================================
def load_ground_truth():
    """Load ground truth Excel, return building-level data."""
    df = pd.read_excel(GT_EXCEL)
    print(f'Ground Truth: {len(df)} buildings, {df["社区"].nunique()} communities')
    print(f'Columns: {list(df.columns)}')
    return df


def load_poi():
    """Load POI JSON, return {community: category_distribution} with only 7 target categories."""
    with open(POI_JSON, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Only keep the 7 target categories (matches fetch_poi_amap.py)
    TARGET_POI_CATS = {'餐饮服务', '生活服务', '购物服务', '公司企业', '医疗保健', '科教文化', '住宿服务'}

    poi = {}
    for name, info in raw.items():
        raw_cats = info.get('category_distribution', {})
        # Filter to target categories only
        filtered = {k: v for k, v in raw_cats.items() if k in TARGET_POI_CATS}
        poi[name] = filtered

    print(f'POI: {len(poi)} communities (7 target categories only)')
    return poi, sorted(TARGET_POI_CATS)


# ============================================================
# 2. 构建社区级数据矩阵
# ============================================================
def build_community_matrix(df_gt, poi_dict, poi_categories):
    """
    Aggregate by community:
      - Problem rate = buildings with problem detected / total buildings in community
      - POI features = category ratios (category count / total POI count)
    """
    # Aggregate problem rates by community
    community_groups = df_gt.groupby('社区')
    rows = []
    for community, group in community_groups:
        n_buildings = len(group)
        row = {'Community': community, 'N_buildings': n_buildings}

        for cls in CLASS_NAMES:
            # Check column name variants
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

            # Problem rate: buildings with problem / total buildings
            detected = (group[col].astype(float) > 0).sum()
            row[f'{cls}_rate'] = detected / n_buildings

        # POI features
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
    print(f'  Problem rate cols: {[f"{c}_rate" for c in CLASS_NAMES]}')
    print(f'  POI ratio cols: {[f"poi_{c}_ratio" for c in poi_categories]}')
    return df


# ============================================================
# 3. 相关性分析
# ============================================================
def compute_correlation(df, poi_categories):
    """
    Compute Pearson correlation and p-values between 7 problem rates and POI category ratios.
    Returns correlation matrix, p-value matrix, FDR-corrected p-value matrix, and reject matrix.
    """
    problem_cols = [f'{c}_rate' for c in CLASS_NAMES]
    poi_cols = [f'poi_{c}_ratio' for c in poi_categories]

    # Only keep communities with POI data
    mask = df['poi_total'] > 0
    df_valid = df[mask].copy()
    print(f'\nValid communities (with POI data): {len(df_valid)} / {len(df)}')

    n_problems = len(problem_cols)
    n_poi = len(poi_cols)
    corr_matrix = np.zeros((n_problems, n_poi))
    pval_matrix = np.zeros((n_problems, n_poi))

    for i, pcol in enumerate(problem_cols):
        for j, ocol in enumerate(poi_cols):
            r, p = stats.pearsonr(df_valid[pcol], df_valid[ocol])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p

    # FDR multiple testing correction
    pvals_flat = pval_matrix.flatten()
    reject, pvals_corrected, _, _ = multipletests(pvals_flat, method='fdr_bh', alpha=0.05)
    pval_corrected_matrix = pvals_corrected.reshape(n_problems, n_poi)
    reject_matrix = reject.reshape(n_problems, n_poi)

    # Build DataFrame
    problem_labels = [CLASS_NAMES_CN.get(c, c) for c in CLASS_NAMES]
    poi_labels = poi_categories  # keep original Chinese names for reference

    # Translate POI category names to English for publication
    POI_CN_TO_EN = {
        '住宿服务': 'Accommodation',
        '公司企业': 'Corporate',
        '医疗保健': 'Healthcare',
        '汽车服务': 'Auto Service',
        '汽车维修': 'Auto Repair',
        '汽车销售': 'Auto Sales',
        '生活服务': 'Daily Service',
        '科教文化': 'Education/Culture',
        '购物服务': 'Shopping',
        '餐饮服务': 'Dining',
    }
    poi_labels_en = [POI_CN_TO_EN.get(c, c) for c in poi_categories]

    corr_df = pd.DataFrame(corr_matrix, index=problem_labels, columns=poi_labels_en)
    pval_df = pd.DataFrame(pval_matrix, index=problem_labels, columns=poi_labels_en)
    pval_corr_df = pd.DataFrame(pval_corrected_matrix, index=problem_labels, columns=poi_labels_en)
    reject_df = pd.DataFrame(reject_matrix, index=problem_labels, columns=poi_labels_en)

    return corr_df, pval_df, pval_corr_df, reject_df, df_valid


# ============================================================
# 4. 打印结果
# ============================================================
def print_results(corr_df, pval_df, pval_corr_df, reject_df):
    """Print correlation coefficients, p-values and significance markers."""
    print('\n' + '=' * 80)
    print('Pearson Correlation Matrix (r)')
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
            print(f'  {problem} x {poi}: r={r:.3f}, p(FDR)={p:.4f} [{direction}]')
    else:
        print('  No statistically significant pairs (FDR < 0.05)')
        # Show uncorrected significant pairs
        print('\n  Uncorrected p < 0.05:')
        for i in range(corr_df.shape[0]):
            for j in range(corr_df.shape[1]):
                if pval_df.iloc[i, j] < 0.05:
                    r = corr_df.iloc[i, j]
                    p = pval_df.iloc[i, j]
                    print(f'    {corr_df.index[i]} x {corr_df.columns[j]}: r={r:.3f}, p={p:.4f}')


# ============================================================
# 5. 热力图
# ============================================================
def plot_heatmap(corr_df, pval_corr_df, reject_df):
    """Plot correlation heatmap with significance markers."""
    fig, ax = plt.subplots(figsize=(12, 7))

    vmax = max(abs(corr_df.values.min()), abs(corr_df.values.max()), 0.5)

    sns.heatmap(
        corr_df,
        annot=False,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        fmt='.2f',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Pearson r', 'shrink': 0.8},
        ax=ax,
    )

    # Annotate each cell with r value only
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            r = corr_df.iloc[i, j]
            text = f'{r:.2f}'
            color = 'white' if abs(r) > vmax * 0.6 else 'black'
            ax.text(j + 0.5, i + 0.5, text,
                    ha='center', va='center', fontsize=9, color=color,
                    fontweight='bold', fontfamily='DejaVu Sans')

    ax.set_xlabel('POI Category', fontsize=12)
    ax.set_ylabel('Building Problem Type', fontsize=12)
    ax.set_title('Pearson Correlation: Problem Rate vs POI Features',
                 fontsize=14, pad=15)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'poi_correlation_heatmap.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'\nHeatmap saved: {out_path}')
    plt.close()


# ============================================================
# 6. 散点图（显著相关对）
# ============================================================
def plot_scatter(corr_df, pval_df, pval_corr_df, df_valid, poi_categories):
    """Plot scatter plots with regression lines for significant pairs."""
    # Build reverse mapping: English label -> Chinese name (for DataFrame column lookup)
    POI_CN_TO_EN = {
        '住宿服务': 'Accommodation', '公司企业': 'Corporate', '医疗保健': 'Healthcare',
        '生活服务': 'Daily Service', '科教文化': 'Education/Culture',
        '购物服务': 'Shopping', '餐饮服务': 'Dining',
    }
    POI_EN_TO_CN = {v: k for k, v in POI_CN_TO_EN.items()}
    sig_pairs = []
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            if pval_corr_df.iloc[i, j] < 0.05:
                sig_pairs.append((corr_df.index[i], corr_df.columns[j],
                                  corr_df.iloc[i, j], pval_corr_df.iloc[i, j]))

    # 如果没有显著的，取 top 6 未校正 p 最小的
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

    # Column name mapping: English label -> DataFrame column name
    rate_col_map = {CLASS_NAMES_CN.get(c, c): f'{c}_rate' for c in CLASS_NAMES}
    ratio_col_map = {en: f'poi_{cn}_ratio' for cn, en in POI_CN_TO_EN.items()}

    for idx, (problem, poi_cat, r, p) in enumerate(sig_pairs):
        ax = axes[idx]
        x_col = ratio_col_map.get(poi_cat, f'poi_{poi_cat}_ratio')
        y_col = rate_col_map.get(problem, f'{problem}_rate')

        if x_col not in df_valid.columns or y_col not in df_valid.columns:
            ax.text(0.5, 0.5, f'Missing column:\n{x_col}\n{y_col}',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        x = df_valid[x_col]
        y = df_valid[y_col]

        ax.scatter(x, y, alpha=0.5, s=30, edgecolors='white', linewidth=0.5)

        # 回归线
        z = np.polyfit(x, y, 1)
        p_line = np.poly1d(z)
        x_sorted = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_sorted, p_line(x_sorted), 'r-', linewidth=1.5, alpha=0.8)

        ax.set_xlabel(f'{poi_cat}', fontsize=10)
        ax.set_ylabel(f'{problem} Rate', fontsize=10)
        ax.set_title(f'r={r:.3f}, p={p:.4f}', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Problem Rate vs POI Features {subtitle}', fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'poi_correlation_scatter.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Scatter plot saved: {out_path}')
    plt.close()


# ============================================================
# 7. 保存详细结果到 Excel
# ============================================================
def save_results(corr_df, pval_df, pval_corr_df, reject_df, df_community):
    """Save correlation analysis results and community data to Excel."""
    out_path = os.path.join(OUTPUT_DIR, 'poi_correlation_results.xlsx')
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        corr_df.to_excel(writer, sheet_name='Correlation_r')
        pval_df.to_excel(writer, sheet_name='P_raw')
        pval_corr_df.to_excel(writer, sheet_name='P_FDR_corrected')
        reject_df.to_excel(writer, sheet_name='Significant_FDR05')
        df_community.to_excel(writer, sheet_name='Community_Data', index=False)
    print(f'\nResults saved: {out_path}')


# ============================================================
# Main
# ============================================================
def main():
    print('=' * 60)
    print('Correlation Analysis: POI Features vs Building Problem Types')
    print('=' * 60)

    # 1. Load data
    df_gt = load_ground_truth()
    poi_dict, poi_categories = load_poi()

    # 2. Build community-level matrix
    df_community = build_community_matrix(df_gt, poi_dict, poi_categories)

    # 3. Compute correlation
    corr_df, pval_df, pval_corr_df, reject_df, df_valid = compute_correlation(
        df_community, poi_categories
    )

    # 4. Print results
    print_results(corr_df, pval_df, pval_corr_df, reject_df)

    # 5. Heatmap
    plot_heatmap(corr_df, pval_corr_df, reject_df)

    # 6. Scatter plots
    plot_scatter(corr_df, pval_df, pval_corr_df, df_valid, poi_categories)

    # 7. Save to Excel
    save_results(corr_df, pval_df, pval_corr_df, reject_df, df_community)

    print('\nDone!')


if __name__ == '__main__':
    main()

