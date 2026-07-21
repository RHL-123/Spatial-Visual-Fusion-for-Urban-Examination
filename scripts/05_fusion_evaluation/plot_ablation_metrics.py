import os
import re
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np

# 学术绘图全局设置
plt.rcParams['font.family'] = 'DejaVu Sans' # 使用内置兼容字体
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.axisbelow'] = True

def parse_txt(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按照类别分割
    blocks = re.split(r'\[(.*?)\]:', content)[1:]
    for i in range(0, len(blocks), 2):
        cls_name = blocks[i].strip()
        block_text = blocks[i+1]
        
        data[cls_name] = {}
        # 匹配提取数据：-> Baseline (...) P: 0.0419, R: 0.2258, F1: 0.0707, ACC: 0.9000
        lines = block_text.strip().split('\n')
        for line in lines:
            match = re.search(
                r'-> (.*?)\s*P:\s*([\d.]+),\s*R:\s*([\d.]+),\s*F1:\s*([\d.]+)(?:,\s*ACC:\s*([\d.]+))?',
                line,
            )
            if match:
                method = match.group(1).strip()
                p, r, f1 = float(match.group(2)), float(match.group(3)), float(match.group(4))
                acc = float(match.group(5)) if match.group(5) is not None else 0.0
                
                # 规范化方法名称缩写，适合图例展示
                method_mapped = method
                if 'Baseline' in method: method_mapped = 'Baseline'
                elif 'Ablation A' in method: method_mapped = 'Ablation A (Visual Only)'
                elif 'Ablation B' in method: method_mapped = 'Ablation B (POI Only)'
                elif 'Fusion' in method: method_mapped = 'Full Fusion'
                
                data[cls_name][method_mapped] = {'P': p, 'R': r, 'F1': f1, 'ACC': acc}
    return data

def plot_grouped_bar_metric(data, metric_key, metric_label, output_dir, source_file=""):
    classes = list(data.keys())
    # 类别名称格式化换行以适应X轴
    clean_classes = [c.replace('_', '\n').title() for c in classes]
    methods = ['Baseline', 'Ablation A (Visual Only)', 'Ablation B (POI Only)', 'Full Fusion']
    
    x = np.arange(len(classes))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(15, 7))
    # 科研标准配色 (Colorblind-friendly)
    colors = ['#ced4da', '#2ca02c', '#1f77b4', '#d62728'] 
    
    for i, method in enumerate(methods):
        scores = [data[cls].get(method, {}).get(metric_key, 0.0) for cls in classes]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, scores, width, label=method, color=colors[i], edgecolor='black', linewidth=0.8, zorder=3)
        
        # 顶部标注数值
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=90)
                
    ax.set_ylabel(metric_label, fontweight='bold')
    ax.set_title(f'Ablation Study: {metric_label} Comparison Across Features', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_classes, rotation=0)
    
    # 放置图例
    ax.legend(title='Methods', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, edgecolor='black')
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    if source_file:
        plt.figtext(0.5, -0.05, f"Data Source: {source_file}", ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'academic_{metric_key.lower()}_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Done] Generated: {out_path}")

def plot_metric_improvement_heatmap(data, metric_key, metric_label, output_dir, source_file=""):
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn not found, skipping heatmap (You can install it with `pip install seaborn`).")
        return

    classes = list(data.keys())
    clean_classes = [c.replace('_', ' ').title() for c in classes]
    methods = ['Ablation A (Visual Only)', 'Ablation B (POI Only)', 'Full Fusion'] 
    
    heatmap_data = np.zeros((len(classes), len(methods)))
    
    for i, cls in enumerate(classes):
        base_score = data[cls].get('Baseline', {}).get(metric_key, 0.0)
        for j, method in enumerate(methods):
            method_score = data[cls].get(method, {}).get(metric_key, 0.0)
            heatmap_data[i, j] = method_score - base_score
            
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='+.3f', cmap='RdYlGn', center=0, 
                xticklabels=methods, yticklabels=clean_classes,
                cbar_kws={'label': f'{metric_label} Change ($\\Delta$) vs Baseline'}, ax=ax, linewidths=0.5, linecolor='gray')
    
    ax.set_title(f'Absolute {metric_label} Improvement over Baseline', fontweight='bold', pad=15)
    plt.xticks(rotation=15, ha='right')

    if source_file:
        plt.figtext(0.5, -0.05, f"Data Source: {source_file}", ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'academic_{metric_key.lower()}_delta_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Done] Generated: {out_path}")

def generate_academic_plots(report_path, output_dir=None):
    report_path = os.path.abspath(report_path)
    source_filename = os.path.basename(report_path)
    base_dir = os.path.dirname(report_path)
    if output_dir is None:
        output_dir = os.path.join(base_dir, os.path.splitext(source_filename)[0])
    os.makedirs(output_dir, exist_ok=True)

    print(f"Parsing academic data from: {source_filename}")
    parsed_data = parse_txt(report_path)

    # 循环生成核心指标的可视化图表
    metrics_to_plot = [
        ('P', 'Precision'),
        ('R', 'Recall'),
        ('F1', 'F1-Score'),
        ('ACC', 'Accuracy'),
    ]
    
    for metric_key, metric_label in metrics_to_plot:
        plot_grouped_bar_metric(parsed_data, metric_key, metric_label, output_dir, source_filename)
        plot_metric_improvement_heatmap(parsed_data, metric_key, metric_label, output_dir, source_filename)
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate academic ablation comparison plots from txt reports.')
    parser.add_argument('--txt-report', help='Specific ablation txt report to plot.')
    parser.add_argument('--all', action='store_true', help='Plot all ablation txt reports in ablation_results.')
    args = parser.parse_args()

    search_dir = os.path.dirname(os.path.abspath(__file__))
    ablation_dir = os.path.join(search_dir, 'ablation_results')

    if args.txt_report:
        txt_files = [args.txt_report]
    else:
        txt_files = glob.glob(os.path.join(ablation_dir, 'ablation_*.txt'))
        if not txt_files:
            txt_files = glob.glob(os.path.join(search_dir, 'legacy', 'ablation_*.txt'))
        if not args.all and txt_files:
            txt_files = [max(txt_files, key=os.path.getmtime)]

    if not txt_files:
        print("No text reports found in new_experiment/ablation_results or new_experiment/legacy")
        exit(1)

    for txt_file in txt_files:
        generate_academic_plots(txt_file)

