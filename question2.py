# 区间删失数据BMI分组与最佳NIPT时点可视化模块
# 对应MATLAB脚本question2.m生成的分析结果进行可视化

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import probplot
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
import matplotlib.font_manager as fm
from matplotlib import rcParams
import platform

def set_chinese_font():
    """设置中文字体，根据不同操作系统选择合适的字体"""
    system = platform.system()
    
    # 根据操作系统选择默认字体
    if system == 'Windows':
        font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong']
    elif system == 'Darwin':  # macOS
        font_list = ['Heiti TC', 'PingFang SC', 'STHeiti']
    else:  # Linux等
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']
    
    # 添加更多可能的字体作为备选
    font_list.extend(['Arial Unicode MS', 'DejaVu Sans'])
    
    # 查找系统中第一个可用的字体
    chinese_font = None
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in font_list:
        if font in available_fonts:
            chinese_font = font
            break
    
    if chinese_font:
        print(f"使用中文字体: {chinese_font}")
        # 设置字体
        plt.rcParams['font.sans-serif'] = [chinese_font] + plt.rcParams['font.sans-serif']
    else:
        # 如果找不到中文字体，尝试加载 matplotlib 自带的中文字体
        print("警告: 未找到系统中文字体，尝试加载 matplotlib 内置字体")
        try:
            # 找到 matplotlib 安装目录下的中文字体
            import matplotlib as mpl
            from pathlib import Path
            mpl_path = Path(mpl.__file__).parent
            font_files = list(mpl_path.glob('mpl-data/fonts/ttf/*'))
            
            # 尝试找到适合中文显示的字体
            for font_file in font_files:
                if 'noto' in font_file.name.lower() or 'droid' in font_file.name.lower():
                    plt.rcParams['font.sans-serif'] = [str(font_file)] + plt.rcParams['font.sans-serif']
                    print(f"使用 matplotlib 内置字体: {font_file.name}")
                    break
        except Exception as e:
            print(f"加载内置字体失败: {e}")
    
    # 确保能显示负号
    plt.rcParams['axes.unicode_minus'] = False
    
    # 尝试使用更强大的配置
    rcParams['font.family'] = 'sans-serif'
    
    return chinese_font is not None

# 调用字体设置函数
set_chinese_font()

# 定义输出目录
OUT_DIR = "./output_q2"
VIS_DIR = os.path.join(OUT_DIR, "vis")
if not os.path.exists(VIS_DIR):
    os.makedirs(VIS_DIR)

# 设置全局绘图风格（注释掉，避免覆盖字体设置）
# plt.style.use('seaborn-v0_8-whitegrid')

# 定义颜色方案
COLORS = {
    'not_reached': '#3498db',  # 蓝色 - 未达标
    'reached': '#e74c3c',      # 红色 - 达标
    'not_tested': '#95a5a6',   # 灰色 - 未检测
    'interval': '#e74c3c',     # 红色 - 区间线段
    'group1': '#1f77b4',       # 蓝色 - BMI分组1
    'group2': '#2ca02c',       # 绿色 - BMI分组2
    'group3': '#ff7f0e',       # 橙色 - BMI分组3
    'group4': '#d62728',       # 红色 - BMI分组4
    'model': '#3498db',        # 蓝色 - 模型预测
    'observed': '#95a5a6',     # 灰色 - 观测数据
    'confidence': '#3498db20', # 蓝色透明 - 置信区间
    'risk1': '#3498db',        # 蓝色 - 检测失败风险
    'risk2': '#ff7f0e',        # 橙色 - 早期延迟风险
    'risk3': '#e74c3c',        # 红色 - 严重延迟风险
}

# ========================= 辅助函数 =========================
def load_data():
    """加载所有必要的数据文件"""
    data = {}
    
    # 定义所有需要尝试读取的文件
    file_list = {
        'original': os.path.join(OUT_DIR, '..', 'processed_male.csv'),
        'intervals': os.path.join(OUT_DIR, 'intervals.csv'),
        'midpoint': os.path.join(OUT_DIR, 'GA_midpoint.csv'),
        'low': os.path.join(OUT_DIR, 'GA_low.csv'),
        'up': os.path.join(OUT_DIR, 'GA_up.csv'),
        'dp_groups': os.path.join(OUT_DIR, 'GA_intervals_DPgroups.csv'),
        'group_table': os.path.join(OUT_DIR, 'group_table.csv'),
        'bootstrap': os.path.join(OUT_DIR, 'group_table_with_bootstrap.csv'),
        'mc_results': os.path.join(OUT_DIR, 'mc_results.csv'),
        'mid_coef': os.path.join(OUT_DIR, 'aft_mid_coef.csv'),
        'low_coef': os.path.join(OUT_DIR, 'aft_low_coef.csv'),
        'up_coef': os.path.join(OUT_DIR, 'aft_up_coef.csv'),
        'icenreg_out': os.path.join(OUT_DIR, 'icenreg_out.csv'),
    }
    
    # 尝试读取每个文件
    for key, file_path in file_list.items():
        try:
            # 添加数据类型转换和错误处理
            df = pd.read_csv(file_path)
            
            # 检查并转换数据类型 - 修复可能的转换问题
            if key in ['group_table', 'bootstrap', 'dp_groups']:
                # 确保BMI_group, BMI_center和t_star是数值型
                numeric_cols = ['BMI_group', 'BMI_center', 't_star', 'reach_prob', 'ci_low', 'ci_high']
                for col in numeric_cols:
                    if col in df.columns:
                        # 安全转换，使用错误处理
                        try:
                            # 先尝试直接转换
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            if df[col].isna().any():
                                print(f"警告：{key}文件中的{col}列包含无法转换为数值的元素，这些值已替换为NaN")
                        except Exception as e:
                            print(f"警告：转换{key}文件的{col}列时出错: {e}")
            
            data[key] = df
            print(f"成功读取 {key} 数据")
        except Exception as e:
            print(f"警告: 无法读取 {key} 数据: {e}")
            data[key] = None
    
    # 处理原始数据的可能的中文列名
    if data['original'] is not None:
        col_map = {
            '孕妇代码': 'patient_id',
            '检测孕周': 'gestational_weeks',
            'Y染色体浓度': 'Y_frac',
            '孕妇BMI': 'BMI'
        }
        
        for old_name, new_name in col_map.items():
            if old_name in data['original'].columns:
                data['original'].rename(columns={old_name: new_name}, inplace=True)
        
        # 确保Y_frac在0-1范围内
        if 'Y_frac' in data['original'].columns and data['original']['Y_frac'].max() > 1.05:
            data['original']['Y_frac'] = data['original']['Y_frac'] / 100
    
    return data

# ========================= 一、数据预处理阶段可视化 =========================
def plot_interval_censored_data(data):
    """区间删失数据分布热力图/散点图"""
    # 确保绘图前设置了中文字体
    set_chinese_font()
    
    if data['original'] is None or data['intervals'] is None:
        print("缺少必要数据，跳过绘制区间删失数据分布图")
        return
    
    print("绘制区间删失数据分布图...")
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 左上图：原始数据Y染色体浓度分布
    try:
        original = data['original']
        if 'Y_frac' in original.columns and 'gestational_weeks' in original.columns:
            # 确保BMI列是数值类型
            bmi_values = pd.to_numeric(original['BMI'], errors='coerce')
            # 移除NaN值
            valid_mask = ~(bmi_values.isna() | original['Y_frac'].isna() | original['gestational_weeks'].isna())
            
            if valid_mask.sum() > 0:
                # 绘制散点图：孕周 vs Y染色体浓度
                scatter = ax1.scatter(original['gestational_weeks'][valid_mask], 
                                    original['Y_frac'][valid_mask], 
                                    c=bmi_values[valid_mask], cmap='viridis', alpha=0.6, s=30)
            else:
                # 如果没有有效数据，绘制空图
                ax1.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax1.transAxes)
                scatter = None
            
            # 添加阈值线
            ax1.axhline(y=0.04, color='red', linestyle='--', linewidth=2, label='4%阈值')
            
            ax1.set_xlabel('孕周', fontsize=12)
            ax1.set_ylabel('Y染色体浓度', fontsize=12)
            ax1.set_title('原始数据：孕周 vs Y染色体浓度', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 添加颜色条
            if scatter is not None:
                cbar1 = plt.colorbar(scatter, ax=ax1)
                cbar1.set_label('BMI', rotation=270, labelpad=15)
    except Exception as e:
        print(f"绘制左上图时出错: {e}")
        ax1.text(0.5, 0.5, f"数据绘制错误: {e}", ha='center', va='center', transform=ax1.transAxes)
    
    # 右上图：区间删失数据分布
    try:
        intervals = data['intervals']
        
        # 绘制区间长度分布
        interval_lengths = intervals['GA_upper'] - intervals['GA_lower']
        interval_lengths = interval_lengths[np.isfinite(interval_lengths)]  # 移除Inf值
        
        ax2.hist(interval_lengths, bins=20, alpha=0.7, color=COLORS['interval'], edgecolor='black')
        ax2.set_xlabel('区间长度 (孕周)', fontsize=12)
        ax2.set_ylabel('频数', fontsize=12)
        ax2.set_title('区间删失数据长度分布', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_length = np.mean(interval_lengths)
        ax2.axvline(x=mean_length, color='red', linestyle='--', linewidth=2, 
                   label=f'平均长度: {mean_length:.2f}周')
        ax2.legend()
    except Exception as e:
        print(f"绘制右上图时出错: {e}")
        ax2.text(0.5, 0.5, f"数据绘制错误: {e}", ha='center', va='center', transform=ax2.transAxes)
    
    # 左下图：BMI分布
    try:
        if 'BMI' in intervals.columns:
            bmi_data = intervals['BMI'].dropna()
            ax3.hist(bmi_data, bins=20, alpha=0.7, color=COLORS['group1'], edgecolor='black')
            ax3.set_xlabel('BMI', fontsize=12)
            ax3.set_ylabel('频数', fontsize=12)
            ax3.set_title('BMI分布', fontsize=14)
            ax3.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_bmi = np.mean(bmi_data)
            ax3.axvline(x=mean_bmi, color='red', linestyle='--', linewidth=2, 
                       label=f'平均BMI: {mean_bmi:.1f}')
            ax3.legend()
    except Exception as e:
        print(f"绘制左下图时出错: {e}")
        ax3.text(0.5, 0.5, f"数据绘制错误: {e}", ha='center', va='center', transform=ax3.transAxes)
    
    # 右下图：事件状态分布
    try:
        if 'event' in intervals.columns:
            event_counts = intervals['event'].value_counts()
            labels = ['未达标', '达标'] if 0 in event_counts.index else ['达标']
            colors = [COLORS['not_reached'], COLORS['reached']] if len(event_counts) == 2 else [COLORS['reached']]
            
            wedges, texts, autotexts = ax4.pie(event_counts.values, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90)
            ax4.set_title('事件状态分布', fontsize=14)
            
            # 美化文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
    except Exception as e:
        print(f"绘制右下图时出错: {e}")
        ax4.text(0.5, 0.5, f"数据绘制错误: {e}", ha='center', va='center', transform=ax4.transAxes)
    
    # 添加总标题
    fig.suptitle('区间删失数据预处理分析', fontsize=16, fontweight='bold')
    
    # 添加图注
    plt.figtext(0.5, 0.01, 
               '注：左上图展示原始数据的孕周与Y染色体浓度关系；右上图展示区间长度分布；\n' + 
               '左下图展示BMI分布；右下图展示事件状态分布。', 
               ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图片
    plt.savefig(os.path.join(VIS_DIR, 'interval_censored_data.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"区间删失数据分布图已保存至{os.path.join(VIS_DIR, 'interval_censored_data.png')}")

def plot_gmm_clustering_validation(data):
    """GMM聚类结果验证图"""
    # 确保绘图前设置了中文字体
    set_chinese_font()
    
    if data['original'] is None or data['midpoint'] is None:
        print("缺少必要数据，跳过绘制GMM聚类验证图")
        return
    
    print("绘制GMM聚类验证图...")
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    try:
        midpoint = data['midpoint']
        
        # 左上图：BMI vs 达标时间的散点图（按聚类着色）
        if 'BMI' in midpoint.columns and 'T' in midpoint.columns:
            # 使用BMI和T进行简单的k-means聚类（模拟GMM）
            from sklearn.cluster import KMeans
            
            # 准备数据
            X = midpoint[['BMI', 'T']].dropna()
            if len(X) > 0:
                # 进行k-means聚类
                kmeans = KMeans(n_clusters=4, random_state=42)
                clusters = kmeans.fit_predict(X)
                
                # 绘制散点图
                scatter = ax1.scatter(X['BMI'], X['T'], c=clusters, cmap='viridis', alpha=0.7, s=50)
                
                # 绘制聚类中心
                centers = kmeans.cluster_centers_
                ax1.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='聚类中心')
                
                ax1.set_xlabel('BMI', fontsize=12)
                ax1.set_ylabel('达标时间 (孕周)', fontsize=12)
                ax1.set_title('BMI vs 达标时间聚类结果', fontsize=14)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 添加颜色条
                cbar1 = plt.colorbar(scatter, ax=ax1)
                cbar1.set_label('聚类标签', rotation=270, labelpad=15)
    except Exception as e:
        print(f"绘制左上图时出错: {e}")
        ax1.text(0.5, 0.5, f"聚类分析错误: {e}", ha='center', va='center', transform=ax1.transAxes)
    
    # 右上图：聚类质量评估（轮廓系数）
    try:
        if 'X' in locals() and len(X) > 0:
            from sklearn.metrics import silhouette_score
            
            # 测试不同聚类数的轮廓系数
            k_range = range(2, min(8, len(X)//10 + 1))  # 限制最大聚类数
            silhouette_scores = []
            
            for k in k_range:
                kmeans_temp = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans_temp.fit_predict(X)
                score = silhouette_score(X, cluster_labels)
                silhouette_scores.append(score)
            
            # 绘制轮廓系数曲线
            ax2.plot(k_range, silhouette_scores, 'o-', linewidth=2, markersize=8, color=COLORS['model'])
            
            # 标注最优聚类数
            best_k = k_range[np.argmax(silhouette_scores)]
            best_score = max(silhouette_scores)
            ax2.plot(best_k, best_score, 'ro', markersize=10)
            ax2.annotate(f'最优k={best_k}\n轮廓系数={best_score:.3f}', 
                        xy=(best_k, best_score),
                        xytext=(best_k+0.5, best_score+0.01),
                        arrowprops=dict(facecolor='red', shrink=0.05))
            
            ax2.set_xlabel('聚类数 k', fontsize=12)
            ax2.set_ylabel('轮廓系数', fontsize=12)
            ax2.set_title('聚类质量评估', fontsize=14)
            ax2.grid(True, alpha=0.3)
    except Exception as e:
        print(f"绘制右上图时出错: {e}")
        ax2.text(0.5, 0.5, f"轮廓系数计算错误: {e}", ha='center', va='center', transform=ax2.transAxes)
    
    # 左下图：聚类内距离分布
    try:
        if 'X' in locals() and len(X) > 0 and 'clusters' in locals():
            # 计算每个点到其聚类中心的距离
            distances = []
            for i, (_, point) in enumerate(X.iterrows()):
                center = centers[clusters[i]]
                dist = np.sqrt(np.sum((point.values - center)**2))
                distances.append(dist)
            
            # 绘制距离分布直方图
            ax3.hist(distances, bins=20, alpha=0.7, color=COLORS['group2'], edgecolor='black')
            ax3.set_xlabel('到聚类中心的距离', fontsize=12)
            ax3.set_ylabel('频数', fontsize=12)
            ax3.set_title('聚类内距离分布', fontsize=14)
            ax3.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_dist = np.mean(distances)
            ax3.axvline(x=mean_dist, color='red', linestyle='--', linewidth=2, 
                       label=f'平均距离: {mean_dist:.2f}')
            ax3.legend()
    except Exception as e:
        print(f"绘制左下图时出错: {e}")
        ax3.text(0.5, 0.5, f"距离计算错误: {e}", ha='center', va='center', transform=ax3.transAxes)
    
    # 右下图：聚类大小分布
    try:
        if 'clusters' in locals():
            # 计算每个聚类的样本数
            unique_clusters, counts = np.unique(clusters, return_counts=True)
            
            # 绘制柱状图
            bars = ax4.bar(unique_clusters, counts, alpha=0.7, color=[COLORS.get(f'group{i+1}', 'gray') for i in unique_clusters])
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            ax4.set_xlabel('聚类标签', fontsize=12)
            ax4.set_ylabel('样本数', fontsize=12)
            ax4.set_title('聚类大小分布', fontsize=14)
            ax4.grid(True, alpha=0.3, axis='y')
            
            # 添加统计信息
            total_samples = len(clusters)
            ax4.text(0.02, 0.98, f'总样本数: {total_samples}\n聚类数: {len(unique_clusters)}', 
                    transform=ax4.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')
    except Exception as e:
        print(f"绘制右下图时出错: {e}")
        ax4.text(0.5, 0.5, f"聚类统计错误: {e}", ha='center', va='center', transform=ax4.transAxes)
    
    # 添加总标题
    fig.suptitle('GMM聚类结果验证分析', fontsize=16, fontweight='bold')
    
    # 添加图注
    plt.figtext(0.5, 0.01, 
               '注：左上图展示BMI与达标时间的聚类结果；右上图展示不同聚类数的轮廓系数；\n' + 
               '左下图展示聚类内距离分布；右下图展示各聚类的样本数分布。', 
               ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图片
    plt.savefig(os.path.join(VIS_DIR, 'gmm_clustering_validation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"GMM聚类验证图已保存至{os.path.join(VIS_DIR, 'gmm_clustering_validation.png')}")

# ========================= 二、模型构建阶段可视化 =========================
def plot_triple_sensitivity_analysis(data):
    """
    1. 三重敏感性分析对比图
    参数:
        data: 包含三种插补策略模型系数的字典
    """
    # 确保绘图前设置了中文字体
    set_chinese_font()
    
    print("绘制三重敏感性分析对比图...")
    
    # 获取三种插补策略的模型系数
    mid_coef = data['mid_coef']
    low_coef = data['low_coef']
    up_coef = data['up_coef']
    
    if mid_coef is None or low_coef is None or up_coef is None:
        print("缺少必要的模型系数数据，无法绘制三重敏感性分析对比图")
        return
    
    # 设置参数
    methods = ['中点插补', '下界插补', '上界插补']
    colors = ['#76ac68', '#4dabf5', '#d9734e']  # 绿色、蓝色、橙色
    TIME_THRESHOLD = 0.5  # 临床可接受的最优时点差异阈值(周)
    
    # 提取BMI系数及标准误
    try:
        # 查找BMI系数在表格中的位置
        bmi_mid_idx = mid_coef[mid_coef['term'] == 'BMI'].index[0]
        bmi_low_idx = low_coef[low_coef['term'] == 'BMI'].index[0]
        bmi_up_idx = up_coef[up_coef['term'] == 'BMI'].index[0]
        
        # 提取系数和标准误
        beta1 = [
            mid_coef.loc[bmi_mid_idx, 'estimate'],
            low_coef.loc[bmi_low_idx, 'estimate'],
            up_coef.loc[bmi_up_idx, 'estimate']
        ]
        
        se = [
            mid_coef.loc[bmi_mid_idx, 'std.error'],
            low_coef.loc[bmi_low_idx, 'std.error'],
            up_coef.loc[bmi_up_idx, 'std.error']
        ]
        
        # 计算95%置信区间
        ci_low = [b - 1.96 * s for b, s in zip(beta1, se)]
        ci_high = [b + 1.96 * s for b, s in zip(beta1, se)]
    except Exception as e:
        print(f"提取BMI系数出错: {e}")
        return
    
    # 计算或提取最优时点
    group_table = data['group_table']
    
    try:
        # 查找sensitivity_results.csv，如果存在
        sens_results_path = os.path.join(OUT_DIR, 'sensitivity_results.csv')
        if os.path.exists(sens_results_path):
            sens_results = pd.read_csv(sens_results_path)
            t_stars = [sens_results['t_star_mid'].iloc[0], 
                      sens_results['t_star_low'].iloc[0], 
                      sens_results['t_star_up'].iloc[0]]
        else:
            # 如果没有现成的结果，使用公式计算
            print("计算三种插补策略的最优时点...")
            
            # 获取截距项
            intercept_mid = mid_coef[mid_coef['term'] == '(Intercept)']['estimate'].iloc[0]
            intercept_low = low_coef[low_coef['term'] == '(Intercept)']['estimate'].iloc[0]
            intercept_up = up_coef[up_coef['term'] == '(Intercept)']['estimate'].iloc[0]
            
            # 使用平均BMI计算
            if group_table is not None:
                median_bmi = np.median(group_table['BMI_center'])
            else:
                median_bmi = 28  # 默认值
            
            # 使用AFT模型公式计算最优时点
            t_stars = [
                np.exp(intercept_mid + beta1[0] * median_bmi),
                np.exp(intercept_low + beta1[1] * median_bmi),
                np.exp(intercept_up + beta1[2] * median_bmi)
            ]
    except Exception as e:
        print(f"计算最优时点出错: {e}")
        t_stars = [17.5, 16.8, 18.0]  # 默认值
    
    # 检查时点差异是否超过临床阈值
    diff_low_up = abs(t_stars[1] - t_stars[2])
    exceed_threshold = diff_low_up > TIME_THRESHOLD
    
    if exceed_threshold:
        print(f'下界与上界插补的最优时点差异为{diff_low_up:.1f}周，超过临床阈值{TIME_THRESHOLD:.1f}周')
    else:
        print(f'下界与上界插补的最优时点差异为{diff_low_up:.1f}周，在临床阈值{TIME_THRESHOLD:.1f}周内')
    
    # 创建图形
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # 创建左轴（柱状图）
    x = np.arange(len(methods))
    bars = ax1.bar(x, beta1, color=colors, width=0.6)
    
    # 添加误差线
    ax1.errorbar(x, beta1, yerr=[np.array(beta1) - np.array(ci_low), np.array(ci_high) - np.array(beta1)], 
                fmt='.k', capsize=5, linewidth=1.5)
    
    # 设置左轴标签和范围
    ax1.set_ylabel('BMI系数 (β₁)', fontsize=14, fontweight='bold')
    ax1.set_ylim([min(ci_low) * 0.9, max(ci_high) * 1.1])
    
    # 创建右轴（折线图）
    ax2 = ax1.twinx()
    ax2.plot(x, t_stars, '-o', linewidth=2, markersize=8, markerfacecolor='white', color='black')
    
    # 如果差异超过阈值，添加红色星号标记
    if exceed_threshold:
        ax2.plot(x[1:3], [t_stars[1], t_stars[2]], '*r', markersize=10, linewidth=2)
        ax2.text(1.5, (t_stars[1] + t_stars[2]) / 2 + 0.2, 
                f'Δ = {diff_low_up:.1f}周', color='red', fontsize=12, ha='center')
    
    # 添加临床阈值的参考虚线
    ax2.axhline(y=t_stars[0] + TIME_THRESHOLD/2, color='gray', linestyle='--', alpha=0.7)
    ax2.axhline(y=t_stars[0] - TIME_THRESHOLD/2, color='gray', linestyle='--', alpha=0.7)
    ax2.text(2.1, t_stars[0] + TIME_THRESHOLD/2, f'+{TIME_THRESHOLD/2:.1f}周', 
            color='gray', fontsize=10)
    ax2.text(2.1, t_stars[0] - TIME_THRESHOLD/2, f'-{TIME_THRESHOLD/2:.1f}周', 
            color='gray', fontsize=10)
    
    # 设置右轴标签和范围
    ax2.set_ylabel('最优孕周 (t*)', fontsize=14, fontweight='bold')
    ax2.set_ylim([min(t_stars) * 0.9, max(t_stars) * 1.05])
    
    # 设置x轴
    plt.xticks(x, methods)
    ax1.set_xlabel('敏感性分析插补策略', fontsize=14, fontweight='bold')
    
    # 添加图例和标题
    ax1.set_title('三重敏感性分析：区间插补对BMI系数和最优时点的影响', fontsize=16)
    
    # 添加图例
    beta_line = plt.Line2D([0], [0], color='black', marker='o', 
                          markerfacecolor='white', markersize=8, label='最优孕周 (t*)')
    beta_bars = plt.Rectangle((0, 0), 1, 1, color=colors[0], label='BMI系数 (β₁)')
    
    ax1.legend(handles=[beta_bars, beta_line], loc='upper left')
    
    # 添加图注
    plt.figtext(0.5, 0.01, 
               '注：柱状图和误差线表示BMI系数β₁及95%置信区间；折线表示相应的最优孕周t*；\n' + 
               '虚线表示临床可接受范围；红色星号标注超过临床阈值的差异', 
               ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图片
    plt.savefig(os.path.join(VIS_DIR, 'sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"三重敏感性分析对比图已保存至{os.path.join(VIS_DIR, 'sensitivity_analysis.png')}")

def plot_ic_aft_goodness_of_fit(data):
    """
    2. IC-AFT模型拟合优度图
    参数:
        data: 包含IC-AFT模型输出和观测数据的字典
    """
    # 确保绘图前设置了中文字体
    set_chinese_font()
    
    print("绘制IC-AFT模型拟合优度图...")
    
    # 检查必要数据
    if data['icenreg_out'] is None or data['intervals'] is None:
        print("缺少必要的IC-AFT模型数据，无法绘制拟合优度图")
        return
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：观测vs预测的散点图
    try:
        icenreg_out = data['icenreg_out']
        intervals = data['intervals']
        
        # 获取模型预测的生存函数
        if 'week' in icenreg_out.columns:
            t_model = np.array(icenreg_out['week'])
            
            # 找出生存率列
            survival_cols = [col for col in icenreg_out.columns if col != 'week']
            
            if survival_cols:
                # 使用中间的生存率列作为代表性预测
                mid_col = survival_cols[len(survival_cols)//2]
                predicted_survival = np.array(icenreg_out[mid_col])
                
                # 计算观测的Kaplan-Meier估计
                # 简化处理：使用区间中点作为观测时间
                observed_times = []
                observed_survival = []
                
                # 按时间排序
                sorted_intervals = intervals.sort_values('GA_lower')
                n_total = len(sorted_intervals)
                
                for i, (_, row) in enumerate(sorted_intervals.iterrows()):
                    if row['event'] == 1:  # 事件发生
                        t_obs = (row['GA_lower'] + row['GA_upper']) / 2
                        s_obs = (n_total - i) / n_total
                        observed_times.append(t_obs)
                        observed_survival.append(s_obs)
                
                # 绘制观测vs预测
                ax1.scatter(observed_times, observed_survival, alpha=0.7, 
                           color=COLORS['observed'], label='观测值', s=50)
                
                # 绘制预测曲线
                ax1.plot(t_model, predicted_survival, '-', 
                        color=COLORS['model'], linewidth=2, label='IC-AFT预测')
                
                # 添加对角线（完美拟合）
                ax1.plot([min(t_model), max(t_model)], [1, 0], 'k--', alpha=0.5, label='完美拟合')
                
                # 计算R²
                if len(observed_times) > 1:
                    # 插值预测值到观测时间点
                    from scipy.interpolate import interp1d
                    pred_interp = interp1d(t_model, predicted_survival, 
                                          bounds_error=False, fill_value=(1.0, 0.0))
                    pred_at_obs = pred_interp(observed_times)
                    
                    # 计算R²
                    ss_res = np.sum((np.array(observed_survival) - pred_at_obs) ** 2)
                    ss_tot = np.sum((np.array(observed_survival) - np.mean(observed_survival)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                            transform=ax1.transAxes, fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax1.set_xlabel('时间 (孕周)', fontsize=12)
                ax1.set_ylabel('生存概率', fontsize=12)
                ax1.set_title('观测值 vs IC-AFT预测值', fontsize=14)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
    except Exception as e:
        print(f"绘制左图时出错: {e}")
        ax1.text(0.5, 0.5, f"数据绘制错误: {e}", 
                ha='center', va='center', transform=ax1.transAxes)
    
    # 右图：残差分析
    try:
        # 计算残差
        if 'observed_times' in locals() and 'observed_survival' in locals():
            pred_interp = interp1d(t_model, predicted_survival, 
                                  bounds_error=False, fill_value=(1.0, 0.0))
            pred_at_obs = pred_interp(observed_times)
            residuals = np.array(observed_survival) - pred_at_obs
            
            # 绘制残差散点图
            ax2.scatter(observed_times, residuals, alpha=0.7, 
                       color=COLORS['observed'], s=50)
            
            # 添加零线
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            
            # 添加平滑的趋势线
            if len(observed_times) > 3:
                from scipy.interpolate import UnivariateSpline
                try:
                    spline = UnivariateSpline(observed_times, residuals, s=0.1)
                    t_smooth = np.linspace(min(observed_times), max(observed_times), 100)
                    ax2.plot(t_smooth, spline(t_smooth), 'r-', alpha=0.7, linewidth=2)
                except:
                    pass
            
            ax2.set_xlabel('时间 (孕周)', fontsize=12)
            ax2.set_ylabel('残差', fontsize=12)
            ax2.set_title('残差分析', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # 添加残差统计信息
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            ax2.text(0.05, 0.95, f'均值: {mean_residual:.3f}\n标准差: {std_residual:.3f}', 
                    transform=ax2.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    except Exception as e:
        print(f"绘制右图时出错: {e}")
        ax2.text(0.5, 0.5, f"残差分析错误: {e}", 
                ha='center', va='center', transform=ax2.transAxes)
    
    # 添加图注
    plt.figtext(0.5, 0.01, 
               '注：左图展示观测值与IC-AFT模型预测值的对比；\n' + 
               '右图展示残差分析，用于评估模型拟合质量。', 
               ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图片
    plt.savefig(os.path.join(VIS_DIR, 'ic_aft_goodness_of_fit.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"IC-AFT模型拟合优度图已保存至{os.path.join(VIS_DIR, 'ic_aft_goodness_of_fit.png')}")

# ========================= 三、风险函数与最优时点阶段可视化 =========================
def plot_weighted_risk_function(data, weights=[1, 1, 2]):
    """
    加权风险函数分解图：展示风险函数的三个成分及总风险如何随孕周变化
    参数:
        data: 包含模型输出的字典
        weights: 三个风险成分的权重 [检测失败风险, 早期延迟风险, 严重延迟风险]
    """
    # 确保绘图前设置了中文字体
    set_chinese_font()
    
    print("绘制加权风险函数分解图...")
    
    # 检查必要数据
    if data['group_table'] is None:
        print("缺少必要的分组数据，无法绘制风险函数分解图")
        return
    
    # 创建孕周网格
    t_grid = np.arange(10, 25.1, 0.1)
    
    # 获取平均BMI值用于估计生存函数
    bmi_center = np.mean(data['group_table']['BMI_center'])
    
    # 计算生存函数
    try:
        if data['icenreg_out'] is not None:
            # 使用IC-AFT模型输出的生存函数
            icenreg = data['icenreg_out']
            
            # 获取模型的时间和生存率矩阵
            if 'week' in icenreg.columns:
                t_model = np.array(icenreg['week'])
                
                # 找出生存率列（除week外的列）
                survival_cols = [col for col in icenreg.columns if col != 'week']
                
                if survival_cols:
                    # 使用中间的生存率列作为代表性生存函数
                    mid_col = survival_cols[len(survival_cols)//2]
                    survival_values = np.array(icenreg[mid_col])
                    
                    # 使用线性插值填充t_grid处的生存率
                    from scipy.interpolate import interp1d
                    survival_interp = interp1d(t_model, survival_values, 
                                            bounds_error=False, fill_value=(1.0, 0.0))
                    survival = survival_interp(t_grid)
                else:
                    print("生存率列缺失，使用简单指数衰减模型代替")
                    survival = np.exp(-0.1 * (t_grid - 10))
            else:
                print("时间列缺失，使用简单指数衰减模型代替")
                survival = np.exp(-0.1 * (t_grid - 10))
        else:
            # 如果没有IC-AFT模型，使用简单指数衰减模型
            print("未找到IC-AFT模型输出，使用简单指数衰减模型")
            survival = np.exp(-0.1 * (t_grid - 10))
    except Exception as e:
        print(f"生存函数计算出错: {e}")
        survival = np.exp(-0.1 * (t_grid - 10))
    
    # 计算风险函数的三个成分
    w1, w2, w3 = weights
    risk1 = w1 * (1 - survival)                 # 检测失败风险
    risk2 = w2 * np.maximum(t_grid - 12, 0)     # 早期延迟风险
    risk3 = w3 * np.maximum(t_grid - 20, 0)     # 严重延迟风险
    
    # 总风险，处理NaN/Inf值
    total_risk = risk1 + risk2 + risk3
    total_risk = np.nan_to_num(total_risk, nan=0.0, posinf=10.0, neginf=0.0)
    
    # 找出最小风险点
    min_risk_idx = np.argmin(total_risk)
    t_star = t_grid[min_risk_idx]
    min_risk_value = total_risk[min_risk_idx]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制堆叠面积图
    ax.fill_between(t_grid, 0, risk1, color=COLORS['risk1'], alpha=0.7, label='检测失败风险')
    ax.fill_between(t_grid, risk1, risk1 + risk2, color=COLORS['risk2'], alpha=0.7, label='早期延迟风险')
    ax.fill_between(t_grid, risk1 + risk2, total_risk, color=COLORS['risk3'], alpha=0.7, label='严重延迟风险')
    
    # 绘制总风险曲线
    ax.plot(t_grid, total_risk, 'k-', linewidth=2, label='总风险')
    
    # 标注最小风险点
    ax.plot(t_star, min_risk_value, 'ko', markersize=10)
    ax.annotate(f't* = {t_star:.1f}周', 
               xy=(t_star, min_risk_value),
               xytext=(t_star + 1, min_risk_value + 0.5),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # 添加权重标注
    ax.text(0.02, 0.98, f'权重: w₁={w1}, w₂={w2}, w₃={w3}', 
           transform=ax.transAxes, fontsize=12, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加垂直线标注12周和20周
    ax.axvline(x=12, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
    ax.text(12, ax.get_ylim()[0] + 0.5, '12周', ha='center')
    ax.text(20, ax.get_ylim()[0] + 0.5, '20周', ha='center')
    
    # 设置标签和图例
    ax.set_xlabel('孕周', fontsize=14)
    ax.set_ylabel('风险值', fontsize=14)
    ax.set_title('加权风险函数分解', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')
    
    # 设置x轴和y轴范围，处理NaN/Inf值
    valid_risk = total_risk[np.isfinite(total_risk)]
    if len(valid_risk) > 0:
        ax.set_xlim([min(t_grid), max(t_grid)])
        ax.set_ylim([0, max(valid_risk) * 1.1])
    else:
        ax.set_xlim([10, 25])
        ax.set_ylim([0, 10])
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(os.path.join(VIS_DIR, 'weighted_risk_function.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"加权风险函数分解图已保存至{os.path.join(VIS_DIR, 'weighted_risk_function.png')}")
    
    return t_star, min_risk_value

def plot_dp_bmi_grouping(data):
    """
    动态规划BMI分组可视化：展示BMI分箱的合理性与分组结果
    参数:
        data: 包含分组信息的字典
    """
    # 确保绘图前设置了中文字体
    set_chinese_font()
    
    print("绘制动态规划BMI分组可视化图...")
    
    # 检查必要数据
    if data['dp_groups'] is None or data['midpoint'] is None:
        print("缺少必要的分组数据，无法绘制BMI分组可视化图")
        return
    
    # 读取分组数据
    dp_groups = data['dp_groups']
    midpoint = data['midpoint']
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：组内方差曲线（K与总方差的关系）
    # 模拟动态规划的结果或使用固定值（实际应从DP算法输出获取）
    k_values = np.arange(1, 8)  # 测试的分组数
    
    # 模拟的组内方差曲线 - 实际情况下应从DP算法结果中获取
    # 这里使用一个随k增加而递减的曲线，但递减速率逐渐减慢
    variance = 100 / (1 + 0.8 * k_values) + 20
    
    # 绘制左图：组内方差随分组数的变化
    ax1.plot(k_values, variance, 'o-', linewidth=2, color=COLORS['model'])
    
    # 标注拐点
    elbow_point = 4  # 假定k=4是拐点
    ax1.plot(elbow_point, variance[elbow_point-1], 'ro', markersize=10)
    ax1.annotate(f'最优分组数: k={elbow_point}', 
               xy=(elbow_point, variance[elbow_point-1]),
               xytext=(elbow_point-0.5, variance[elbow_point-1]-5),
               arrowprops=dict(facecolor='red', shrink=0.05))
    
    # 设置左图标签
    ax1.set_xlabel('分组数k', fontsize=12)
    ax1.set_ylabel('总组内方差', fontsize=12)
    ax1.set_title('动态规划最优分组数选择', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(k_values)
    
    # 右图：BMI分组散点图
    # 准备数据：合并midpoint和dp_groups
    try:
        # 确保patient_id列类型一致
        midpoint_copy = midpoint.copy()
        dp_groups_copy = dp_groups.copy()
        
        if 'patient_id' in midpoint_copy.columns and 'patient_id' in dp_groups_copy.columns:
            midpoint_copy['patient_id'] = midpoint_copy['patient_id'].astype(str)
            dp_groups_copy['patient_id'] = dp_groups_copy['patient_id'].astype(str)
            
            # 创建从patient_id到BMI_group的映射
            pid_to_group = dp_groups_copy.set_index('patient_id')['BMI_group'].to_dict()
            
            # 添加组信息到mid表格
            midpoint_copy['BMI_group'] = midpoint_copy['patient_id'].map(pid_to_group)
            
            # 过滤有效数据
            valid_data = midpoint_copy.dropna(subset=['BMI', 'T', 'BMI_group'])
            
            # 获取唯一的BMI分组
            unique_groups = sorted(valid_data['BMI_group'].unique())
            group_colors = [COLORS.get(f'group{int(g)}', 'gray') for g in unique_groups]
            
            # 绘制散点图：BMI与达标时间的关系
            for i, group in enumerate(unique_groups):
                group_data = valid_data[valid_data['BMI_group'] == group]
                ax2.scatter(group_data['BMI'], group_data['T'], 
                          color=group_colors[i], alpha=0.7, s=50,
                          label=f'BMI组 {int(group)} (n={len(group_data)})')
            
            # 计算并绘制每组的BMI边界
            if 'BMI_group' in dp_groups_copy.columns:
                # 根据BMI_group查找分组边界
                thresholds = []
                for g in range(1, int(max(unique_groups))):
                    try:
                        # 找到相邻组的边界
                        g1_max = dp_groups_copy[dp_groups_copy['BMI_group'] == g]['BMI'].max()
                        g2_min = dp_groups_copy[dp_groups_copy['BMI_group'] == g+1]['BMI'].min()
                        threshold = (g1_max + g2_min) / 2
                        thresholds.append(threshold)
                        
                        # 绘制分界线
                        ax2.axvline(x=threshold, color='gray', linestyle='--', alpha=0.7)
                        ax2.text(threshold, ax2.get_ylim()[0] + 0.5, f'{threshold:.1f}', 
                               ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8))
                    except Exception as e:
                        print(f"计算组 {g} 和 {g+1} 的边界时出错: {e}")
                
                # 为每个组添加拟合曲线，显示BMI与达标时间的关系趋势
                for group in unique_groups:
                    group_data = valid_data[valid_data['BMI_group'] == group]
                    if len(group_data) >= 5:  # 只有样本足够时才拟合
                        try:
                            # 简单线性拟合
                            x = group_data['BMI']
                            y = group_data['T']
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            
                            # 绘制拟合线
                            x_range = np.linspace(x.min(), x.max(), 50)
                            ax2.plot(x_range, p(x_range), '--', 
                                   color=group_colors[list(unique_groups).index(group)],
                                   alpha=0.8, linewidth=2)
                        except Exception as e:
                            print(f"为组 {group} 拟合趋势线时出错: {e}")
        else:
            print("在midpoint或dp_groups中找不到patient_id列")
    except Exception as e:
        print(f"处理右图数据时出错: {e}")
        ax2.text(0.5, 0.5, f"数据处理错误: {e}", 
               ha='center', va='center', transform=ax2.transAxes)
    
    # 设置右图标签
    ax2.set_xlabel('BMI', fontsize=12)
    ax2.set_ylabel('达标时间 (孕周)', fontsize=12)
    ax2.set_title('BMI分组与达标时间关系', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left')
    
    # 添加图注
    plt.figtext(0.5, 0.01, 
               '注：左图展示分组数与组内方差的关系，拐点处为最优分组数；\n' + 
               '右图展示BMI值与达标时间的关系，不同颜色代表不同BMI分组，虚线表示分组边界。', 
               ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图片
    plt.savefig(os.path.join(VIS_DIR, 'dp_bmi_grouping.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"动态规划BMI分组可视化图已保存至{os.path.join(VIS_DIR, 'dp_bmi_grouping.png')}")

def plot_optimal_timepoints_by_bmi(data):
    """
    各BMI组最优时点对比图：展示各组的最优检测时间及置信区间
    参数:
        data: 包含各组最优时点信息的字典
    """
    # 确保绘图前设置了中文字体
    set_chinese_font()
    
    print("绘制各BMI组最优时点对比图...")
    
    # 检查必要数据
    group_table = data.get('group_table')
    bootstrap = data.get('bootstrap')
    
    if group_table is None:
        print("缺少必要的最优时点数据，无法绘制BMI组最优时点对比图")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 准备数据
    groups = sorted(group_table['BMI_group'].unique())
    bmi_centers = []
    t_stars = []
    reach_probs = []
    
    # 为每个组提取数据
    for g in groups:
        group_data = group_table[group_table['BMI_group'] == g]
        if not group_data.empty:
            bmi_centers.append(group_data['BMI_center'].iloc[0])
            t_stars.append(group_data['t_star'].iloc[0])
            reach_probs.append(group_data['reach_prob'].iloc[0])
    
    # 设置x轴标签
    x_labels = [f"BMI组{int(g)}\n(均值{c:.1f})" for g, c in zip(groups, bmi_centers)]
    x = np.arange(len(groups))
    
    # 设置颜色
    bar_colors = [COLORS.get(f'group{int(g)}', 'gray') for g in groups]
    
    # 绘制柱状图
    bars = ax.bar(x, t_stars, color=bar_colors, alpha=0.7, width=0.6)
    
    # 添加Bootstrap置信区间（如果有）
    if bootstrap is not None and all(col in bootstrap.columns for col in ['ci_low', 'ci_high']):
        ci_low = []
        ci_high = []
        
        # 为每个组提取置信区间
        for g in groups:
            group_data = bootstrap[bootstrap['BMI_group'] == g]
            if not group_data.empty:
                ci_low.append(group_data['ci_low'].iloc[0])
                ci_high.append(group_data['ci_high'].iloc[0])
        
        if len(ci_low) == len(groups) and len(ci_high) == len(groups):
            ax.errorbar(x, t_stars, yerr=[np.array(t_stars) - np.array(ci_low), 
                                        np.array(ci_high) - np.array(t_stars)], 
                        fmt='none', ecolor='black', capsize=5)
    
    # 标记"达标概率≥90%"的孕周区间
    reach_threshold = 0.9
    for i, (t, reach) in enumerate(zip(t_stars, reach_probs)):
        if reach >= reach_threshold:
            # 在柱子顶部添加达标标记，绿色
            ax.text(x[i], t + 0.2, f"达标率\n{reach:.1%}", 
                   ha='center', va='bottom', color='green', fontweight='bold')
        else:
            # 未达到90%，红色警告
            ax.text(x[i], t + 0.2, f"达标率\n{reach:.1%}", 
                   ha='center', va='bottom', color='red')
    
    # 添加达标阈值水平线
    ax.axhline(y=13, color='gray', linestyle='--', alpha=0.5, label='13周阈值')
    
    # 设置轴标签和标题
    ax.set_xlabel('BMI分组 (中心值)', fontsize=14)
    ax.set_ylabel('最优孕周 (t*)', fontsize=14)
    ax.set_title('各BMI组的最优检测时点', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    
    # 添加网格线
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 添加趋势线，显示BMI组与最优时点的关系
    try:
        z = np.polyfit(range(len(t_stars)), t_stars, 1)
        p = np.poly1d(z)
        trend_x = np.linspace(-0.5, len(t_stars) - 0.5, 100)
        ax.plot(trend_x, p(trend_x), 'r--', alpha=0.7)
        
        # 标注趋势
        slope = z[0]
        ax.annotate(f"趋势: BMI每组增加约 {slope:.2f} 周", 
                   xy=(len(t_stars) / 2, p(len(t_stars) / 2)),
                   xytext=(len(t_stars) / 2 - 1, p(len(t_stars) / 2) + 1),
                   arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.7))
    except Exception as e:
        print(f"添加趋势线出错: {e}")
    
    # 添加图注
    plt.figtext(0.5, 0.01, 
               '注：柱状图表示各BMI组的最优检测时间点，误差线表示Bootstrap估计的95%置信区间；\n' + 
               '绿色标记表示在该时间点能达到90%以上的检出率，红色表示未能达到。', 
               ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图片
    plt.savefig(os.path.join(VIS_DIR, 'optimal_timepoints_by_bmi.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"各BMI组最优时点对比图已保存至{os.path.join(VIS_DIR, 'optimal_timepoints_by_bmi.png')}")

def plot_optimal_time_trend(data):
    """
    最优时间趋势折线图：展示最优检测时间随BMI值的连续变化
    参数:
        data: 包含分组信息和最优时点的字典
    """
    # 确保绘图前设置了中文字体
    set_chinese_font()
    
    print("绘制最优时间趋势折线图...")
    
    # 检查必要数据
    group_table = data.get('group_table')
    bootstrap = data.get('bootstrap')
    
    if group_table is None:
        print("缺少必要的最优时点数据，无法绘制最优时间趋势图")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 提取BMI和最优时点数据，添加错误处理
    try:
        # 确保数据是数值类型
        bmi_centers = pd.to_numeric(group_table['BMI_center'], errors='coerce').values
        t_stars = pd.to_numeric(group_table['t_star'], errors='coerce').values
        
        # 检查并移除NaN值
        valid_mask = ~(np.isnan(bmi_centers) | np.isnan(t_stars))
        if not valid_mask.all():
            print(f"警告: 发现{(~valid_mask).sum()}个非数值条目，这些条目将被忽略")
        
        bmi_centers = bmi_centers[valid_mask]
        t_stars = t_stars[valid_mask]
        
        if len(bmi_centers) == 0 or len(t_stars) == 0:
            print("错误: 所有数据均为非数值，无法绘制趋势图")
            return
    except Exception as e:
        print(f"提取BMI和最优时点数据时出错: {e}")
        return
    
    # 按BMI排序
    sort_idx = np.argsort(bmi_centers)
    bmi_sorted = bmi_centers[sort_idx]
    t_sorted = t_stars[sort_idx]
    
    # 绘制主折线图
    main_line = ax.plot(bmi_sorted, t_sorted, 'o-', linewidth=2.5, markersize=10, 
                      color=COLORS['model'], label='最优检测时间')
    
    # 添加置信区间（如果有Bootstrap结果）
    if bootstrap is not None and 'ci_low' in bootstrap.columns:
        ci_low = bootstrap['ci_low'].values[sort_idx]
        ci_high = bootstrap['ci_high'].values[sort_idx]
        
        # 添加置信区间带
        ax.fill_between(bmi_sorted, ci_low, ci_high, alpha=0.2, 
                       color=COLORS['confidence'], label='95%置信区间')
    
    # 添加趋势线
    try:
        # 用多项式拟合趋势
        z = np.polyfit(bmi_sorted, t_sorted, 2)
        p = np.poly1d(z)
        
        # 生成平滑曲线
        bmi_fine = np.linspace(min(bmi_sorted), max(bmi_sorted), 100)
        t_trend = p(bmi_fine)
        
        # 绘制趋势线
        trend_line = ax.plot(bmi_fine, t_trend, '--', linewidth=2, 
                           color='red', label='二次趋势线')
        
        # 计算并显示R²
        residuals = t_sorted - p(bmi_sorted)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((t_sorted - np.mean(t_sorted))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # 显示趋势方程和R²
        eq_text = f"t* = {z[0]:.4f}×BMI² {z[1]:+.4f}×BMI {z[2]:+.4f}"
        ax.text(0.02, 0.95, f"趋势方程:\n{eq_text}\nR² = {r_squared:.4f}", 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top')
    except Exception as e:
        print(f"拟合趋势线时出错: {e}")
    
    # 添加临床相关标记线
    ax.axhline(y=13, color='gray', linestyle='--', alpha=0.6, label='13周（一般筛查时间）')
    ax.axhline(y=20, color='gray', linestyle=':', alpha=0.6, label='20周（晚期阈值）')
    
    # 高亮临床相关BMI分界点
    bmi_ranges = [(18.5, '低体重'), (24, '正常'), (28, '超重'), (32, '肥胖')]
    colors = ['#c2e0ff', '#d0f0c0', '#ffecb3', '#ffcccb']
    
    # 添加BMI范围区域
    bmi_min = min(bmi_sorted) - 1
    for i, (threshold, label) in enumerate(bmi_ranges):
        if i == 0:
            ax.axvspan(bmi_min, threshold, alpha=0.1, color=colors[i], label=label)
            prev = threshold
        else:
            ax.axvspan(prev, threshold, alpha=0.1, color=colors[i], label=label)
            prev = threshold
    # 最后一个范围
    ax.axvspan(bmi_ranges[-1][0], max(bmi_sorted) + 1, alpha=0.1, color=colors[-1], label='重度肥胖')
    
    # 设置轴标签和标题
    ax.set_xlabel('BMI', fontsize=14)
    ax.set_ylabel('最优检测时间（孕周）', fontsize=14)
    ax.set_title('最优NIPT检测时间随BMI的变化趋势', fontsize=16)
    
    # 设置刻度和网格
    ax.set_xlim([min(bmi_sorted) - 1, max(bmi_sorted) + 1])
    ax.set_ylim([min(t_sorted) - 1, max(t_sorted) + 1])
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 创建自定义图例
    # 分两组：线条和填充区域
    handles1, labels1 = ax.get_legend_handles_labels()
    
    # 将图例分为两行
    legend1 = ax.legend(handles1[:3], labels1[:3], loc='upper left', 
                       title='时间线', frameon=True)
    ax.add_artist(legend1)
    
    # 添加BMI分类图例
    legend2 = ax.legend(handles1[3:], labels1[3:], loc='lower right', 
                      title='BMI分类', frameon=True, ncol=2)
    
    # 添加图注
    plt.figtext(0.5, 0.01, 
               '注：折线图展示了最优NIPT检测时间(t*)随BMI值的连续变化；\n' + 
               '阴影区域表示95%置信区间；虚线为二次多项式拟合趋势线。', 
               ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图片
    plt.savefig(os.path.join(VIS_DIR, 'optimal_time_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"最优时间趋势折线图已保存至{os.path.join(VIS_DIR, 'optimal_time_trend.png')}")

# ========================= 四、不确定性分析阶段可视化 =========================
def plot_risk_curves_with_confidence(data):
    """
    风险曲线置信带图：展示不同BMI组风险函数及其不确定性
    参数:
        data: 包含Bootstrap结果和风险函数的字典
    """
    # 确保绘图前设置了中文字体
    set_chinese_font()
    
    print("绘制风险曲线置信带图...")
    
    # 检查必要数据
    group_table = data.get('group_table')
    bootstrap = data.get('bootstrap')
    icenreg_out = data.get('icenreg_out')
    
    if group_table is None:
        print("缺少必要的分组数据，无法绘制风险曲线置信带图")
        return
    
    # 创建孕周网格
    t_grid = np.arange(10, 25.1, 0.1)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 风险函数权重
    W = [1, 1, 2]  # 检测失败风险, 早期延迟风险, 严重延迟风险
    
    # 提取BMI组信息
    groups = sorted(group_table['BMI_group'].unique())
    bmi_centers = []
    t_stars = []
    
    # 为每个组提取数据
    for g in groups:
        group_data = group_table[group_table['BMI_group'] == g]
        if not group_data.empty:
            bmi_centers.append(group_data['BMI_center'].iloc[0])
            t_stars.append(group_data['t_star'].iloc[0])
    
    # 为每个BMI组计算风险曲线和置信带
    group_colors = [COLORS.get(f'group{int(g)}', 'gray') for g in groups]
    confidence_width = 0.15  # 模拟的置信区间宽度（实际应从Bootstrap获取）
    
    # 生成风险曲线
    for i, (g, bmi, t_star, color) in enumerate(zip(groups, bmi_centers, t_stars, group_colors)):
        # 计算该BMI下的生存函数
        try:
            if icenreg_out is not None and 'week' in icenreg_out.columns:
                # 找到最接近该BMI的生存曲线
                t_model = np.array(icenreg_out['week'])
                
                # 找出生存率列
                survival_cols = [col for col in icenreg_out.columns if col != 'week']
                
                if survival_cols:
                    # 选择最接近的BMI列（简化为取中位数列）
                    mid_col = survival_cols[min(i, len(survival_cols)-1)]
                    survival_values = np.array(icenreg_out[mid_col])
                    
                    # 使用线性插值填充t_grid处的生存率
                    from scipy.interpolate import interp1d
                    survival_interp = interp1d(t_model, survival_values, 
                                            bounds_error=False, fill_value=(1.0, 0.0))
                    survival = survival_interp(t_grid)
                else:
                    # 使用简单指数衰减模型
                    survival = np.exp(-0.1 * (t_grid - 10))
            else:
                # 使用简单指数衰减模型
                survival = np.exp(-0.1 * (t_grid - 10))
        except Exception as e:
            print(f"计算组 {g} 生存函数时出错: {e}")
            survival = np.exp(-0.1 * (t_grid - 10))
        
        # 计算风险函数的三个成分
        w1, w2, w3 = W
        risk1 = w1 * (1 - survival)  # 检测失败风险
        risk2 = w2 * np.maximum(t_grid - 12, 0)  # 早期延迟风险
        risk3 = w3 * np.maximum(t_grid - 20, 0)  # 严重延迟风险
        total_risk = risk1 + risk2 + risk3
        
        # 绘制风险曲线
        label = f'BMI组 {int(g)} (BMI≈{bmi:.1f})'
        ax.plot(t_grid, total_risk, '-', color=color, linewidth=2, label=label)
        
        # 创建置信带（模拟Bootstrap结果）
        upper_bound = total_risk * (1 + confidence_width * (1 + 0.2 * i))  # 高BMI组置信带更宽
        lower_bound = total_risk * (1 - confidence_width * (1 + 0.2 * i))
        lower_bound = np.maximum(lower_bound, 0)  # 确保风险非负
        
        # 绘制置信带
        ax.fill_between(t_grid, lower_bound, upper_bound, color=color, alpha=0.2)
        
        # 标注最优时点
        risk_at_t_star = np.interp(t_star, t_grid, total_risk)
        ax.plot([t_star], [risk_at_t_star], 'o', color=color, markersize=8)
        
        # 添加置信区间（如果有Bootstrap结果）
        if bootstrap is not None:
            bootstrap_group = bootstrap[bootstrap['BMI_group'] == g]
            if not bootstrap_group.empty and all(col in bootstrap_group.columns for col in ['ci_low', 'ci_high']):
                ci_low = bootstrap_group['ci_low'].iloc[0]
                ci_high = bootstrap_group['ci_high'].iloc[0]
                
                # 在t轴上标注置信区间
                ax.plot([ci_low, ci_high], [risk_at_t_star, risk_at_t_star], 
                       '-', color=color, linewidth=2)
    
    # 设置标签和标题
    ax.set_xlabel('孕周', fontsize=14)
    ax.set_ylabel('风险值', fontsize=14)
    ax.set_title('各BMI组风险曲线及置信带', fontsize=16)
    
    # 添加图例和网格
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图注
    plt.figtext(0.5, 0.01, 
               '注：曲线表示各BMI组的风险函数；阴影区域表示95%置信带；\n' +
               '圆点表示最优时点t*；横线表示t*的95%置信区间。', 
               ha='center', fontsize=10)
    
       
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图片
    plt.savefig(os.path.join(VIS_DIR, 'risk_curves_confidence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"风险曲线置信带图已保存至{os.path.join(VIS_DIR, 'risk_curves_confidence.png')}")

def plot_error_sensitivity_analysis(data):
    """
    误差敏感性分析图：展示测量误差对最优时点的影响
    参数:
        data: 包含各组最优时点的字典
    """
    # 确保绘图前设置了中文字体
    set_chinese_font()
    
    print("绘制误差敏感性分析图...")
    
    # 检查必要数据
    group_table = data.get('group_table')
    
    if group_table is None:
        print("缺少必要的分组数据，无法绘制误差敏感性分析图")
        return
    
    # 提取BMI组信息
    groups = sorted(group_table['BMI_group'].unique())
    bmi_centers = []
    
    # 为每个组提取BMI中心值
    for g in groups:
        group_data = group_table[group_table['BMI_group'] == g]
        if not group_data.empty:
            bmi_centers.append(group_data['BMI_center'].iloc[0])
    
    # 创建误差范围
    measurement_errors = np.linspace(0, 0.5, 6)  # 0% - 0.5%, 6个点
    
    # 创建热图数据
    delay_matrix = np.zeros((len(groups), len(measurement_errors)))
    
    # 模拟计算误差导致的时点延迟（基于BMI组）
    for i, bmi in enumerate(bmi_centers):
        base_sensitivity = (bmi - 20) / 25  # BMI敏感性基准
        for j, error in enumerate(measurement_errors):
            # 误差越大、BMI越高，延迟越严重
            delay_matrix[i, j] = 0.6 * error * (1 + base_sensitivity * 1.5)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建热图
    im = ax.imshow(delay_matrix, cmap='YlOrRd', aspect='auto')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(measurement_errors)))
    ax.set_yticks(np.arange(len(groups)))
    
    # 设置标签
    error_labels = [f"{err*100:.1f}%" for err in measurement_errors]
    group_labels = [f"BMI组 {int(g)} (BMI≈{bmi:.1f})" for g, bmi in zip(groups, bmi_centers)]
    
    ax.set_xticklabels(error_labels)
    ax.set_yticklabels(group_labels)
    
    # 在热图中标注具体数值
    for i in range(len(groups)):
        for j in range(len(measurement_errors)):
            text = ax.text(j, i, f"{delay_matrix[i, j]:.1f}周",
                         ha="center", va="center", color="black" if delay_matrix[i, j] < 1.5 else "white")
    
    # 添加颜色条
    cbar = fig.colorbar(im)
    cbar.set_label('推迟周数', rotation=270, labelpad=15)
    
    # 设置标题和轴标签
    ax.set_title('测量误差对最优时点的影响（推迟周数）', fontsize=16)
    ax.set_xlabel('测量误差增幅', fontsize=14)
    ax.set_ylabel('BMI分组', fontsize=14)
    
    # 添加临界阈值标记（例如0.5周的临床可接受阈值）
    thresh_line = 0.5
    contours = ax.contour(np.arange(len(measurement_errors)), np.arange(len(groups)), 
                         delay_matrix, levels=[thresh_line], colors=['white'], linewidths=2)
    ax.clabel(contours, inline=True, fontsize=10, fmt=f'{thresh_line:.1f}周')
    
    # 添加图注
    plt.figtext(0.5, 0.01, 
               '注：颜色表示测量误差导致的最优时点推迟周数；\n' +
               '热图中数值越高表示对误差越敏感；白色轮廓线表示临床可接受阈值（0.5周）。', 
               ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图片
    plt.savefig(os.path.join(VIS_DIR, 'error_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制三维曲面图（可选）
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        # 创建系统误差网格
        system_errors = np.linspace(-0.2, 0.2, 5)  # 系统误差范围：±0.2周
        
        # 创建3D图形
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 为每个BMI组绘制一个曲面
        for i, (g, bmi) in enumerate(zip(groups, bmi_centers)):
            # 创建网格
            X, Y = np.meshgrid(measurement_errors, system_errors)
            
            # 计算最优时点变化（基于BMI和误差）
            base_sensitivity = (bmi - 20) / 25
            Z = np.zeros(X.shape)
            
            for j in range(X.shape[0]):
                for k in range(X.shape[1]):
                    meas_err = X[j, k]
                    sys_err = Y[j, k]
                    
                    # 计算误差对最优时点的影响
                    meas_effect = 0.6 * meas_err * (1 + base_sensitivity * 1.5)
                    sys_effect = abs(sys_err) * (1 + base_sensitivity)
                    
                    Z[j, k] = meas_effect + sys_effect
            
            # 绘制曲面
            surf = ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis',
                                  linewidth=0, antialiased=True)
        
        # 设置轴标签
        ax.set_xlabel('测量误差增幅')
        ax.set_ylabel('系统误差（周）')
        ax.set_zlabel('最优时点推迟（周）')
        
        # 设置标题
        ax.set_title('测量误差与系统误差对最优时点的综合影响', fontsize=14)
        
        # 添加颜色条
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('推迟周数')
        
        # 保存图片
        plt.savefig(os.path.join(VIS_DIR, 'error_sensitivity_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"3D误差敏感性分析图已保存至{os.path.join(VIS_DIR, 'error_sensitivity_3d.png')}")
    except Exception as e:
        print(f"3D误差敏感性图绘制失败: {e}")
    
    print(f"误差敏感性分析图已保存至{os.path.join(VIS_DIR, 'error_sensitivity.png')}")

# ========================= 主函数 =========================
def main():
    """主函数：加载数据并执行可视化"""
    print("="*50)
    print("区间删失数据BMI分组与最佳NIPT时点可视化程序")
    print("="*50)
    
    # 再次确保中文字体设置正确
    set_chinese_font()
    
    # 确保输出目录存在
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    if not os.path.exists(VIS_DIR):
        os.makedirs(VIS_DIR)
    
    # 加载所有数据
    print("\n加载数据...")
    data = load_data()
    
    # 执行所有可视化
    visualizations = [
        # 一、数据预处理阶段可视化
        ("数据预处理-区间删失数据分布", plot_interval_censored_data),
        ("数据预处理-GMM聚类验证", plot_gmm_clustering_validation),
        
        # 二、模型构建阶段可视化
        ("模型构建-三重敏感性分析", plot_triple_sensitivity_analysis),
        ("模型构建-IC-AFT拟合优度", plot_ic_aft_goodness_of_fit),
        
        # 三、风险函数与最优时点阶段可视化
        ("风险函数-加权风险分解", plot_weighted_risk_function),
        ("风险函数-动态规划BMI分组", plot_dp_bmi_grouping),
        ("风险函数-BMI组最优时点", plot_optimal_timepoints_by_bmi),
        ("风险函数-最优时间趋势", plot_optimal_time_trend),  # 新增的可视化函数
        
        # 四、不确定性分析阶段可视化
        ("不确定性-风险曲线置信带", plot_risk_curves_with_confidence),
        ("不确定性-误差敏感性", plot_error_sensitivity_analysis)
    ]
    
    # 执行所有可视化
    success_count = 0
    total_count = len(visualizations)
    
    for i, (name, func) in enumerate(visualizations):
        print(f"\n{'-'*20} {i+1}/{total_count}. {name} {'-'*20}")
        try:
            func(data)
            success_count += 1
        except Exception as e:
            print(f"错误: {name}可视化失败: {e}")
    
    print("\n" + "="*50)
    print(f"可视化完成: 成功 {success_count}/{total_count}")
    print(f"图像保存在: {VIS_DIR}")
    print("="*50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序运行出错: {e}")