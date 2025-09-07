# 方法框架总览（简要）
# 本脚本实现“BMA-GMM-Joint-NSGA-II”四步框架，用于在“高风险-高达标率-低误差敏感”三维目标间取得均衡：
# 1) BMA: 通过BIC加权近似计算每个候选协变量的PIP，保留PIP>阈值的变量；
# 2) GMM: 对保留变量执行软聚类（BIC选择簇数），得到簇归属与概率；
# 3) Joint (两步近似): 先拟合纵向模型得到浓度预测，再在AFT中引入预测作为时变协变量；
# 4) NSGA-II: 在多目标（风险↓、达标率↑、误差敏感↓）下搜索Pareto前沿，给出临床可选解。

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from lifelines import LogNormalAFTFitter
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
import scipy.stats as stats
import warnings
import statsmodels.api as sm
import os
warnings.filterwarnings('ignore')

# 定义输出目录
OUT_DIR = "./output_q3"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# 设置随机种子，保证结果可复现
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
# --------------------------
# 数据读取与预处理，从CSV文件读取数据
# --------------------------
def load_and_preprocess_data(file_path):
    """读取CSV并做列名自动映射、缺失值处理和标准化（支持中/英列名）。
    扩展：自动处理检测孕周 -> GA_lower/GA_upper的映射，若只有单个测量则上下界相同；
    若存在胎儿是否健康列，会尝试构造event列（0/1）；若无法判断則默认为1（已观察）。
    新增：支持读取所有列并映射为变量名。"""
    df = pd.read_csv(file_path)
    # 去除列名首尾空格，方便匹配中文列名（例如文件中有尾随空格）
    df.columns = df.columns.str.strip()
    print(f"原始数据形状: {df.shape}")
    print("所有列名：", df.columns.tolist())

    # 期望的标准列名（后续代码使用这些名字）
    std_cols = [
        "patient_id", "BMI", "age", "height", "weight", "IVF",
        "GC_content", "alignment_rate", "Y_frac", "GA_lower", "GA_upper", "event"
    ]

    # 扩展的同义词映射（包含所有可能的列名）
    synonyms = {
        "patient_id": ["patient_id", "孕妇代码"],
        "BMI": ["bmi", "孕妇BMI"],
        "age": ["age", "年龄"],
        "height": ["height", "身高"],
        "weight": ["weight",  "体重"],
        "IVF": ["ivf", "IVF妊娠"],
        "GC_content": ["gc_content", "GC含量"],
        "alignment_rate": ["alignment_rate", "在参考基因组上比对的比例"],
        "Y_frac": ["y_frac", "y_fraction", "Y染色体浓度", "Y染色体的Z值", "Y染色体的Z值", "Y染色体浓度"],
        "GA_lower": ["ga_lower", "gestational_age_lower", "GA_lower", "孕周_lower", "检测孕周", "检测孕周"],
        "GA_upper": ["ga_upper", "gestational_age_upper", "GA_upper", "孕周_upper", "GA上限"],
        "event": ["event", "status", "observed", "事件", "是否事件", "胎儿是否健康"],
        # 其余列的映射，用于热力图
        "last_menstrual_period": ["末次月经", "last_menstrual_period", "lmp"],
        "detection_date": ["检测日期", "detection_date", "test_date"],
        "detection_count": ["检测抽血次数", "detection_count", "blood_test_count"],
        "raw_reads": ["原始读段数", "raw_reads", "total_reads"],
        "duplicate_rate": ["重复读段的比例", "duplicate_rate", "dup_rate"],
        "unique_reads": ["唯一比对的读段数", "unique_reads", "unique_aligned_reads"],
        "chr13_z": ["13号染色体的Z值", "chr13_z", "chromosome_13_z"],
        "chr18_z": ["18号染色体的Z值", "chr18_z", "chromosome_18_z"],
        "chr21_z": ["21号染色体的Z值", "chr21_z", "chromosome_21_z"],
        "chrX_z": ["X染色体的Z值", "chrX_z", "chromosome_X_z"],
        "chrY_z": ["Y染色体的Z值", "chrY_z", "chromosome_Y_z"],
        "X_chromosome_concentration": ["X染色体浓度", "X_chromosome_concentration", "X_conc"],
        "chr13_gc": ["13号染色体的GC含量", "chr13_gc", "chromosome_13_gc"],
        "chr18_gc": ["18号染色体的GC含量", "chr18_gc", "chromosome_18_gc"],
        "chr21_gc": ["21号染色体的GC含量", "chr21_gc", "chromosome_21_gc"],
        "filtered_reads_rate": ["被过滤掉读段数的比例", "filtered_reads_rate", "filter_rate"],
        "aneuploidy": ["染色体的非整倍体", "aneuploidy", "chromosomal_aneuploidy"],
        "pregnancy_count": ["怀孕次数", "pregnancy_count", "gravida"],
        "delivery_count": ["生产次数", "delivery_count", "parity"]
    }

    lower_to_orig = {c.lower(): c for c in df.columns}
    rename_map = {}
    missing = []
    
    # 处理所有标准列
    all_std_cols = list(synonyms.keys())
    for std in all_std_cols:
        found = False
        for alt in synonyms.get(std, [std]):
            if alt.lower() in lower_to_orig:
                rename_map[lower_to_orig[alt.lower()]] = std
                found = True
                break
        if not found and std in std_cols:  # 只对核心列报告缺失
            missing.append(std)

    # 先重命名已有列
    df = df.rename(columns=rename_map)
    
    print(f"成功映射的列: {list(rename_map.values())}")
    print(f"核心列缺失: {missing}")

    # 若只有 GA_lower 存在但 GA_upper 不存在，则令 GA_upper = GA_lower
    if "GA_lower" in df.columns and "GA_upper" not in df.columns:
        df["GA_upper"] = df["GA_lower"]

    # 若没有 GA_lower，但存在 '检测孕周' 原列（未被映射），再尝试直接用该列
    if "GA_lower" not in df.columns:
        for cand in ["检测孕周", "检测孕周"]:
            if cand in df.columns:
                df["GA_lower"] = df[cand]
                df["GA_upper"] = df[cand]
                break

    # event 列处理：优先数值0/1，否则尝试根据字符串判定，无法判定则全部置1（已观察）
    if "event" not in df.columns:
        if "胎儿是否健康" in df.columns:
            col = df["胎儿是否健康"]
            if set(col.dropna().unique()).issubset({0,1}):
                df["event"] = col
            else:
                # 尝试字符串匹配：'健康' -> 0, 其他 ->1
                df["event"] = col.astype(str).apply(lambda x: 0 if "健康" in x else 1)
        else:
            df["event"] = 1  # 默认所有为观察到的事件（可根据实际数据修改）

    # 数值列（后续要 impute/scale）- 扩展包含所有可能的数值列
    numeric_cols = ["BMI", "age", "height", "weight", "Y_frac", "GC_content", "alignment_rate",
                   "raw_reads", "duplicate_rate", "unique_reads", "chr13_z", "chr18_z", "chr21_z",
                   "chrX_z", "chrY_z", "X_chromosome_concentration", "chr13_gc", "chr18_gc", 
                   "chr21_gc", "filtered_reads_rate", "pregnancy_count", "delivery_count"]
    
    # 只处理实际存在的数值列
    existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
    miss_num = [c for c in ["BMI", "age", "height", "weight", "Y_frac", "GC_content", "alignment_rate"] 
                if c not in df.columns]
    if miss_num:
        raise ValueError(f"缺少用于数值处理的列: {miss_num}，当前文件列: {df.columns.tolist()[:80]}")

    # 清洗数值列（移除括号或注释等非数字字符），并强制转换为数值，不能解析的设为 NaN
    for col in existing_numeric_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'': np.nan, 'nan': np.nan})
        df[col] = df[col].str.replace(r'[^\d\.\-eE]+', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 新增：清洗 GA_lower / GA_upper ---
    for ga_col in ["GA_lower", "GA_upper"]:
        if ga_col in df.columns:
            df[ga_col] = df[ga_col].astype(str).str.strip()
            # 去掉非数字字符，但保留小数点和负号、科学计数法
            df[ga_col] = df[ga_col].str.replace(r'[^\d\.\-eE]+', '', regex=True)
            df[ga_col] = pd.to_numeric(df[ga_col], errors='coerce')
    # 若两者皆缺失则尝试用检测孕周原列或填充中位数
    if "GA_lower" not in df.columns or df["GA_lower"].isna().all():
        # 尝试用检测孕周原列（若存在）
        for cand in ["检测孕周", "孕周"]:
            if cand in df.columns:
                df["GA_lower"] = pd.to_numeric(df[cand].astype(str).str.replace(r'[^\d\.\-eE]+','',regex=True), errors='coerce')
                df["GA_upper"] = df["GA_lower"]
                break
    # 最终仍然有缺失则用中位数填充
    if "GA_lower" in df.columns:
        med = df["GA_lower"].median(skipna=True)
        df["GA_lower"] = df["GA_lower"].fillna(med)
    if "GA_upper" in df.columns:
        med_u = df["GA_upper"].median(skipna=True)
        df["GA_upper"] = df["GA_upper"].fillna(med_u)
    # 确保为浮点数
    if "GA_lower" in df.columns:
        df["GA_lower"] = df["GA_lower"].astype(float)
    if "GA_upper" in df.columns:
        df["GA_upper"] = df["GA_upper"].astype(float)

    # 缺失值处理（数值列）- 只处理实际存在的列
    if existing_numeric_cols:
        imputer = KNNImputer(n_neighbors=5)
        df[existing_numeric_cols] = imputer.fit_transform(df[existing_numeric_cols])

    # 特征工程
    if "BMI" in df.columns and "age" in df.columns:
        df["BMI_age_interact"] = df["BMI"] * df["age"]
    if "BMI" in df.columns and "weight" in df.columns:
        df["BMI_weight_ratio"] = df["BMI"] / df["weight"]

    # 标准化 - 只对实际存在的列进行标准化
    scale_cols = existing_numeric_cols.copy()
    if "BMI_age_interact" in df.columns:
        scale_cols.append("BMI_age_interact")
    if "BMI_weight_ratio" in df.columns:
        scale_cols.append("BMI_weight_ratio")
    
    if scale_cols:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    print(f"预处理后的数据形状: {df.shape}")
    return df

# 读取并预处理数据（请确保processed_male.csv在正确路径）
processed_data = load_and_preprocess_data("processed_male.csv")
print(f"预处理后的数据形状: {processed_data.shape}")
print(f"前5行数据:\n{processed_data.head()}")

# --------------------------
# 绘制变量间的相关性热力图
# --------------------------
def plot_correlation_heatmap(df, save_path=None):
    """绘制所有数值变量间的相关性热力图，使用原始列名"""
    # 选择数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 排除一些不需要的列
    exclude_cols = ['cluster', 'cluster_prob', 'Y_pred', 'gestational_weeks']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) < 2:
        print("数值列数量不足，无法绘制相关性热力图")
        return
    
    # 计算相关性矩阵
    corr_matrix = df[numeric_cols].corr()
    
    # 定义映射关系（映射后的列名 -> 原始列名）
    reverse_mapping = {
        "patient_id": "孕妇代码",
        "BMI": "孕妇BMI", 
        "age": "年龄",
        "height": "身高",
        "weight": "体重",
        "IVF": "IVF妊娠",
        "GC_content": "GC含量",
        "alignment_rate": "在参考基因组上比对的比例",
        "Y_frac": "Y染色体浓度",
        "GA_lower": "检测孕周",
        "GA_upper": "检测孕周",
        "event": "胎儿是否健康",
        "last_menstrual_period": "末次月经",
        "detection_date": "检测日期",
        "detection_count": "检测抽血次数",
        "raw_reads": "原始读段数",
        "duplicate_rate": "重复读段的比例",
        "unique_reads": "唯一比对的读段数",
        "chr13_z": "13号染色体的Z值",
        "chr18_z": "18号染色体的Z值",
        "chr21_z": "21号染色体的Z值",
        "chrX_z": "X染色体的Z值",
        "chrY_z": "Y染色体的Z值",
        "X_chromosome_concentration": "X染色体浓度",
        "chr13_gc": "13号染色体的GC含量",
        "chr18_gc": "18号染色体的GC含量",
        "chr21_gc": "21号染色体的GC含量",
        "filtered_reads_rate": "被过滤掉读段数的比例",
        "aneuploidy": "染色体的非整倍体",
        "pregnancy_count": "怀孕次数",
        "delivery_count": "生产次数"
    }
    
    # 创建显示标签（使用原始列名）
    display_labels = []
    for col in corr_matrix.columns:
        if col in reverse_mapping:
            display_labels.append(reverse_mapping[col])
        else:
            display_labels.append(col)
    
    # 创建图形
    plt.figure(figsize=(16, 12))
    
    # 绘制热力图
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8},
                annot_kws={'size': 8},
                xticklabels=display_labels,
                yticklabels=display_labels)
    
    plt.title('变量间相关性热力图', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片
    if save_path is None:
        save_path = os.path.join(OUT_DIR, 'correlation_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印高相关性对（使用原始列名）
    print("\n高相关性变量对 (|相关系数| > 0.7):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                var1_display = display_labels[i]
                var2_display = display_labels[j]
                high_corr_pairs.append((var1_display, var2_display, corr_val))
    
    if high_corr_pairs:
        for var1, var2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"{var1} - {var2}: {corr:.3f}")
    else:
        print("未发现高相关性变量对")
    
    return corr_matrix

# 绘制相关性热力图
correlation_matrix = plot_correlation_heatmap(processed_data)

# --------------------------
# 贝叶斯模型平均(BMA)特征选择
# --------------------------
def bma_feature_selection(df, target_col="Y_frac", n_models=200, threshold=0.5, max_subset_size=None):
    """基于BIC加权的近似BMA：
    - 随机采样子模型（或遍历全部若p小），对每个子模型用OLS计算BIC；
    - 用 exp(-0.5*BIC) 归一化得到模型权重，计算每个变量的PIP。"""
    candidate_features = ["BMI", "age", "height", "weight", "IVF", "GC_content", "BMI_age_interact", "BMI_weight_ratio"]
    X_full = df[candidate_features]
    y = df[target_col]

    p = len(candidate_features)
    if max_subset_size is None:
        max_subset_size = p

    models = []
    bics = []
    subsets = []

    # 若候选变量较少且可遍历，则遍历全部子集（除空集）
    if p <= 15:
        from itertools import combinations
        for r in range(1, p+1):
            for comb in combinations(range(p), r):
                subsets.append(list(comb))
    else:
        # 随机采样子集
        for _ in range(n_models):
            k = np.random.randint(1, max_subset_size+1)
            subset = list(np.random.choice(p, size=k, replace=False))
            subsets.append(subset)

    # 去重子集
    unique_subsets = []
    seen = set()
    for s in subsets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_subsets.append(s)

    for s in unique_subsets:
        cols = [candidate_features[i] for i in s]
        X = sm.add_constant(X_full[cols])
        try:
            model = sm.OLS(y, X, missing='drop').fit()
            bics.append(model.bic)
            models.append(model)
        except Exception:
            # 若拟合失败，记一个大BIC以降权
            bics.append(1e6)
            models.append(None)

    bics = np.array(bics)
    # 为数值稳定性减去最小BIC
    w = np.exp(-0.5 * (bics - np.nanmin(bics)))
    w = w / np.nansum(w)

    pip = np.zeros(p)
    for i, s in enumerate(unique_subsets):
        weight = w[i]
        for idx in s:
            pip[idx] += weight

    pip_results = pd.DataFrame({
        "feature": candidate_features,
        "pip": pip
    }).sort_values("pip", ascending=False)

    selected_features = pip_results[pip_results["pip"] > threshold]["feature"].tolist()

    print("\nBMA (BIC加权) 特征选择结果:")
    print(pip_results)
    print(f"\n选中的特征 (PIP>{threshold}): {selected_features}")

    return selected_features, pip_results

# 执行BMA特征选择
selected_features, pip_results = bma_feature_selection(processed_data)

# --------------------------
#  高斯混合模型(GMM)聚类
# --------------------------
def gmm_clustering(df, features, max_k=5):
    """使用GMM进行软聚类，基于BIC选择最佳簇数"""
    X = df[features].values
    
    # 计算不同簇数的BIC
    bic_scores = []
    models = []
    for k in range(2, max_k+1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        models.append(gmm)
    
    # 选择BIC最小的模型
    best_idx = np.argmin(bic_scores)
    best_k = best_idx + 2
    best_model = models[best_idx]
    
    # 获取聚类结果
    df['cluster'] = best_model.predict(X)
    df['cluster_prob'] = np.max(best_model.predict_proba(X), axis=1)  # 最大后验概率
    
    print(f"\nGMM最佳簇数: {best_k} (BIC={bic_scores[best_idx]:.2f})")
    
    # 聚类结果可视化
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="BMI", y="age", hue="cluster", palette="viridis", 
                   size="cluster_prob", sizes=(50, 200), alpha=0.7)
    plt.title("GMM聚类结果 (按BMI和年龄可视化)")
    plt.savefig(os.path.join(OUT_DIR, 'BMI_Age_result.png'))
    plt.show()
    
    return df, best_model

# 执行GMM聚类
data_with_clusters, gmm_model = gmm_clustering(processed_data, selected_features)

# --------------------------
#  多变量AFT模型 (层级1)
# --------------------------
def fit_multivariate_aft(df):
    """拟合多变量区间删失AFT模型（兼容不同 lifelines 版本）"""
    aft_data = df.copy()
    aft_data['GA_upper'] = aft_data['GA_upper'].fillna(aft_data['GA_lower'])
    model_features = selected_features + ['cluster']
    
    # 先对 event 列进行文本到数值的转换
    if aft_data['event'].dtype == 'object':
        # 若是字符串类型，进行映射：将中文"是"/"否"或"健康"/"异常"等转为 1/0
        aft_data['event'] = aft_data['event'].astype(str).apply(
            lambda x: 1 if x in ['1', '是', '异常', 'True', 'true'] else 0
        )
    
    # 确保 event 为整数
    aft_data['event'] = aft_data['event'].astype(int)
    
    # 新增: 确保区间删失数据格式正确
    mask_observed = (aft_data['event'] == 1)
    aft_data.loc[mask_observed, 'GA_upper'] = aft_data.loc[mask_observed, 'GA_lower']
    
    # 对于未观察事件，如果上下界相等，则略微增加上界
    mask_unobserved = (aft_data['event'] == 0) & (aft_data['GA_lower'] >= aft_data['GA_upper'])
    aft_data.loc[mask_unobserved, 'GA_upper'] = aft_data.loc[mask_unobserved, 'GA_lower'] + 0.01
    
    aft = LogNormalAFTFitter()
    formula = f"~ {' + '.join(model_features)}"

    # 优先使用区间删失接口（若可用），否则退回右删失拟合
    if hasattr(aft, "fit_interval_censoring"):
        aft.fit_interval_censoring(
            aft_data,
            lower_bound_col='GA_lower',
            upper_bound_col='GA_upper',
            event_col='event',
            formula=formula
        )
    else:
        # 若没有区间删失接口，则使用右删失拟合（将 GA_lower 视作 duration）
        print("警告: 当前 lifelines 版本不支持 fit_interval_censoring，退回为右删失拟合（duration_col=GA_lower）")
        aft.fit(
            aft_data,
            duration_col='GA_lower',
            event_col='event',
            formula=formula
        )

    print("\n多变量AFT模型 summary:")
    print(aft.summary)

    return aft

# 拟合AFT模型
aft_model = fit_multivariate_aft(data_with_clusters)

# --------------------------
#  两步近似联合模型 (层级2)
# --------------------------
def two_step_joint_model(df):
    """两步近似联合模型: 先拟合纵向模型，再将预测值代入AFT模型（兼容不同 lifelines 版本）"""
    df['gestational_weeks'] = (df['GA_lower'] + df['GA_upper']) / 2
    X_long = df[selected_features + ['gestational_weeks']]
    y_long = df['Y_frac']
    long_model = LinearRegression()
    long_model.fit(X_long, y_long)
    df['Y_pred'] = long_model.predict(X_long)

    aft_data = df.copy()
    aft_data['GA_upper'] = aft_data['GA_upper'].fillna(aft_data['GA_lower'])
    
    # 添加变量定义
    model_features_joint = selected_features + ['cluster', 'Y_pred']
    
    # 先对 event 列进行文本到数值的转换
    if aft_data['event'].dtype == 'object':
        # 若是字符串类型，进行映射：将中文"是"/"否"或"健康"/"异常"等转为 1/0
        aft_data['event'] = aft_data['event'].astype(str).apply(
            lambda x: 1 if x in ['1', '是', '异常', 'True', 'true'] else 0
        )
    
    # 确保 event 为整数
    aft_data['event'] = aft_data['event'].astype(int)
    
    # 新增: 确保区间删失数据格式正确
    mask_observed = (aft_data['event'] == 1)
    aft_data.loc[mask_observed, 'GA_upper'] = aft_data.loc[mask_observed, 'GA_lower']
    
    # 对于未观察事件，如果上下界相等，则略微增加上界
    mask_unobserved = (aft_data['event'] == 0) & (aft_data['GA_lower'] >= aft_data['GA_upper'])
    aft_data.loc[mask_unobserved, 'GA_upper'] = aft_data.loc[mask_unobserved, 'GA_lower'] + 0.01
    
    joint_aft = LogNormalAFTFitter()
    formula = f"~ {' + '.join(model_features_joint)}"

    if hasattr(joint_aft, "fit_interval_censoring"):
        joint_aft.fit_interval_censoring(
            aft_data,
            lower_bound_col='GA_lower',
            upper_bound_col='GA_upper',
            event_col='event',
            formula=formula
        )
    else:
        print("警告: 当前 lifelines 版本不支持 fit_interval_censoring，退回为右删失拟合（duration_col=GA_lower）")
        joint_aft.fit(
            aft_data,
            duration_col='GA_lower',
            event_col='event',
            formula=formula
        )

    print("\n两步近似联合模型 summary:")
    print(joint_aft.summary)

    return long_model, joint_aft, df

# 拟合两步近似联合模型
long_model, joint_model, data_with_joint = two_step_joint_model(data_with_clusters)

# ---------- 可行性诊断（） ----------
class NIPTProblem(Problem):
    """定义多目标优化问题，支持可配置 success/fnr 阈值与放宽策略"""
    def __init__(self, model, data, cluster_id, success_thresh=0.9, fnr_thresh=0.05, enforce_constraints=True):
        super().__init__(n_var=1, n_obj=3, n_constr=3, xl=[10], xu=[25])
        self.model = model
        self.data = data[data['cluster'] == cluster_id].copy()
        self.cluster_id = cluster_id
        self.success_thresh = success_thresh
        self.fnr_thresh = fnr_thresh
        self.enforce_constraints = enforce_constraints

    def calculate_risk(self, t):
        return np.mean(np.abs(self.data['GA_lower'] - t) / self.data['GA_lower'])

    def calculate_success_rate(self, t):
        sel = self.data[self.data['GA_lower'] <= t]
        return np.mean(sel['Y_frac'] > 0.04) if sel.shape[0] > 0 else 0.0

    def calculate_error_sensitivity(self, t):
        sel = self.data[self.data['GA_lower'] <= t]
        return np.std(sel['Y_frac']) if sel.shape[0] > 0 else 0.0

    def calculate_fnr(self, t):
        positive_cases = self.data[self.data['Y_frac'] > 0.04]
        if positive_cases.shape[0] == 0:
            return 0.0
        fn = positive_cases[positive_cases['GA_lower'] > t].shape[0]
        return fn / positive_cases.shape[0]

    def grid_feasible(self, t_grid=None):
        """在网格上检查是否存在满足当前阈值的可行 t"""
        if t_grid is None:
            t_grid = np.linspace(10, 25, 151)
        for t in t_grid:
            sr = self.calculate_success_rate(float(t))
            fnr = self.calculate_fnr(float(t))
            if (sr >= self.success_thresh) and (fnr <= self.fnr_thresh):
                return True
        return False

    def _evaluate(self, X, out, *args, **kwargs):
        risk = np.array([self.calculate_risk(t[0]) for t in X])
        success_rate = np.array([self.calculate_success_rate(t[0]) for t in X])
        success_obj = 1 - success_rate
        error_sens = np.array([self.calculate_error_sensitivity(t[0]) for t in X])

        # 若 enforce_constraints 为 True，则输出约束，否则输出空约束（均为满足）
        if self.enforce_constraints:
            constr1 = self.success_thresh - success_rate  # >=0 表示满足
            fnr = np.array([self.calculate_fnr(t[0]) for t in X])
            constr2 = fnr - self.fnr_thresh  # <=0 表示满足
        else:
            constr1 = np.full_like(success_rate, -1.0)  # 恒满足
            constr2 = np.full_like(success_rate, -1.0)

        constr3 = np.zeros_like(X[:, 0])
        out["F"] = np.column_stack([risk, success_obj, error_sens])
        out["G"] = np.column_stack([constr1, constr2, constr3])

# ---------- 可行性诊断 ----------
def diagnostic_feasibility(model, data, cluster_id, t_grid=None, success_thresh=0.9, fnr_thresh=0.05):
    prob = NIPTProblem(model, data, cluster_id)
    if t_grid is None:
        t_grid = np.linspace(10, 25, 151)
    rows = []
    for t in t_grid:
        sr = prob.calculate_success_rate(float(t))
        fnr = prob.calculate_fnr(float(t))
        risk = prob.calculate_risk(float(t))
        es = prob.calculate_error_sensitivity(float(t))
        rows.append((t, sr, fnr, risk, es))
    df_check = pd.DataFrame(rows, columns=['t', 'success_rate', 'fnr', 'risk', 'error_sens'])
    feasible = df_check[(df_check['success_rate'] >= success_thresh) & (df_check['fnr'] <= fnr_thresh)]
    print("\n诊断：t 网格统计（min/mean/max）")
    print(df_check.describe().T[['min','mean','max']])
    print(f"\n满足 success>={success_thresh} 且 fnr<={fnr_thresh} 的点数: {feasible.shape[0]}")
    if feasible.shape[0] > 0:
        print("示例可行 t（前5）：")
        print(feasible.head())
    else:
        print("未发现可行 t。建议：降低 success_thresh（例如 0.85 或 0.8），或放宽 fnr_thresh，或检查 success_rate/ fnr 的计算逻辑。")
    return df_check, feasible

# 在当前 cluster_id 上做诊断（先用当前 cluster_id 0，再用最大聚类）
for cid in [0, data_with_clusters['cluster'].value_counts().idxmax()]:
    print(f"\n--- 对聚类 {cid} 做可行性诊断 ---")
    df_check, feasible = diagnostic_feasibility(joint_model, data_with_clusters, cid)
    if feasible.shape[0] == 0:
        print("Top 5 按 success_rate 排序（用于参考）:")
        print(df_check.sort_values('success_rate', ascending=False).head())
        print("Top 5 按 risk 升序排序（用于参考）:")
        print(df_check.sort_values('risk', ascending=True).head())

# --------------------------
# 6. 多目标优化 (NSGA-II)
# --------------------------

# ---------- 使用诊断结果决定优化设置 ----------
# 先尝试严格阈值
cluster_id = data_with_joint['cluster'].value_counts().idxmax()  # 默认用最大聚类
print(f"\n开始优化，使用聚类 {cluster_id} (样本数 {data_with_joint[data_with_joint['cluster']==cluster_id].shape[0]})")
problem = NIPTProblem(joint_model, data_with_joint, cluster_id,
                      success_thresh=0.9, fnr_thresh=0.05, enforce_constraints=True)

# 若网格上无可行点，则自动尝试放宽阈值，再不行则关闭硬约束
if not problem.grid_feasible():
    print("严格阈值下无可行解，尝试放宽阈值 success>=0.85, fnr<=0.10")
    problem = NIPTProblem(joint_model, data_with_joint, cluster_id,
                          success_thresh=0.85, fnr_thresh=0.10, enforce_constraints=True)
    if not problem.grid_feasible():
        print("放宽后仍无可行解，改为不强制约束（将把约束作为目标/后处理参考）")
        problem = NIPTProblem(joint_model, data_with_joint, cluster_id,
                              success_thresh=0.85, fnr_thresh=0.10, enforce_constraints=False)

# 配置并运行 NSGA-II
algorithm = NSGA2(
    pop_size=100,
    n_offsprings=50,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9),
    mutation=PM(prob=0.1),
    eliminate_duplicates=True
)

res = minimize(problem, algorithm, ('n_gen', 200), seed=42, verbose=False)

# 结果健壮性检查与可视化（与之前一致）
if res is None:
    print("优化未返回结果 (res is None)。")
else:
    F = getattr(res, "F", None)
    X = getattr(res, "X", None)
    if F is None or getattr(F, "size", 0) == 0:
        print("优化未找到可行解或 Pareto 集为空 (res.F 为空)。请检查约束条件或考虑放宽/取消硬约束。")
    else:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.scatter(F[:, 0], 1 - F[:, 1], c=F[:, 2], cmap='viridis')
        plt.colorbar(label="误差敏感性")
        plt.xlabel("风险"); plt.ylabel("达标率"); plt.title("风险 vs 达标率")
        plt.subplot(1, 3, 2)
        plt.scatter(F[:, 0], F[:, 2], c=1 - F[:, 1], cmap='viridis')
        plt.colorbar(label="达标率")
        plt.xlabel("风险"); plt.ylabel("误差敏感性"); plt.title("风险 vs 误差敏感性")
        plt.subplot(1, 3, 3)
        plt.scatter(1 - F[:, 1], F[:, 2], c=F[:, 0], cmap='viridis')
        plt.colorbar(label="风险")
        plt.xlabel("达标率"); plt.ylabel("误差敏感性"); plt.title("达标率 vs 误差敏感性")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'pareto_front_2d_auto.png'))
        plt.show()

# 绘制三维Pareto前沿在二维平面上的投影
# 使用优化得到的结果集(res)，如果没有res，则使用诊断数据
def plot_pareto_2d_projection(res=None, df_check=None, cluster_id=None):
    plt.figure(figsize=(10, 8))
    
    if res is not None and hasattr(res, "F") and res.F is not None and res.F.size > 0:
        # 使用优化结果
        data_source = "优化得到的Pareto前沿"
        risk = res.F[:, 0]
        success_rate = 1 - res.F[:, 1]  # 因为第1列是1-达标率，所以需要反转
        error_sens = res.F[:, 2]
        
        # 找出Pareto前沿上的几个典型点（极值点和平衡点）
        min_risk_idx = np.argmin(risk)
        max_success_idx = np.argmax(success_rate)
        min_error_idx = np.argmin(error_sens)
        
        # 平衡点（欧几里得距离最小）
        norm_risk = (risk - np.min(risk)) / (np.max(risk) - np.min(risk) + 1e-10)
        norm_success = (success_rate - np.min(success_rate)) / (np.max(success_rate) - np.min(success_rate) + 1e-10) 
        norm_error = (error_sens - np.min(error_sens)) / (np.max(error_sens) - np.min(error_sens) + 1e-10)
        dist_to_ideal = np.sqrt(norm_risk**2 + (1-norm_success)**2 + norm_error**2)
        balanced_idx = np.argmin(dist_to_ideal)
        
        # 标记关键点
        key_points = [min_risk_idx, max_success_idx, min_error_idx, balanced_idx]
        key_labels = ["最小风险", "最高达标率", "最小误差敏感性", "平衡点"]
        
    elif df_check is not None:
        # 使用诊断网格数据
        data_source = f"网格搜索数据 (聚类 {cluster_id})"
        risk = df_check['risk']
        success_rate = df_check['success_rate']  # 这里已经是达标率，不需要反转
        error_sens = df_check['error_sens']
        
        # 找出几个特征点
        min_risk_idx = np.argmin(risk)
        max_success_idx = np.argmax(success_rate) 
        min_error_idx = np.argmin(error_sens)
        
        # 平衡点（标准化后的欧几里得距离最小）
        norm_risk = (risk - np.min(risk)) / (np.max(risk) - np.min(risk) + 1e-10)
        norm_success = (success_rate - np.min(success_rate)) / (np.max(success_rate) - np.min(success_rate) + 1e-10)
        norm_error = (error_sens - np.min(error_sens)) / (np.max(error_sens) - np.min(error_sens) + 1e-10)
        dist_to_ideal = np.sqrt(norm_risk**2 + (1-norm_success)**2 + norm_error**2)
        balanced_idx = np.argmin(dist_to_ideal)
        
        # 标记关键点
        key_points = [min_risk_idx, max_success_idx, min_error_idx, balanced_idx]
        key_labels = ["最小风险", "最高达标率", "最小误差敏感性", "平衡点"]
    else:
        print("错误：未提供可视化数据")
        return
    
    # 散点图
    sc = plt.scatter(risk, success_rate, c=error_sens, cmap='viridis', 
                    s=80, alpha=0.8, edgecolor='k', linewidth=0.5)
    
    # 标记关键点并添加标签
    for i, idx in enumerate(key_points):
        plt.scatter(risk[idx], success_rate[idx], s=150, 
                   facecolors='none', edgecolors='red', linewidth=2)
        plt.annotate(f"{key_labels[i]}\nt={df_check['t'][idx]:.1f}w", 
                    (risk[idx], success_rate[idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
    
    # 添加颜色条和标签
    cbar = plt.colorbar(sc, label='误差敏感性')
    cbar.ax.tick_params(labelsize=10)
    
    # 设置标题和轴标签
    plt.title(f"NIPT三维Pareto前沿投影图\n数据来源: {data_source}", fontsize=14)
    plt.xlabel("风险 R(t)", fontsize=12)
    plt.ylabel("达标率", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加说明文本
    plt.figtext(0.02, 0.02, 
               "注释:\n"
               "• 风险 R(t): 检测时间偏离理想时间的平均相对误差\n"
               "• 达标率: 在t周前检出Y染色体浓度>4%的比例\n"
               "• 误差敏感性: t周前Y染色体浓度的标准差\n"
               "• 红圈标记的点为特征解，可供临床决策参考", 
               fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'nipt_pareto_2d_projection_{cluster_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印关键点的详细信息
    print("\n关键点详情:")
    for i, idx in enumerate(key_points):
        print(f"{key_labels[i]}: 孕周={df_check['t'][idx]:.1f}w, 风险={risk[idx]:.4f}, "
              f"达标率={success_rate[idx]:.4f}, 误差敏感性={error_sens[idx]:.4f}")

# 为每个聚类分别生成可视化
for cid in [0, 1]:
    print(f"\n--- 聚类 {cid} 的Pareto前沿可视化 ---")
    # 获取该聚类的诊断数据
    df_check, _ = diagnostic_feasibility(joint_model, data_with_clusters, cid)
    # 绘制前沿投影
    plot_pareto_2d_projection(res=None, df_check=df_check, cluster_id=cid)
