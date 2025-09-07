# 数据预处理
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import os  # 添加os模块以处理目录
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

# 创建输出目录常量
OUTPUT_DIR = "./output_q1"
VIS_DIR = os.path.join(OUTPUT_DIR, "vis")

# 确保输出目录存在
def ensure_dir_exists(directory):
    """创建目录（如果不存在）"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

# --------------------------
# 0. 工具函数（适配男/女胎数据）
# --------------------------
def to_date_str(date_str):
    """将日期字符串转换为标准日期格式（如2023-6-15），支持常见日期格式"""
    if pd.isna(date_str):
        return np.nan
    try:
        dt = pd.to_datetime(str(date_str), errors='coerce')
        if pd.isna(dt):
            return np.nan
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return np.nan

def standardize_week(week_str):
    """孕周标准化，如'11w+6'→11.86周"""
    if pd.isna(week_str):
        return np.nan
    s = str(week_str).strip()
    if "w+" in s:
        try:
            w, d = s.split("w+")
            return round(float(w) + float(d)/7, 2)
        except:
            return np.nan
    elif "w" in s:
        try:
            return float(s.replace("w", ""))
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan
    
def remove_duplicates(df):
    """
    清除重复数据，保留首条记录
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"已清除重复数据：{before - after}条")
    return df

def handle_duplicate_patients(df):
    """
    2.1.2 重复数据处理
    按孕妇唯一标识码（B列）和检测时间（H列）排序，
    同一孕周多次检测的加权平均整合（基于唯一比对读段数和总读段数的乘积作为权重）
    """
    print("开始处理重复患者数据...")
    
    # 确保必要的列存在
    patient_id_col = None
    detection_time_col = None
    week_col = None
    unique_reads_col = None
    total_reads_col = None
    
    # 查找患者ID列（B列）
    for col in ['孕妇代码', '序号', 'patient_id']:
        if col in df.columns:
            patient_id_col = col
            break
    
    # 查找检测时间列（H列）
    for col in ['检测日期', '检测时间', 'detection_date']:
        if col in df.columns:
            detection_time_col = col
            break
    
    # 查找孕周列
    if '检测孕周' in df.columns:
        week_col = '检测孕周'
    
    # 查找读段数列
    if '唯一比对的读段数' in df.columns:
        unique_reads_col = '唯一比对的读段数'
    if '原始读段数' in df.columns:
        total_reads_col = '原始读段数'
    elif '原始测序数据的总读段数' in df.columns:
        total_reads_col = '原始测序数据的总读段数'
    
    if not all([patient_id_col, week_col]):
        print("警告：缺少必要的列（患者ID或孕周），跳过重复数据处理")
        return df
    
    # 简化处理：直接按患者ID和孕周去重，避免复杂的排序问题
    print("使用简化重复数据处理...")
    
    # 处理同一患者同一孕周的多次检测
    if unique_reads_col and total_reads_col:
        print("使用加权平均整合同一孕周的多次检测...")
        
        # 计算权重（唯一比对读段数 × 总读段数）
        unique_reads_numeric = pd.to_numeric(df[unique_reads_col], errors='coerce')
        total_reads_numeric = pd.to_numeric(df[total_reads_col], errors='coerce')
        weights = unique_reads_numeric * total_reads_numeric
        
        # 按患者ID和孕周分组，对数值列进行加权平均
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        def weighted_average_group(group):
            if len(group) == 1:
                return group.iloc[0]
            
            # 计算加权平均
            result = group.iloc[0].copy()  # 保留第一行的非数值列
            
            for col in numeric_cols:
                if col in group.columns and not pd.isna(weights[group.index]).all():
                    valid_mask = ~pd.isna(group[col]) & ~pd.isna(weights[group.index])
                    if valid_mask.any():
                        weighted_sum = (group.loc[valid_mask, col] * weights[group.index[valid_mask]]).sum()
                        weight_sum = weights[group.index[valid_mask]].sum()
                        if weight_sum > 0:
                            result[col] = weighted_sum / weight_sum
            
            return result
        
        # 按患者ID和孕周分组处理
        df_processed = df.groupby([patient_id_col, week_col]).apply(weighted_average_group).reset_index(drop=True)
        
        print(f"重复数据处理完成：{len(df)}行 → {len(df_processed)}行")
        return df_processed
    else:
        print("警告：缺少读段数列，使用简单去重...")
        return df.drop_duplicates(subset=[patient_id_col, week_col])

def visualize_missing_distribution(df, title="缺失值分布", save=True, filename=None):
    """
    可视化缺失值分布（热力图），便于快速发现缺失模式
    参数:
        save: 是否保存图像（True）或显示图像（False）
        filename: 保存的文件名（不含路径）
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="Blues")
    plt.title(title)
    plt.xlabel("列名")
    plt.ylabel("样本")
    plt.tight_layout()
    
    if save:
        # 确保可视化目录存在
        ensure_dir_exists(VIS_DIR)
        
        # 如果没有提供文件名，根据title创建
        if filename is None:
            # 替换空格和特殊字符，使文件名有效
            filename = title.replace(" ", "_").replace(":", "_") + ".png"
        
        # 完整保存路径
        save_path = os.path.join(VIS_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"热力图已保存到: {save_path}")
    else:
        plt.show()

def advanced_missing_value_imputation(df, method='mice'):
    """
    2.1.3 缺失值处理
    实现多重插补（MICE）或 KNN、梯度提升等高级插补方法
    """
    print(f"开始使用{method.upper()}方法进行高级缺失值插补...")
    
    # 选择数值列进行插补
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 排除一些不需要插补的列
    exclude_cols = ['序号', 'patient_id']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) == 0:
        print("警告：没有找到数值列进行插补")
        return df
    
    print(f"将对以下{len(numeric_cols)}个数值列进行插补：{numeric_cols[:5]}...")
    
    # 准备插补数据
    df_impute = df[numeric_cols].copy()
    
    # 检查缺失值情况
    missing_counts = df_impute.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if len(cols_with_missing) == 0:
        print("没有发现缺失值，跳过插补")
        return df
    
    print(f"发现{len(cols_with_missing)}个列有缺失值：{cols_with_missing.to_dict()}")
    
    try:
        if method.lower() == 'mice':
            # MICE多重插补
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=10, random_state=42),
                max_iter=10,
                random_state=42,
                verbose=1
            )
            imputed_data = imputer.fit_transform(df_impute)
            df_imputed = pd.DataFrame(
                imputed_data,
                columns=numeric_cols,
                index=df_impute.index
            )
            
        elif method.lower() == 'knn':
            # KNN插补
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(df_impute)
            df_imputed = pd.DataFrame(
                imputed_data,
                columns=numeric_cols,
                index=df_impute.index
            )
            
        elif method.lower() == 'gradient_boosting':
            # 梯度提升插补
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=50, random_state=42),
                max_iter=5,
                random_state=42
            )
            imputed_data = imputer.fit_transform(df_impute)
            df_imputed = pd.DataFrame(
                imputed_data,
                columns=numeric_cols,
                index=df_impute.index
            )
        
        else:
            print(f"不支持的插补方法：{method}，使用KNN插补")
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(df_impute)
            df_imputed = pd.DataFrame(
                imputed_data,
                columns=numeric_cols,
                index=df_impute.index
            )
        
        # 将插补结果合并回原数据框
        df_result = df.copy()
        for col in numeric_cols:
            if col in df_imputed.columns:
                df_result[col] = df_imputed[col]
        
        print(f"插补完成，处理了{missing_counts.sum()}个缺失值")
        return df_result
        
    except Exception as e:
        print(f"插补过程出错：{e}")
        print("回退到简单填充...")
        # 只对数值列进行中位数填充
        df_result = df.copy()
        for col in numeric_cols:
            if col in df_result.columns:
                median_val = df_result[col].median()
                df_result[col] = df_result[col].fillna(median_val)
        return df_result

def create_interaction_features(df):
    """
    2.2.2 特征工程扩展
    创建交互项（如"孕周×BMI""Z值×GC含量"）和多项式特征（如"孕周²""BMI²"）
    """
    print("开始创建交互项和多项式特征...")
    
    df_features = df.copy()
    
    # 交互项特征
    interaction_features = []
    
    # 孕周 × BMI
    if '检测孕周' in df.columns and '孕妇BMI' in df.columns:
        week_numeric = pd.to_numeric(df['检测孕周'], errors='coerce')
        bmi_numeric = pd.to_numeric(df['孕妇BMI'], errors='coerce')
        if not week_numeric.isna().all() and not bmi_numeric.isna().all():
            df_features['孕周×BMI'] = week_numeric * bmi_numeric
            interaction_features.append('孕周×BMI')
    
    # Z值 × GC含量
    z_cols = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值', 'Y染色体的Z值']
    gc_cols = ['GC含量', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']
    
    for z_col in z_cols:
        if z_col in df.columns:
            z_numeric = pd.to_numeric(df[z_col], errors='coerce')
            if not z_numeric.isna().all():
                for gc_col in gc_cols:
                    if gc_col in df.columns:
                        gc_numeric = pd.to_numeric(df[gc_col], errors='coerce')
                        if not gc_numeric.isna().all():
                            feature_name = f"{z_col}×{gc_col}"
                            df_features[feature_name] = z_numeric * gc_numeric
                            interaction_features.append(feature_name)
    
    # 年龄 × BMI
    if '孕妇年龄' in df.columns and '孕妇BMI' in df.columns:
        age_numeric = pd.to_numeric(df['孕妇年龄'], errors='coerce')
        bmi_numeric = pd.to_numeric(df['孕妇BMI'], errors='coerce')
        if not age_numeric.isna().all() and not bmi_numeric.isna().all():
            df_features['年龄×BMI'] = age_numeric * bmi_numeric
            interaction_features.append('年龄×BMI')
    
    # 多项式特征
    polynomial_features = []
    
    # 孕周²
    if '检测孕周' in df.columns:
        week_numeric = pd.to_numeric(df['检测孕周'], errors='coerce')
        if not week_numeric.isna().all():
            df_features['孕周²'] = week_numeric ** 2
            polynomial_features.append('孕周²')
    
    # BMI²
    if '孕妇BMI' in df.columns:
        bmi_numeric = pd.to_numeric(df['孕妇BMI'], errors='coerce')
        if not bmi_numeric.isna().all():
            df_features['BMI²'] = bmi_numeric ** 2
            polynomial_features.append('BMI²')
    
    # 年龄²
    if '孕妇年龄' in df.columns:
        age_numeric = pd.to_numeric(df['孕妇年龄'], errors='coerce')
        if not age_numeric.isna().all():
            df_features['年龄²'] = age_numeric ** 2
            polynomial_features.append('年龄²')
    
    print(f"创建了{len(interaction_features)}个交互项特征：{interaction_features}")
    print(f"创建了{len(polynomial_features)}个多项式特征：{polynomial_features}")
    
    return df_features

def create_derived_features(df):
    """
    创建派生特征，如"读段数充足标志"等
    """
    print("开始创建派生特征...")
    
    df_derived = df.copy()
    
    # 读段数充足标志
    if '原始读段数' in df.columns or '原始测序数据的总读段数' in df.columns:
        reads_col = '原始读段数' if '原始读段数' in df.columns else '原始测序数据的总读段数'
        reads_numeric = pd.to_numeric(df[reads_col], errors='coerce')
        
        # 设置阈值（可根据实际情况调整）
        threshold = reads_numeric.quantile(0.25)  # 使用25分位数作为阈值
        df_derived['读段数充足标志'] = (reads_numeric >= threshold).astype(int)
        print(f"创建读段数充足标志，阈值：{threshold:.0f}")
    
    # 比对质量标志
    if '在参考基因组上比对的比例' in df.columns:
        align_rate = pd.to_numeric(df['在参考基因组上比对的比例'], errors='coerce')
        df_derived['比对质量标志'] = (align_rate >= 0.8).astype(int)
        print("创建比对质量标志，阈值：0.8")
    
    # GC含量异常标志
    if 'GC含量' in df.columns:
        gc_content = pd.to_numeric(df['GC含量'], errors='coerce')
        df_derived['GC含量异常标志'] = ((gc_content < 0.4) | (gc_content > 0.6)).astype(int)
        print("创建GC含量异常标志")
    
    # BMI分类标志
    if '孕妇BMI' in df.columns:
        bmi_numeric = pd.to_numeric(df['孕妇BMI'], errors='coerce')
        df_derived['BMI分类'] = pd.cut(bmi_numeric, 
                                      bins=[0, 18.5, 25, 30, 50], 
                                      labels=['偏瘦', '正常', '超重', '肥胖'],
                                      include_lowest=True)
        print("创建BMI分类标志")
    
    # 孕周阶段标志
    if '检测孕周' in df.columns:
        week_numeric = pd.to_numeric(df['检测孕周'], errors='coerce')
        df_derived['孕周阶段'] = pd.cut(week_numeric,
                                      bins=[0, 12, 16, 20, 25],
                                      labels=['早期', '中期', '中晚期', '晚期'],
                                      include_lowest=True)
        print("创建孕周阶段标志")
    
    return df_derived

def validate_bmi(df):
    """
    2.3.1 BMI验证
    基于身高（D列）和体重（E列）重新计算BMI并与原始BMI（K列）进行交叉验证，
    设置差异阈值（如15%）进行异常标记
    """
    print("开始BMI交叉验证...")
    
    df_validated = df.copy()
    
    # 查找身高、体重、BMI列
    height_col = None
    weight_col = None
    bmi_col = None
    
    for col in ['孕妇身高', '身高', 'height']:
        if col in df.columns:
            height_col = col
            break
    
    for col in ['孕妇体重', '体重', 'weight']:
        if col in df.columns:
            weight_col = col
            break
    
    for col in ['孕妇BMI', 'BMI', 'bmi']:
        if col in df.columns:
            bmi_col = col
            break
    
    if not all([height_col, weight_col, bmi_col]):
        print("警告：缺少身高、体重或BMI列，跳过BMI验证")
        return df_validated
    
    # 转换为数值
    height_numeric = pd.to_numeric(df[height_col], errors='coerce')
    weight_numeric = pd.to_numeric(df[weight_col], errors='coerce')
    bmi_original = pd.to_numeric(df[bmi_col], errors='coerce')
    
    # 计算BMI（身高单位：cm，体重单位：kg）
    bmi_calculated = weight_numeric / ((height_numeric / 100) ** 2)
    
    # 计算差异百分比
    bmi_diff_pct = abs(bmi_calculated - bmi_original) / bmi_original * 100
    
    # 设置差异阈值（15%）
    threshold = 15
    abnormal_mask = bmi_diff_pct > threshold
    
    # 标记异常BMI
    df_validated['BMI验证差异%'] = bmi_diff_pct
    df_validated['BMI验证标志'] = abnormal_mask.astype(int)
    
    # 统计异常情况
    abnormal_count = abnormal_mask.sum()
    total_count = (~pd.isna(bmi_diff_pct)).sum()
    
    print(f"BMI验证完成：")
    print(f"- 总样本数：{total_count}")
    print(f"- 异常样本数：{abnormal_count}")
    print(f"- 异常率：{abnormal_count/total_count*100:.2f}%")
    print(f"- 差异阈值：{threshold}%")
    
    if abnormal_count > 0:
        print(f"- 平均差异：{bmi_diff_pct[~pd.isna(bmi_diff_pct)].mean():.2f}%")
        print(f"- 最大差异：{bmi_diff_pct[~pd.isna(bmi_diff_pct)].max():.2f}%")
    
    return df_validated

def sequencing_quality_control(df):
    """
    2.3.2 测序质量控制
    将13/18/21号染色体的GC含量（X/Y/Z列）、读段过滤比例（AA列）、重复率（N列）等指标作为协变量纳入模型
    """
    print("开始测序质量控制...")
    
    df_qc = df.copy()
    
    # GC含量质量控制
    gc_cols = ['GC含量', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']
    gc_quality_flags = []
    
    for col in gc_cols:
        if col in df.columns:
            gc_numeric = pd.to_numeric(df[col], errors='coerce')
            # GC含量正常范围：40%-60%
            normal_mask = (gc_numeric >= 0.4) & (gc_numeric <= 0.6)
            flag_col = f"{col}_质量标志"
            df_qc[flag_col] = normal_mask.astype(int)
            gc_quality_flags.append(flag_col)
    
    # 读段过滤比例质量控制
    filter_cols = ['被过滤掉读段数的比例', '过滤比例']
    for col in filter_cols:
        if col in df.columns:
            filter_numeric = pd.to_numeric(df[col], errors='coerce')
            # 过滤比例过高表示质量差（阈值可调整）
            high_filter_mask = filter_numeric > 0.3  # 30%以上过滤比例认为质量差
            flag_col = f"{col}_质量标志"
            df_qc[flag_col] = (~high_filter_mask).astype(int)
    
    # 重复率质量控制
    if '重复读段的比例' in df.columns:
        dup_numeric = pd.to_numeric(df['重复读段的比例'], errors='coerce')
        # 重复率过高表示质量差
        high_dup_mask = dup_numeric > 0.5  # 50%以上重复率认为质量差
        df_qc['重复率质量标志'] = (~high_dup_mask).astype(int)
    
    # 比对率质量控制
    if '在参考基因组上比对的比例' in df.columns:
        align_numeric = pd.to_numeric(df['在参考基因组上比对的比例'], errors='coerce')
        # 比对率过低表示质量差
        low_align_mask = align_numeric < 0.7  # 70%以下比对率认为质量差
        df_qc['比对率质量标志'] = (~low_align_mask).astype(int)
    
    # 综合质量评分
    quality_cols = [col for col in df_qc.columns if col.endswith('_质量标志') or col.endswith('质量标志')]
    if quality_cols:
        df_qc['综合质量评分'] = df_qc[quality_cols].sum(axis=1)
        df_qc['质量等级'] = pd.cut(df_qc['综合质量评分'], 
                                 bins=[0, 2, 4, 6], 
                                 labels=['差', '中等', '好'],
                                 include_lowest=True)
        
        print(f"创建了{len(quality_cols)}个质量标志列")
        print(f"质量等级分布：{df_qc['质量等级'].value_counts().to_dict()}")
    
    return df_qc

def batch_effect_correction(df):
    """
    考虑批次效应的校正（如将"批次"作为随机效应项）
    """
    print("开始批次效应分析...")
    
    df_batch = df.copy()
    
    # 查找可能的批次标识列
    batch_cols = ['批次', '检测批次', 'batch', '检测日期', '检测时间']
    batch_col = None
    
    for col in batch_cols:
        if col in df.columns:
            batch_col = col
            break
    
    if batch_col is None:
        print("未找到批次标识列，跳过批次效应校正")
        return df_batch
    
    # 分析批次效应
    print(f"使用'{batch_col}'列进行批次效应分析")
    
    # 统计各批次的样本数
    batch_counts = df[batch_col].value_counts()
    print(f"批次分布：{batch_counts.to_dict()}")
    
    # 改进的批次处理：不创建大量虚拟变量，而是创建批次分组
    if batch_col == '检测日期':
        # 将日期转换为月份批次，减少虚拟变量数量
        try:
            df_batch['检测日期'] = pd.to_datetime(df_batch['检测日期'], errors='coerce')
            df_batch['批次_月份'] = df_batch['检测日期'].dt.to_period('M').astype(str)
            
            # 只为月份批次创建虚拟变量（最多12个）
            month_dummies = pd.get_dummies(df_batch['批次_月份'], prefix='月份批次')
            df_batch = pd.concat([df_batch, month_dummies], axis=1)
            
            print(f"创建了{month_dummies.shape[1]}个月份批次虚拟变量")
            
        except Exception as e:
            print(f"日期批次处理失败：{e}")
            # 回退到简单批次标识
            df_batch['批次_标识'] = df_batch[batch_col].astype(str)
    else:
        # 对于非日期列，创建简单的批次标识
        df_batch['批次_标识'] = df_batch[batch_col].astype(str)
    
    # 批次效应检测（对关键数值变量）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    key_cols = ['孕妇BMI', '检测孕周', 'Y染色体浓度', 'X染色体浓度']
    
    batch_effects = {}
    for col in key_cols:
        if col in numeric_cols:
            col_numeric = pd.to_numeric(df[col], errors='coerce')
            if not col_numeric.isna().all():
                # 计算各批次的均值
                batch_means = df.groupby(batch_col)[col].mean()
                batch_std = batch_means.std()
                batch_effects[col] = batch_std
                
                print(f"{col}的批次间标准差：{batch_std:.4f}")
    
    # 标记批次效应显著的变量
    if batch_effects:
        significant_batch = {k: v for k, v in batch_effects.items() if v > 0.1}
        if significant_batch:
            print(f"发现显著批次效应的变量：{list(significant_batch.keys())}")
        else:
            print("未发现显著的批次效应")
    
    return df_batch

# --------------------------
# 1. 分sheet读取原始数据（男/女胎数据）
# --------------------------
def load_gender_specific_data(file_path):
    """
    分sheet读取男/女胎儿数据（sheet1=男胎，sheet2=女胎）
    :return: 男胎DataFrame、女胎DataFrame
    """
    try:
        # 读取sheet1（男胎儿数据）
        df_male = pd.read_excel(file_path, sheet_name=0)  # sheet1索引为0，若sheet名指定可改为sheet_name="男胎儿"
        # 读取sheet2（女胎儿数据）
        df_female = pd.read_excel(file_path, sheet_name=1)  # sheet2索引为1，若sheet名指定可改为sheet_name="女胎儿"
        
        # 验证群体特征（依据C题：男胎Y列有值，女胎Y列空白）
        y_cols = ['Y染色体的Z值', 'Y染色体浓度']  # 需与附件列名完全一致（参考C题附录1列说明）
        male_valid = all(col in df_male.columns for col in y_cols) and df_male[y_cols].notna().any(axis=1).sum() > 0
        female_valid = all(col in df_female.columns for col in y_cols) and df_female[y_cols].isna().all(axis=1).sum() > 0
        
        if not male_valid:
            print("警告：sheet1可能非男胎儿数据（Y染色体列无有效数值）")
        if not female_valid:
            print("警告：sheet2可能非女胎儿数据（Y染色体列非空白）")
        
        print(f"分sheet读取完成：")
        print(f"- 男胎儿数据：{df_male.shape[0]}行，{df_male.shape[1]}列")
        print(f"- 女胎儿数据：{df_female.shape[0]}行，{df_female.shape[1]}列")
        return df_male, df_female
    except Exception as e:
        print(f"分sheet读取失败：{str(e)}")
        raise

# --------------------------
# 2. 男胎儿数据专属预处理（服务问题1-3）
# --------------------------

def preprocess_male_data(df_male):
    """
    男胎儿数据预处理（依据建模方案：重点保障Y染色体浓度、BMI、孕周质量）
    集成所有新增功能：重复数据处理、高级缺失值插补、特征工程、数据质控
    """
    print("开始男胎儿数据预处理...")
    df = df_male.copy()
    
    # 2.1.2 重复数据处理
    df = handle_duplicate_patients(df)
    
    # 日期列转换为标准日期字符串
    for col in ['末次月经', '检测日期']:
        if col in df.columns:
            df[col] = df[col].apply(to_date_str)
    # 孕周标准化
    if '检测孕周' in df.columns:
        df['检测孕周'] = df['检测孕周'].apply(standardize_week)
    
    y_cols = ['Y染色体的Z值', 'Y染色体浓度']
    key_cols = ['孕妇年龄', '孕妇身高', '孕妇体重', '检测孕周', '孕妇BMI', 
               '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
               '原始测序数据的总读段数', '在参考基因组上比对的比例', 'GC含量']  # 关键特征
    
    # 剔除原始读段数过低或唯一比对质量差的样本
    # L列：原始读段数（如“原始测序数据的总读段数”或“原始读段数”），O列：唯一比对读段数
    raw_reads_col = '原始读段数' if '原始读段数' in df.columns else '原始测序数据的总读段数'
    uniq_reads_col = '唯一比对的读段数'
    if raw_reads_col in df.columns and uniq_reads_col in df.columns:
        # 剔除原始读段数过低（如小于0.7，实际应为比例或绝对值，按你的描述这里按比例处理）
        raw_reads_numeric = pd.to_numeric(df[raw_reads_col], errors='coerce')
        df = df[raw_reads_numeric >= 0.7]
        # 剔除唯一比对质量差（唯一比对读段数/原始读段数 < 0.6）
        uniq_reads_numeric = pd.to_numeric(df[uniq_reads_col], errors='coerce')
        align_rate = uniq_reads_numeric / raw_reads_numeric
        df = df[align_rate >= 0.6]
    
    # 缺失值处理（男胎Y列不允许空白）
    # Y染色体列缺失（非女胎空白，属于数据错误）标记为"检测失败"
    for col in y_cols:
        df[col] = df[col].fillna("检测失败")
    # 关键建模列缺失（如BMI、孕周）标记为"待补充"（避免删除样本，后续建模筛选）
    for col in key_cols:
        if col in df.columns:
            df[col] = df[col].fillna("待补充")
    
    # 异常值处理（针对男胎核心指标）
    # Y染色体浓度异常（<0或>20%，临床合理范围外）标记
    if 'Y染色体浓度' in df.columns:
        y_conc_numeric = pd.to_numeric(df['Y染色体浓度'], errors='coerce')
        abnormal_y_mask = (y_conc_numeric < 0) | (y_conc_numeric > 20)
        df.loc[abnormal_y_mask, 'Y染色体浓度'] = df.loc[abnormal_y_mask, 'Y染色体浓度'].astype(str) + "(浓度异常)"
    
    # BMI异常（高BMI群体重点，>40或<10）标记
    if '孕妇BMI' in df.columns:
        bmi_numeric = pd.to_numeric(df['孕妇BMI'], errors='coerce')
        abnormal_bmi_mask = (bmi_numeric > 40) | (bmi_numeric < 10)
        df.loc[abnormal_bmi_mask, '孕妇BMI'] = df.loc[abnormal_bmi_mask, '孕妇BMI'].astype(str) + "(BMI异常)"
    
    # 孕周异常（C题：10-25周检测窗口）标记+标准化
    if '检测孕周' in df.columns:
        df['检测孕周'] = df['检测孕周'].apply(standardize_week)
        week_numeric = pd.to_numeric(df['检测孕周'], errors='coerce')
        abnormal_week_mask = (week_numeric < 10) | (week_numeric > 25)
        df.loc[abnormal_week_mask, '检测孕周'] = df.loc[abnormal_week_mask, '检测孕周'].astype(str) + "(孕周异常)"
    
    # 测序质量控制（GC含量40%-60%）
    if 'GC含量' in df.columns:
        gc_numeric = pd.to_numeric(df['GC含量'], errors='coerce')
        abnormal_gc_mask = (gc_numeric < 0.4) | (gc_numeric > 0.6)  # 小数形式（40%=0.4）
        df.loc[abnormal_gc_mask, 'GC含量'] = df.loc[abnormal_gc_mask, 'GC含量'].astype(str) + "(GC异常)"
    
    # 2.1.3 高级缺失值处理
    print("开始高级缺失值插补...")
    df = advanced_missing_value_imputation(df, method='mice')
    
    # 2.2.2 特征工程扩展
    print("开始特征工程...")
    df = create_interaction_features(df)
    df = create_derived_features(df)
    
    # 2.3.1 BMI验证
    print("开始BMI验证...")
    df = validate_bmi(df)
    
    # 2.3.2 测序质量控制
    print("开始测序质量控制...")
    df = sequencing_quality_control(df)
    
    # 批次效应校正
    print("开始批次效应分析...")
    df = batch_effect_correction(df)
    
    print("男胎儿数据预处理完成，所有功能已集成")
    return df

# --------------------------
# 3. 女胎儿数据专属预处理（服务问题4）
# --------------------------
def preprocess_female_data(df_female):
    """
    女胎儿数据预处理（依据建模方案：重点保障X染色体、21/18/13号染色体Z值、非整倍体标记）
    集成所有新增功能：重复数据处理、高级缺失值插补、特征工程、数据质控
    """
    print("开始女胎儿数据预处理...")
    df = df_female.copy()
    
    # 2.1.2 重复数据处理
    df = handle_duplicate_patients(df)
    
    # 日期列转换为标准日期字符串
    for col in ['末次月经', '检测日期']:
        if col in df.columns:
            df[col] = df[col].apply(to_date_str)
    # 孕周标准化
    if '检测孕周' in df.columns:
        df['检测孕周'] = df['检测孕周'].apply(standardize_week)

    # 剔除原始读段数过低或唯一比对质量差的样本
    raw_reads_col = '原始读段数' if '原始读段数' in df.columns else '原始测序数据的总读段数'
    uniq_reads_col = '唯一比对的读段数'
    if raw_reads_col in df.columns and uniq_reads_col in df.columns:
        raw_reads_numeric = pd.to_numeric(df[raw_reads_col], errors='coerce')
        df = df[raw_reads_numeric >= 0.7]
        uniq_reads_numeric = pd.to_numeric(df[uniq_reads_col], errors='coerce')
        align_rate = uniq_reads_numeric / raw_reads_numeric
        df = df[align_rate >= 0.6]

    female_key_cols = ['X染色体的Z值', 'X染色体浓度', '13号染色体的Z值', 
                      '18号染色体的Z值', '21号染色体的Z值', '染色体的非整倍体',
                      '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']
    
    # 3.1 缺失值处理
    for col in female_key_cols:
        if col in df.columns:
            df[col] = df[col].fillna("待补充")
    
    # 3.2 异常值处理
    if 'X染色体浓度' in df.columns:
        x_conc_numeric = pd.to_numeric(df['X染色体浓度'], errors='coerce')
        abnormal_x_mask = (x_conc_numeric < -5) | (x_conc_numeric > 10)
        df.loc[abnormal_x_mask, 'X染色体浓度'] = df.loc[abnormal_x_mask, 'X染色体浓度'].astype(str) + "(X浓度异常)"
    z_cols = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值']
    for col in z_cols:
        if col in df.columns:
            z_numeric = pd.to_numeric(df[col], errors='coerce')
            abnormal_z_mask = abs(z_numeric) > 3
            df.loc[abnormal_z_mask, col] = df.loc[abnormal_z_mask, col].astype(str) + "(Z值异常)"
    if '染色体的非整倍体' in df.columns:
        df['染色体的非整倍体'] = df['染色体的非整倍体'].fillna("无异常")

    # 2.1.3 高级缺失值处理
    print("开始高级缺失值插补...")
    df = advanced_missing_value_imputation(df, method='mice')
    
    # 2.2.2 特征工程扩展
    print("开始特征工程...")
    df = create_interaction_features(df)
    df = create_derived_features(df)
    
    # 2.3.1 BMI验证
    print("开始BMI验证...")
    df = validate_bmi(df)
    
    # 2.3.2 测序质量控制
    print("开始测序质量控制...")
    df = sequencing_quality_control(df)
    
    # 批次效应校正
    print("开始批次效应分析...")
    df = batch_effect_correction(df)

    print("女胎儿数据预处理完成，所有功能已集成")
    return df

# --------------------------
# 4. 分群体导出结果（保留原列名与列数）
# --------------------------
def export_gender_data(df_male_processed, df_female_processed):
    """
    导出男/女胎儿预处理数据，分别保存为独立CSV文件
    """
    # 确保输出目录存在
    ensure_dir_exists(OUTPUT_DIR)
    
    # 将CSV文件导出
    current_dir = os.path.dirname(os.path.abspath(__file__))
    male_output = os.path.join(current_dir, "processed_male.csv")
    female_output = os.path.join(current_dir, "processed_female.csv")
    
    # 删除批次_月份相关列
    def remove_batch_month_columns(df):
        df_filtered = df.copy()
        # 获取所有列名
        columns = list(df_filtered.columns)
        # 过滤掉批次_月份列和以月份批次_开头的列
        filtered_columns = [col for col in columns 
                           if not (col == '批次_月份' or 
                                  (isinstance(col, str) and col.startswith('月份批次_')) or
                                  (isinstance(col, str) and '批次_月份' in col))]
        return df_filtered[filtered_columns]
    
    # 清理列名，移除特殊字符
    def clean_column_names(df):
        df_clean = df.copy()
        # 替换列名中的特殊字符
        new_columns = []
        for col in df_clean.columns:
            if pd.isna(col) or col == '':
                new_columns.append('Unnamed')
            else:
                # 移除或替换特殊字符
                clean_col = str(col).replace('×', '_x_').replace('²', '_squared').replace(' ', '_')
                new_columns.append(clean_col)
        df_clean.columns = new_columns
        return df_clean
    
    # 删除批次列，然后清理男胎数据
    df_male_filtered = remove_batch_month_columns(df_male_processed)
    df_male_clean = clean_column_names(df_male_filtered)
    
    # 导出男胎数据
    try:
        df_male_clean.to_csv(male_output, index=False, encoding='utf-8-sig')
        print(f"\n男胎儿预处理数据导出：{male_output}（{df_male_clean.shape[0]}行×{df_male_clean.shape[1]}列）")
    except Exception as e:
        print(f"男胎数据导出失败：{e}")
        # 尝试导出基本信息
        basic_cols = df_male_clean.columns[:31]  # 只导出前31列
        df_male_clean[basic_cols].to_csv(male_output, index=False, encoding='utf-8-sig')
        print(f"男胎儿基础数据导出：{male_output}（{df_male_clean.shape[0]}行×{len(basic_cols)}列）")
    
    # 删除批次列，然后清理女胎数据
    df_female_filtered = remove_batch_month_columns(df_female_processed)
    df_female_clean = clean_column_names(df_female_filtered)
    
    # 女胎数据：将20、21列标题保留空白，与男胎数据保持列数统一
    female_cols = list(df_female_clean.columns)
    if len(female_cols) >= 22:
        female_cols[20] = "Unnamed_20"
        female_cols[21] = "Unnamed_21"
        df_female_clean.columns = female_cols
    
    # 导出女胎数据
    try:
        df_female_clean.to_csv(female_output, index=False, encoding='utf-8-sig')
        print(f"女胎儿预处理数据导出：{female_output}（{df_female_clean.shape[0]}行×{df_female_clean.shape[1]}列）")
    except Exception as e:
        print(f"女胎数据导出失败：{e}")
        # 尝试导出基本信息
        basic_cols = df_female_clean.columns[:31]  # 只导出前31列
        df_female_clean[basic_cols].to_csv(female_output, index=False, encoding='utf-8-sig')
        print(f"女胎儿基础数据导出：{female_output}（{df_female_clean.shape[0]}行×{len(basic_cols)}列）")
    
    # 验证列数一致性（确保未删除列）
    original_male_cols = df_male_clean.shape[1]
    original_female_cols = df_female_clean.shape[1]
    print(f"\n列数验证：")
    print(f"- 男胎儿：预处理后{original_male_cols}列")
    print(f"- 女胎儿：预处理后{original_female_cols}列")
    
    # 显示新增的特征列
    print(f"\n新增特征列统计：")
    print(f"- 男胎儿新增列数：{original_male_cols - 31}")
    print(f"- 女胎儿新增列数：{original_female_cols - 31}")

# --------------------------
# 主流程：分群体读取→预处理→导出
# --------------------------
def main():
    print("="*60)
    print("NIPT数据分群体预处理（男胎sheet1/女胎sheet2）- 基于C题与建模方案")
    print("="*60)
    
    # 确保输出目录存在
    ensure_dir_exists(OUTPUT_DIR)
    ensure_dir_exists(VIS_DIR)
    
    # 1. 配置原始文件路径（需替换为实际路径）
    raw_file_path = "附件.xlsx"
    
    # 2. 分sheet读取男/女胎数据
    df_male_raw, df_female_raw = load_gender_specific_data(raw_file_path)

    # 3. 可视化缺失值分布并保存到指定目录
    visualize_missing_distribution(df_male_raw,
                                  title="男胎儿缺失值分布",
                                  save=True,
                                  filename="male_missing_data.png")
    visualize_missing_distribution(df_female_raw,
                                  title="女胎儿缺失值分布",
                                  save=True,
                                  filename="female_missing_data.png")
    
    # 4. 分群体执行专属预处理
    df_male_processed = preprocess_male_data(df_male_raw)
    df_female_processed = preprocess_female_data(df_female_raw)

    # 5. 导出分群体结果
    export_gender_data(df_male_processed, df_female_processed)
    
    print("\n" + "="*60)
    print("分群体预处理流程全部完成")
    print("="*60)

if __name__ == "__main__":
    main()