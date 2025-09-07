# 数据预处理
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import os  # 添加os模块以处理目录

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
    """
    df = df_male.copy()
    df = remove_duplicates(df)
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
    
    print("男胎儿数据预处理完成，核心指标已质控")
    return df

# --------------------------
# 3. 女胎儿数据专属预处理（服务问题4）
# --------------------------
def preprocess_female_data(df_female):
    """
    女胎儿数据预处理（依据建模方案：重点保障X染色体、21/18/13号染色体Z值、非整倍体标记）
    """
    df = df_female.copy()
    df = remove_duplicates(df)
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

    print("女胎儿数据预处理完成，异常判定指标已质控")
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
    
    # 导出男胎数据
    df_male_processed.to_csv(male_output, index=False)
    print(f"\n男胎儿预处理数据导出：{male_output}（{df_male_processed.shape[0]}行×{df_male_processed.shape[1]}列）")
    
    # 女胎数据：将20、21列标题保留空白，与男胎数据保持列数统一
    female_cols = list(df_female_processed.columns)
    if len(female_cols) >= 22:
        female_cols[20] = ""
        female_cols[21] = ""
        df_female_processed.columns = female_cols
    # 导出女胎数据
    df_female_processed.to_csv(female_output, index=False)
    print(f"女胎儿预处理数据导出：{female_output}（{df_female_processed.shape[0]}行×{df_female_processed.shape[1]}列）")
    
    # 验证列数一致性（确保未删除列）
    original_male_cols = df_male_processed.shape[1]
    original_female_cols = df_female_processed.shape[1]
    print(f"\n列数验证：")
    print(f"- 男胎儿：预处理后{original_male_cols}列（与原始一致）")
    print(f"- 女胎儿：预处理后{original_female_cols}列（与原始一致）")

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
    
    # 3. 分群体执行专属预处理
    df_male_processed = preprocess_male_data(df_male_raw)
    df_female_processed = preprocess_female_data(df_female_raw)

    # 4. 可视化缺失值分布并保存到指定目录
    visualize_missing_distribution(df_male_processed, 
                                  title="男胎儿缺失值分布", 
                                  save=True, 
                                  filename="male_missing_data.png")
    visualize_missing_distribution(df_female_processed, 
                                  title="女胎儿缺失值分布", 
                                  save=True, 
                                  filename="female_missing_data.png")

    # 5. 导出分群体结果
    export_gender_data(df_male_processed, df_female_processed)
    
    print("\n" + "="*60)
    print("分群体预处理流程全部完成")
    print("="*60)

if __name__ == "__main__":
    main()