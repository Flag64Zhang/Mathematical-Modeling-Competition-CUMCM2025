import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import shap
import pickle
import os
from sklearn.inspection import partial_dependence
try:
    # 尝试导入新API
    from sklearn.inspection import PartialDependenceDisplay
except ImportError:
    # 如果不可用，将创建一个兼容的替代函数
    pass

# 创建输出目录
OUT_DIR = "output_q4"
VIS_DIR = os.path.join(OUT_DIR, "vis")  # 新增可视化专用目录

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
    print(f"创建输出目录: {OUT_DIR}")

if not os.path.exists(VIS_DIR):  # 确保可视化目录存在
    os.makedirs(VIS_DIR)
    print(f"创建可视化目录: {VIS_DIR}")

# 设置随机种子，保证可复现性
SEED = 42
np.random.seed(SEED)

# 添加设置中文字体的函数
def set_chinese_font():
    """设置中文字体，根据不同操作系统选择合适的字体"""
    # 导入matplotlib模块确保可用
    import matplotlib as mpl
    import platform
    from matplotlib import rcParams
    import matplotlib.font_manager as fm
    
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
    
    # 确保能显示负号 - 更强制地设置
    plt.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['axes.unicode_minus'] = False  # 同时设置matplotlib模块级别的参数
    
    # 尝试使用更强大的配置
    rcParams['font.family'] = 'sans-serif'
    
    # 为shap模块设置特定配置
    try:
        import shap.plots as shap_plots
        # 如果shap.plots模块有_waterfall属性，尝试修改其中的字体设置
        if hasattr(shap_plots, '_waterfall'):
            shap_plots._waterfall.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    return chinese_font is not None

# 在初始化时调用字体设置函数
set_chinese_font()

class FemaleFetalModel:
    def __init__(self, data_path=None):
        """初始化模型类"""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        self.xgb_model = None
        self.lgb_model = None
        self.ensemble_model = None
        self.best_threshold = 0.065  # 默认阈值
        self.feature_names = ['21-Z', '18-Z', '13-Z', 'GC异常标志', '读段数', 'BMI', '孕周']

        # 如果提供了数据路径，加载数据
        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path):
        """加载数据并进行初步处理"""
        try:
            self.data = pd.read_csv(data_path)
            print(f"成功加载数据，共 {self.data.shape[0]} 样本，{self.data.shape[1]} 特征")

            # 尝试列名映射
            # 常见列名映射到标准名称
            column_mapping = {
                # 原列名 -> 模型期望的列名
                '21-Z标准差': '21-Z', 
                '18-Z标准差': '18-Z',
                '13-Z标准差': '13-Z',
                'GC_abnormal': 'GC异常标志',
                '质量异常标志': 'GC异常标志',
                '总读段数': '读段数',
                'reads': '读段数',
                '孕妇BMI': 'BMI',
                '检测孕周': '孕周',
                '异常标志': 'AE'
            }
            
            # 显示原始列名，帮助调试
            print("原始列名:", self.data.columns.tolist())
            
            # 应用映射
            self.data = self.data.rename(columns={k: v for k, v in column_mapping.items() 
                                            if k in self.data.columns})
            
            # 检查必要的列是否存在
            required_columns = self.feature_names + ['AE']
            missing_cols = [col for col in required_columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"数据缺少必要的列: {missing_cols}")

            # 划分特征和目标变量
            self.X = self.data[self.feature_names]
            self.y = self.data['AE'].astype(int)  # 确保目标变量是整数类型

            # 查看类别分布
            print(f"阳性样本比例: {self.y.mean():.2%}")

            return True
        except Exception as e:
            print(f"加载数据失败: {str(e)}")
            return False

    def preprocess_data(self, test_size=0.3, use_smote=True):
        """数据预处理，包括划分训练集和测试集，以及SMOTE过采样"""
        if self.data is None:
            print("请先加载数据")
            return False

        # 划分训练集和测试集（保持时间顺序的外部验证）
        # 这里假设数据已经按时间排序，前70%作为训练集，后30%作为验证集
        train_size = int(len(self.data) * (1 - test_size))
        self.X_train, self.X_test = self.X.iloc[:train_size], self.X.iloc[train_size:]
        self.y_train, self.y_test = self.y.iloc[:train_size], self.y.iloc[train_size:]

        print(f"训练集大小: {self.X_train.shape[0]}, 测试集大小: {self.X_test.shape[0]}")
        print(f"训练集阳性比例: {self.y_train.mean():.2%}, 测试集阳性比例: {self.y_test.mean():.2%}")

        # 特征标准化
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # 使用SMOTE处理类别不平衡
        if use_smote:
            smote = SMOTE(random_state=SEED, k_neighbors=5)
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
                self.X_train_scaled, self.y_train)
            print(f"SMOTE后训练集大小: {self.X_train_resampled.shape[0]}")
            print(f"SMOTE后阳性比例: {self.y_train_resampled.mean():.2%}")
        else:
            self.X_train_resampled, self.y_train_resampled = self.X_train_scaled, self.y_train

        return True

    def train_xgboost(self, params=None, perform_grid_search=False):
        """训练XGBoost模型"""
        if self.X_train_resampled is None:
            print("请先预处理数据")
            return False

        # 默认参数
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'scale_pos_weight': 1,  # 由于已经使用SMOTE，这里设为1
                'random_state': SEED
            }

        # 如果需要网格搜索调参
        if perform_grid_search:
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [50, 100, 200]
            }

            grid_search = GridSearchCV(
                estimator=xgb.XGBClassifier(**params),
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(self.X_train_resampled, self.y_train_resampled)
            print(f"最佳参数: {grid_search.best_params_}")
            print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")
            self.xgb_model = grid_search.best_estimator_
        else:
            # 兼容不同版本的XGBoost - 最简单的方法，移除所有可能有问题的参数
            print("使用简化的XGBoost训练（无早停）")
            self.xgb_model = xgb.XGBClassifier(**params)
            
            try:
                # 最基础的fit调用，不使用任何额外参数
                self.xgb_model.fit(self.X_train_resampled, self.y_train_resampled)
            except Exception as e:
                print(f"XGBoost训练出错: {e}")
                # 尝试回退到更简单的模型或参数
                simple_params = {
                    'max_depth': 3,
                    'n_estimators': 50,
                    'random_state': SEED
                }
                print("尝试使用更简单的模型参数")
                self.xgb_model = xgb.XGBClassifier(**simple_params)
                self.xgb_model.fit(self.X_train_resampled, self.y_train_resampled)

        # 评估模型
        self.evaluate_model(self.xgb_model, "XGBoost")
        return True

    def train_lightgbm(self, params=None, perform_grid_search=False):
        """训练LightGBM模型"""
        if self.X_train_resampled is None:
            print("请先预处理数据")
            return False

        # 默认参数
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': SEED
            }

        # 如果需要网格搜索调参
        if perform_grid_search:
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [50, 100, 200]
            }

            grid_search = GridSearchCV(
                estimator=lgb.LGBMClassifier(**params),
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(self.X_train_resampled, self.y_train_resampled)
            print(f"最佳参数: {grid_search.best_params_}")
            print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")
            self.lgb_model = grid_search.best_estimator_
        else:
            print("使用简化的LightGBM训练（无早停）")
            self.lgb_model = lgb.LGBMClassifier(**params)
            
            try:
                # 最基础的fit调用，不使用任何额外参数
                self.lgb_model.fit(self.X_train_resampled, self.y_train_resampled)
            except Exception as e:
                print(f"LightGBM训练出错: {e}")
                # 尝试回退到更简单的模型或参数
                simple_params = {
                    'max_depth': 3,
                    'n_estimators': 50,
                    'random_state': SEED
                }
                print("尝试使用更简单的模型参数")
                self.lgb_model = lgb.LGBMClassifier(**simple_params)
                self.lgb_model.fit(self.X_train_resampled, self.y_train_resampled)

        # 评估模型
        self.evaluate_model(self.lgb_model, "LightGBM")
        return True

    def train_ensemble(self):
        """训练集成模型（模型融合）"""
        if self.xgb_model is None or self.lgb_model is None:
            print("请先训练XGBoost和LightGBM模型")
            return False

        # 使用投票分类器进行模型融合
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('xgb', self.xgb_model),
                ('lgb', self.lgb_model)
            ],
            voting='soft'  # 使用预测概率的加权平均
        )

        self.ensemble_model.fit(self.X_train_resampled, self.y_train_resampled)
        self.evaluate_model(self.ensemble_model, "集成模型")
        return True

    def optimize_threshold(self, model=None, fn_weight=10, fp_weight=1):
        """优化分类阈值，最小化期望代价"""
        if model is None:
            if self.xgb_model is not None:
                model = self.xgb_model
            else:
                print("请先训练模型")
                return False

        # 获取预测概率
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]

        # 网格搜索最佳阈值
        thresholds = np.arange(0.005, 1.0, 0.005)
        min_cost = float('inf')
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(self.y_test, y_pred)

            # 计算假阴性和假阳性
            fn = cm[1, 0] if cm.shape[0] > 1 else 0
            fp = cm[0, 1] if cm.shape[0] > 1 else 0

            # 计算期望代价
            cost = fn_weight * fn + fp_weight * fp

            if cost < min_cost:
                min_cost = cost
                best_threshold = threshold

        self.best_threshold = best_threshold
        print(f"最佳阈值: {best_threshold:.3f}, 最小期望代价: {min_cost}")

        # 使用最佳阈值评估模型
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        self._print_metrics(self.y_test, y_pred, y_pred_proba, "使用最佳阈值的模型")
        return best_threshold

    def evaluate_model(self, model, model_name):
        """评估模型性能"""
        # 获取预测结果
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= self.best_threshold).astype(int)

        # 计算评估指标
        self._print_metrics(self.y_test, y_pred, y_pred_proba, model_name)

        # 绘制ROC曲线
        self.plot_roc_curve(self.y_test, y_pred_proba, model_name)

        # 绘制混淆矩阵
        self.plot_confusion_matrix(self.y_test, y_pred, model_name)

        return {
            'auc': roc_auc_score(self.y_test, y_pred_proba),
            'sensitivity': recall_score(self.y_test, y_pred),
            'specificity': recall_score(self.y_test, y_pred, pos_label=0),
            'ppv': precision_score(self.y_test, y_pred),
            'npv': precision_score(self.y_test, y_pred, pos_label=0)
        }

    def _print_metrics(self, y_true, y_pred, y_pred_proba, model_name):
        """打印评估指标"""
        auc = roc_auc_score(y_true, y_pred_proba)
        sensitivity = recall_score(y_true, y_pred)
        specificity = recall_score(y_true, y_pred, pos_label=0)
        ppv = precision_score(y_true, y_pred)
        npv = precision_score(y_true, y_pred, pos_label=0)

        print(f"\n{model_name} 评估指标:")
        print(f"AUC: {auc:.4f}")
        print(f"敏感性 (Se): {sensitivity:.2%}")
        print(f"特异性 (Sp): {specificity:.2%}")
        print(f"阳性预测值 (PPV): {ppv:.2%}")
        print(f"阴性预测值 (NPV): {npv:.2%}")
        print(classification_report(y_true, y_pred))

    def plot_roc_curve(self, y_true, y_pred_proba, model_name):
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假阳性率 (FPR)')
        plt.ylabel('真阳性率 (TPR)')
        plt.title(f'{model_name} ROC曲线')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(VIS_DIR, f'{model_name}_roc_curve.png'))  # 修改保存路径
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['正常', '异常'],
                    yticklabels=['正常', '异常'])
        plt.xlabel('预测结果')
        plt.ylabel('实际结果')
        plt.title(f'{model_name} 混淆矩阵')
        plt.savefig(os.path.join(VIS_DIR, f'{model_name}_confusion_matrix.png'))  # 修改保存路径
        plt.close()

    def threshold_sensitivity_analysis(self):
        """阈值敏感性分析（3.5%-4.5%）"""
        if self.xgb_model is None:
            print("请先训练XGBoost模型")
            return False

        print("\n阈值敏感性分析:")
        results = []
        base_auc = roc_auc_score(self.y_test, self.xgb_model.predict_proba(self.X_test_scaled)[:, 1])
        base_sensitivity = recall_score(self.y_test, (
                    self.xgb_model.predict_proba(self.X_test_scaled)[:, 1] >= self.best_threshold).astype(int))
        base_specificity = recall_score(self.y_test, (
                    self.xgb_model.predict_proba(self.X_test_scaled)[:, 1] >= self.best_threshold).astype(int),
                                        pos_label=0)

        for threshold in np.arange(3.5, 4.6, 0.1):
            # 这里模拟男胎阈值变化对女胎模型的影响
            # 实际应用中可能需要更复杂的模拟或数据调整
            y_pred_proba = self.xgb_model.predict_proba(self.X_test_scaled)[:, 1]
            y_pred = (y_pred_proba >= self.best_threshold * (threshold / 4.0)).astype(int)  # 简单模拟阈值变化的影响

            auc = roc_auc_score(self.y_test, y_pred_proba)
            sensitivity = recall_score(self.y_test, y_pred)
            specificity = recall_score(self.y_test, y_pred, pos_label=0)

            se_change = (sensitivity - base_sensitivity) * 100
            sp_change = (specificity - base_specificity) * 100

            results.append({
                '男胎阈值(%)': threshold,
                '女胎AUC': auc,
                'Se变化(pp)': se_change,
                'Sp变化(pp)': sp_change
            })

            print(f"男胎阈值: {threshold:.1f}%, AUC: {auc:.4f}, "
                  f"Se变化: {se_change:.1f}pp, Sp变化: {sp_change:.1f}pp")

        # 保存结果为DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(OUT_DIR, 'threshold_sensitivity_analysis.csv'), index=False)
        return results_df

    def weight_sensitivity_analysis(self):
        """代价权重多情景分析"""
        if self.xgb_model is None:
            print("请先训练XGBoost模型")
            return False

        print("\n代价权重敏感性分析:")
        weight_scenarios = [
            (5, 1, "资源充足"),
            (10, 1, "默认"),
            (15, 1, "高风险人群")
        ]

        results = []

        for fn_weight, fp_weight, scenario in weight_scenarios:
            threshold = self.optimize_threshold(self.xgb_model, fn_weight, fp_weight)
            y_pred_proba = self.xgb_model.predict_proba(self.X_test_scaled)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)

            cm = confusion_matrix(self.y_test, y_pred)
            fn = cm[1, 0] if cm.shape[0] > 1 else 0
            fp = cm[0, 1] if cm.shape[0] > 1 else 0
            cost = fn_weight * fn + fp_weight * fp

            sensitivity = recall_score(self.y_test, y_pred)
            specificity = recall_score(self.y_test, y_pred, pos_label=0)

            results.append({
                '权重比': f'{fn_weight}:{fp_weight}',
                '阈值': threshold,
                'Se(%)': sensitivity * 100,
                'Sp(%)': specificity * 100,
                '期望代价': cost,
                '推荐场景': scenario
            })

            print(f"权重 {fn_weight}:{fp_weight} - 阈值: {threshold:.3f}, "
                  f"Se: {sensitivity:.2%}, Sp: {specificity:.2%}, 代价: {cost}")

        # 保存结果为DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(OUT_DIR, 'weight_sensitivity_analysis.csv'), index=False)
        return results_df

    def qc_subgroup_analysis(self):
        """QC子群分析（高/低QC）"""
        if self.xgb_model is None or self.data is None:
            print("请先加载数据并训练XGBoost模型")
            return False

        # 划分高/低QC子群（假设GC异常标志为1表示异常）
        # 注意：这里使用原始数据划分，然后重新预处理
        train_size = int(len(self.data) * 0.7)

        # 高QC子群：GC异常标志为0
        high_qc_data = self.data[self.data['GC异常标志'] == 0]
        high_qc_train = high_qc_data.iloc[:int(len(high_qc_data) * 0.7)]
        high_qc_test = high_qc_data.iloc[int(len(high_qc_data) * 0.7):]

        # 低QC子群：GC异常标志为1
        low_qc_data = self.data[self.data['GC异常标志'] == 1]
        low_qc_train = low_qc_data.iloc[:int(len(low_qc_data) * 0.7)]
        low_qc_test = low_qc_data.iloc[int(len(low_qc_data) * 0.7):]

        print(f"\nQC子群分析 - 高QC样本数: {len(high_qc_data)}, 低QC样本数: {len(low_qc_data)}")

        results = []

        # 评估高QC子群
        for name, train_data, test_data in [
            ("高QC", high_qc_train, high_qc_test),
            ("低QC", low_qc_train, low_qc_test)
        ]:
            if len(train_data) == 0 or len(test_data) == 0:
                print(f"警告: {name}子群样本数不足，无法进行分析")
                continue

            X_train_sub = train_data[self.feature_names]
            y_train_sub = train_data['AE'].astype(int)
            X_test_sub = test_data[self.feature_names]
            y_test_sub = test_data['AE'].astype(int)

            # 标准化
            scaler_sub = StandardScaler()
            X_train_sub_scaled = scaler_sub.fit_transform(X_train_sub)
            X_test_sub_scaled = scaler_sub.transform(X_test_sub)

            # SMOTE过采样
            smote = SMOTE(random_state=SEED, k_neighbors=5)
            X_train_sub_resampled, y_train_sub_resampled = smote.fit_resample(
                X_train_sub_scaled, y_train_sub)

            # 训练模型
            model_sub = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=SEED
            )
            model_sub.fit(X_train_sub_resampled, y_train_sub_resampled)

            # 评估
            y_pred_proba_sub = model_sub.predict_proba(X_test_sub_scaled)[:, 1]
            y_pred_sub = (y_pred_proba_sub >= self.best_threshold).astype(int)

            auc = roc_auc_score(y_test_sub, y_pred_proba_sub)
            sensitivity = recall_score(y_test_sub, y_pred_sub)
            specificity = recall_score(y_test_sub, y_pred_sub, pos_label=0)

            # 计算不达标风险（预测错误的比例）
            error_rate = 1 - accuracy_score(y_test_sub, y_pred_sub)

            results.append({
                'QC子群': name,
                '样本数': len(test_data),
                'AUC': auc,
                'Se(%)': sensitivity * 100,
                'Sp(%)': specificity * 100,
                '不达标风险(%)': error_rate * 100
            })

            print(f"{name}子群 - AUC: {auc:.4f}, Se: {sensitivity:.2%}, "
                  f"Sp: {specificity:.2%}, 不达标风险: {error_rate:.2%}")

        # 保存结果为DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(OUT_DIR, 'qc_subgroup_analysis.csv'), index=False)
        return results_df

    def explain_model(self, model=None, sample_index=None):
        """使用SHAP和部分依赖图解释模型"""
        import warnings
        
        # 在函数内部临时禁用特定的NumPy RNG警告
        warnings.filterwarnings("ignore", 
                               message="The NumPy global RNG was seeded by calling `np.random.seed`", 
                               category=FutureWarning)
        
        if model is None:
            if self.xgb_model is not None:
                model = self.xgb_model
            else:
                print("请先训练模型")
                return False

        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)
        
        # 解决负号显示问题 - 更全面的方法
        # 为数学表示找到更适合的字体
        import matplotlib.font_manager as fm
        
        # 专门为SHAP瀑布图处理负号显示问题
        try:
            # 设置matplotlib rcParams，强制使用DejaVu Sans绘制数学符号
            plt.rcParams['mathtext.fontset'] = 'stix'  # 使用STIX字体集，通常支持数学符号
            plt.rcParams['mathtext.default'] = 'regular'
            # 确保负号显示正确
            plt.rcParams['axes.unicode_minus'] = False
            
            # 修改SHAP的_waterfall模块设置
            import shap.plots as shap_plots
            if hasattr(shap_plots, '_waterfall'):
                if hasattr(shap_plots._waterfall, 'rcParams'):
                    shap_plots._waterfall.rcParams['axes.unicode_minus'] = False
                    shap_plots._waterfall.rcParams['mathtext.fontset'] = 'stix'
        except Exception as e:
            print(f"设置SHAP负号字体时出错: {e}")
        
        # 绘制全局特征重要性 
        plt.figure(figsize=(14, 10)) 
        shap_values = explainer.shap_values(self.X_test_scaled)
        
        # 使用上下文管理器来临时设置随机种子
        import contextlib
        
        @contextlib.contextmanager
        def temp_seed(seed):
            """临时设置NumPy随机种子的上下文管理器"""
            state = np.random.get_state()
            np.random.seed(seed)
            try:
                yield
            finally:
                np.random.set_state(state)
        
        # 绘制SHAP图表
        with temp_seed(42):
            shap.summary_plot(shap_values, self.X_test_scaled, 
                             feature_names=self.feature_names,
                             plot_type="bar", show=False)
        
        plt.title("SHAP全局特征重要性", fontsize=14)  # 增加字体大小
        # 修改保存路径
        plt.savefig(os.path.join(VIS_DIR, 'shap_global_importance.png'), bbox_inches='tight', dpi=300)
        plt.close()

        # 绘制蜂群图 - 增加图形尺寸，特别是宽度
        plt.figure(figsize=(16, 12))  # 从(10, 6)改为(16, 12)
        
        with temp_seed(42):
            shap.summary_plot(shap_values, self.X_test_scaled, 
                             feature_names=self.feature_names, 
                             show=False)
        
        plt.title("SHAP特征影响蜂群图", fontsize=16, pad=20)  # 增加字体大小和顶部填充
        # 修改保存路径，添加bbox_inches参数确保完整显示
        plt.savefig(os.path.join(VIS_DIR, 'shap_beeswarm.png'), bbox_inches='tight', dpi=300)
        plt.close()

        # 绘制部分依赖图（针对最重要的特征）
        # 先确定最重要的特征
        feature_importance = np.abs(shap_values).mean(0)
        top_feature_idx = np.argmax(feature_importance)
        top_feature_name = self.feature_names[top_feature_idx]

        # 使用新版API绘制部分依赖图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            # 尝试使用新版API
            if 'PartialDependenceDisplay' in globals():
                # 如果已成功导入PartialDependenceDisplay
                pdp = partial_dependence(model, self.X_train_scaled, [top_feature_idx], kind='average')
                disp = PartialDependenceDisplay.from_estimator(
                    model, self.X_train_scaled, [top_feature_idx],
                    feature_names=self.feature_names, ax=ax
                )
            else:
                # 手动绘制部分依赖图
                pdp_result = partial_dependence(model, self.X_train_scaled, 
                                               features=[top_feature_idx])
                
                pdp_values = pdp_result["average"]
                pdp_feature_values = pdp_result["values"][0]
                
                ax.plot(pdp_feature_values, pdp_values[0])
                ax.set_xlabel(self.feature_names[top_feature_idx])
                ax.set_ylabel('Partial Dependence')
        except Exception as e:
            print(f"部分依赖图绘制失败: {e}")
            ax.text(0.5, 0.5, f"部分依赖图无法绘制: {e}", 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.title(f"{top_feature_name}的部分依赖图")
        # 修改保存路径
        plt.savefig(os.path.join(VIS_DIR, f'pdp_{top_feature_name}.png'))
        plt.close()

        # 如果指定了样本，绘制个体SHAP瀑布图
        if sample_index is not None and 0 <= sample_index < len(self.X_test_scaled):
            try:
                # 修复维度问题：确保输入是二维数组
                sample_data = self.X_test_scaled[sample_index].reshape(1, -1)
                # 计算SHAP值
                sample_shap_values = explainer.shap_values(sample_data)
                
                # 专门为瀑布图设置字体和样式
                plt.figure(figsize=(10, 6))
                
                # 使用更兼容的ASCII特征名称
                ascii_feature_names = []
                for name in self.feature_names:
                    try:
                        name.encode('ascii')  # 测试是否包含非ASCII字符
                        ascii_feature_names.append(name)
                    except UnicodeEncodeError:
                        # 使用更直观的编号命名
                        ascii_feature_names.append(f"Feature_{len(ascii_feature_names)+1}")
                
                # 创建自定义瀑布图而不是使用SHAP的内置函数
                # 这样我们可以完全控制字体和标签
                try:
                    # 计算基准值和特征贡献
                    base_value = explainer.expected_value
                    feature_values = sample_shap_values[0]
                    
                    # 按绝对贡献排序
                    indices = np.argsort(np.abs(feature_values))[::-1]
                    sorted_names = [ascii_feature_names[i] for i in indices]
                    sorted_values = feature_values[indices]
                    
                    # 创建瀑布图数据
                    cum_values = np.zeros(len(sorted_values) + 1)
                    cum_values[0] = base_value
                    for i in range(len(sorted_values)):
                        cum_values[i+1] = cum_values[i] + sorted_values[i]
                    
                    # 绘制瀑布图
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # 绘制基准线和最终预测线
                    ax.axhline(y=base_value, color='gray', linestyle='-', alpha=0.3)
                    ax.axhline(y=base_value + np.sum(feature_values), color='blue', linestyle='-', alpha=0.3)
                    
                    # 绘制特征贡献
                    for i in range(len(sorted_values)):
                        # 判断是增加还是减少
                        if sorted_values[i] >= 0:
                            color = 'red'
                            label = "+" + f"{sorted_values[i]:.3f}"
                        else:
                            color = 'blue'
                            label = f"{sorted_values[i]:.3f}"  # 负号不需要额外添加
                        
                        # 绘制柱子
                        ax.bar(i, sorted_values[i], bottom=cum_values[i], width=0.6,
                              color=color, alpha=0.7)
                        
                        # 添加数值标签，使用普通减号字符而不是Unicode负号
                        if sorted_values[i] < 0:
                            label = label.replace("−", "-").replace("–", "-")
                        y_pos = cum_values[i] + sorted_values[i]/2
                        ax.text(i, y_pos, label, ha='center', va='center')
                    
                    # 设置X轴刻度标签
                    ax.set_xticks(range(len(sorted_names)))
                    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
                    
                    # 添加标题和标签
                    ax.set_title(f"SHAP瀑布图")
                    ax.set_ylabel("SHAP值")
                    ax.grid(True, axis='y', alpha=0.3)
                    
                    # 如果有中文特征名称，添加映射图例
                    if any(n != f for n, f in zip(self.feature_names, ascii_feature_names)):
                        legend_text = "\n".join(
                            [f"{ascii_feature_names[i]}: {self.feature_names[i]}" 
                             for i in range(len(self.feature_names))
                             if ascii_feature_names[i] != self.feature_names[i]])
                        
                        plt.figtext(1.02, 0.5, legend_text, fontsize=9,
                                   bbox=dict(facecolor='white', alpha=0.8),
                                   transform=ax.transAxes)
                    
                    plt.tight_layout()
                    # 修改保存路径
                    plt.savefig(os.path.join(VIS_DIR, f'shap_waterfall_sample_{sample_index}.png'))
                    plt.close()
                except Exception as e:
                    print(f"自定义瀑布图绘制失败: {e}")
                    
                    # 尝试SHAP库的瀑布图，但使用ASCII特征名称
                    try:
                        shap_explanation = shap.Explanation(
                            values=sample_shap_values[0], 
                            base_values=explainer.expected_value, 
                            data=sample_data[0], 
                            feature_names=ascii_feature_names
                        )
                        
                        # 强制使用支持负号的字体
                        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] + plt.rcParams['font.sans-serif']
                        
                        shap.plots.waterfall(shap_explanation, show=False)
                        
                        # 添加特征映射图例
                        legend_text = []
                        for i, (ascii_name, orig_name) in enumerate(zip(ascii_feature_names, self.feature_names)):
                            if ascii_name != orig_name:
                                legend_text.append(f"{ascii_name}: {orig_name}")
                        
                        if legend_text:
                            plt.figtext(1.05, 0.5, "\n".join(legend_text), fontsize=9,
                                      bbox=dict(facecolor='white', alpha=0.8))
                        
                        plt.tight_layout()
                        # 修改保存路径
                        plt.savefig(os.path.join(VIS_DIR, f'shap_waterfall_sample_{sample_index}.png'))
                        plt.close()
                    except Exception as e:
                        print(f"SHAP瀑布图尝试失败: {e}")
                        # 回退到简化的条形图
                        plt.figure(figsize=(10, 6))
                        plt.bar(range(len(self.feature_names)), 
                              np.abs(sample_shap_values[0]), 
                              tick_label=ascii_feature_names,
                              color='skyblue')
                        plt.xticks(rotation=45)
                        plt.title(f"样本 {sample_index} 的特征重要性")
                        plt.tight_layout()
                        # 修改保存路径
                        plt.savefig(os.path.join(VIS_DIR, f'shap_waterfall_sample_{sample_index}.png'))
                        plt.close()
            except Exception as e:
                print(f"SHAP瀑布图绘制完全失败: {e}")
                # 回退到最简单的图表
                plt.figure(figsize=(10, 6))
                plt.bar(range(len(self.feature_names)), 
                      np.abs(shap_values[sample_index]), 
                      color='skyblue')
                plt.xticks(range(len(self.feature_names)), 
                         range(1, len(self.feature_names)+1))
                plt.title(f"样本 {sample_index} 的特征重要性")
                plt.tight_layout()
                # 修改保存路径
                plt.savefig(os.path.join(VIS_DIR, f'shap_sample_{sample_index}_importance.png'))
                plt.close()

    def save_model(self, model_name='xgb_model', path=None):
        """保存模型"""
        if path is None:
            path = os.path.join(OUT_DIR, 'models/')
        
        if not os.path.exists(path):
            os.makedirs(path)

        if model_name == 'xgb_model' and self.xgb_model is not None:
            pickle.dump(self.xgb_model, open(f"{path}xgb_model.pkl", "wb"))
            pickle.dump(self.scaler, open(f"{path}scaler.pkl", "wb"))
            print("XGBoost模型和标准化器已保存")
            return True
        elif model_name == 'lgb_model' and self.lgb_model is not None:
            pickle.dump(self.lgb_model, open(f"{path}lgb_model.pkl", "wb"))
            print("LightGBM模型已保存")
            return True
        elif model_name == 'ensemble_model' and self.ensemble_model is not None:
            pickle.dump(self.ensemble_model, open(f"{path}ensemble_model.pkl", "wb"))
            print("集成模型已保存")
            return True
        else:
            print("模型不存在或未训练")
            return False

    def load_model(self, model_name='xgb_model', path=None):
        """加载模型"""
        if path is None:
            path = os.path.join(OUT_DIR, 'models/')
        
        try:
            if model_name == 'xgb_model':
                self.xgb_model = pickle.load(open(f"{path}xgb_model.pkl", "rb"))
                self.scaler = pickle.load(open(f"{path}scaler.pkl", "rb"))
                print("XGBoost模型和标准化器已加载")
                return True
            elif model_name == 'lgb_model':
                self.lgb_model = pickle.load(open(f"{path}lgb_model.pkl", "rb"))
                print("LightGBM模型已加载")
                return True
            elif model_name == 'ensemble_model':
                self.ensemble_model = pickle.load(open(f"{path}ensemble_model.pkl", "rb"))
                print("集成模型已加载")
                return True
            else:
                print("模型名称无效")
                return False
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            return False

    def predict(self, features, model=None):
        """预测新样本"""
        if model is None:
            if self.xgb_model is not None:
                model = self.xgb_model
            else:
                print("请先训练或加载模型")
                return None

        # 确保输入是DataFrame并包含所有必要的特征
        if isinstance(features, dict):
            features = pd.DataFrame([features])

        missing_cols = [col for col in self.feature_names if col not in features.columns]
        if missing_cols:
            print(f"输入缺少必要的特征: {missing_cols}")
            return None

        # 选择需要的特征并标准化
        X = features[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # 创建具有正确特征名称的DataFrame (解决特征名称警告)
        if hasattr(model, '_Booster') and isinstance(model, lgb.LGBMClassifier):
            # 对于LightGBM模型，确保预测输入保留特征名称
            X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
            y_proba = model.predict_proba(X_scaled_df)
        else:
            # 对于其他模型，使用标准方式预测
            y_proba = model.predict_proba(X_scaled)

        # 预测概率和类别
        prob = y_proba[:, 1][0]
        pred = 1 if prob >= self.best_threshold else 0

        return {
            '异常概率': prob,
            '预测结果': '异常' if pred == 1 else '正常',
            '推荐行动': '穿刺' if pred == 1 else '观察'
        }


# 主函数，演示模型的使用流程
def main():
    # 确保在程序开始时设置中文字体
    set_chinese_font()
    
    # 初始化模型
    model = FemaleFetalModel()

    # 尝试加载女胎数据
    female_data_path = 'processed_female.csv'
    male_data_path = 'processed_male.csv'
    
    if os.path.exists(female_data_path):
        print(f"加载女胎数据: {female_data_path}")
        success = model.load_data(female_data_path)
        if not success:
            print("加载女胎数据失败，尝试男胎数据")
            if os.path.exists(male_data_path):
                print(f"加载男胎数据: {male_data_path}")
                success = model.load_data(male_data_path)
    elif os.path.exists(male_data_path):
        print(f"未找到女胎数据，加载男胎数据: {male_data_path}")
        success = model.load_data(male_data_path)
    else:
        print("未找到实际数据文件")
        success = False

    # 如果无法加载实际数据，则使用模拟数据
    if not success:
        print("使用模拟数据进行演示")
        model = generate_demo_data(model)

    # 预处理数据
    model.preprocess_data(test_size=0.3, use_smote=True)

    # 训练XGBoost模型
    model.train_xgboost(perform_grid_search=False)

    # 优化阈值
    model.optimize_threshold()

    # 训练LightGBM模型（可选）
    model.train_lightgbm(perform_grid_search=False)

    # 训练集成模型
    model.train_ensemble()

    # 阈值敏感性分析
    model.threshold_sensitivity_analysis()

    # 代价权重敏感性分析
    model.weight_sensitivity_analysis()

    # QC子群分析
    model.qc_subgroup_analysis()

    # 模型解释
    model.explain_model(sample_index=0)

    # 保存模型
    model.save_model()

    # 预测示例
    sample = {
        '21-Z': 3.5,
        '18-Z': 1.2,
        '13-Z': 0.8,
        'GC异常标志': 0,
        '读段数': 1500000,
        'BMI': 25.0,
        '孕周': 16
    }
    result = model.predict(sample)
    print("\n预测结果:")
    print(f"异常概率: {result['异常概率']:.4f}")
    print(f"预测结果: {result['预测结果']}")
    print(f"推荐行动: {result['推荐行动']}")


def generate_demo_data(model):
    """生成演示用的模拟数据"""
    # 生成样本量
    n_samples = 5000

    # 生成特征
    np.random.seed(SEED)
    data = {
        '21-Z': np.random.normal(1.0, 0.8, n_samples),
        '18-Z': np.random.normal(0.8, 0.6, n_samples),
        '13-Z': np.random.normal(0.7, 0.5, n_samples),
        'GC异常标志': np.random.binomial(1, 0.3, n_samples),  # 30%的样本有GC异常
        '读段数': np.random.randint(1000000, 2000000, n_samples),
        'BMI': np.random.normal(24, 4, n_samples),
        '孕周': np.random.uniform(12, 25, n_samples)
    }

    # 生成目标变量（异常概率与21-Z强相关）
    # 阳性率约为2.8%
    risk_score = 0.3 * data['21-Z'] + 0.2 * data['18-Z'] + 0.1 * data['13-Z'] + 0.2 * data['GC异常标志']
    prob = 1 / (1 + np.exp(-(risk_score - 2.5)))  # 逻辑函数转换为概率
    data['AE'] = np.random.binomial(1, prob)

    # 创建DataFrame
    model.data = pd.DataFrame(data)
    model.X = model.data[model.feature_names]
    model.y = model.data['AE'].astype(int)

    print(f"生成模拟数据，共 {model.data.shape[0]} 样本，阳性比例: {model.y.mean():.2%}")
    return model


if __name__ == "__main__":
    main()
