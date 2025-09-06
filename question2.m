% question2.m
% MATLAB 主控脚本：BMI分组与最佳NIPT时点（区间删失 + 风险最优 + DP分箱 + Bootstrap/MC）
% 需要 R 支持 icenReg::ic_par 和 survreg（对数正态AFT）
%
% 使用步骤：
% 1) 准备 data.csv（或 data.xlsx），包含列： patient_id, gestational_weeks, Y_frac, BMI
% 2) 保证系统已安装 R，并安装 R 包： icenReg, survival, flexsurv, mclust
%     在 R 中运行: install.packages(c("icenReg","survival","flexsurv","mclust"))
% 3) 把本文件、icenreg.R 放在同一文件夹，启动 MATLAB，运行 Q2_matlab.m
%
% 输出文件夹 ./output_q2 将包含：
% intervals.csv, mid/low/up csv, group_table.csv, risk_curves/*.png,
% bootstrap_results.csv, mc_results.csv, icenreg_out.csv（如果R可用）

clear; close all; clc;

%% ---------- 配置 ----------
DATA_PATH = "processed_male.csv";        % <-- 修改为你的文件路径 (.csv 或 .xlsx)
USE_XLSX = false;              % 如果用 .xlsx 且路径为 .xlsx，则设为 true
THRESH = 0.04;                 % 4% 达标阈值（0-1）
TGRID = (10:0.1:25)';          % 网格搜索（列向量）
W = [1,1,2];                   % 风险函数权重
BOOT_B = 20;                  % Bootstrap 次数（竞赛可用 200-500）
MC_B = 10;                   % Monte Carlo 次数（视时间调整）
K_BINS = 4;                    % DP 分箱数候选（后续可用 AIC/BIC 选 k）
MIN_PER_BIN = 20;              % 每组最小样本数

OUT_DIR = "./output_q2";
if ~exist(OUT_DIR, 'dir'); mkdir(OUT_DIR); end

R_BIN_DIR = 'E:\R-4.5.1\bin'; % Rscript.exe 所在目录
rscriptExe = fullfile(R_BIN_DIR, 'Rscript.exe');

T = readtable(DATA_PATH);

%% 映射中文列名到标准列名
varMap = containers.Map( ...
    {'孕妇代码','检测孕周','Y染色体浓度','孕妇BMI'}, ...
    {'patient_id','gestational_weeks','Y_frac','BMI'} ...
);

for i = 1:numel(T.Properties.VariableNames)
    v = T.Properties.VariableNames{i};
    if isKey(varMap, v)
        T.Properties.VariableNames{i} = varMap(v);
    end
end

%% ---------- 读取数据 ----------
fprintf("Reading data from %s\n", DATA_PATH);
if USE_XLSX
    T = readtable(DATA_PATH, 'FileType','spreadsheet', 'VariableNamingRule','preserve');
else
    T = readtable(DATA_PATH, 'VariableNamingRule','preserve');
end
% 尝试把常见的中文列名映射为标准列名（patient_id, gestational_weeks, Y_frac, BMI 等）
colMap = containers.Map( ...
    {'孕妇代码','检测孕周','Y染色体浓度','孕妇BMI','检测抽血次数','13号染色体的GC含量','18号染色体的GC含量','21号染色体的GC含量'}, ...
    {'patient_id','gestational_weeks','Y_frac','BMI','blood_draw_count','GC13','GC18','GC21'} ...
);

% 如果 MATLAB 自动修改了变量名，原始表头可能保存在 VariableDescriptions 中
for i = 1:numel(T.Properties.VariableNames)
    vn = T.Properties.VariableNames{i};
    % 直接匹配当前变量名
    if isKey(colMap, vn)
        T.Properties.VariableNames{i} = colMap(vn);
        continue;
    end

    % 安全地尝试读取 VariableDescriptions 中的原始标题（防止索引越界）
    origHeader = '';
    if isprop(T.Properties, 'VariableDescriptions') ...
            && ~isempty(T.Properties.VariableDescriptions) ...
            && numel(T.Properties.VariableDescriptions) >= i
        try
            tmp = T.Properties.VariableDescriptions{i};
            if ~isempty(tmp)
                if isstring(tmp) || ischar(tmp)
                    origHeader = char(tmp);
                elseif iscell(tmp) && ~isempty(tmp)
                    origHeader = char(tmp{1});
                end
            end
        catch
            origHeader = '';
        end
    end

    if ~isempty(origHeader) && isKey(colMap, origHeader)
        T.Properties.VariableNames{i} = colMap(origHeader);
        continue;
    end
end

% 检查所需列
req = {'patient_id','gestational_weeks','Y_frac','BMI'};
for i=1:length(req)
    if ~ismember(req{i}, T.Properties.VariableNames)
        error('缺少必要列: %s', req{i});
    end
end

% 统一 Y_frac 范围（0-1）
if max(T.Y_frac) > 1.05
    T.Y_frac = T.Y_frac / 100;
end

%% ---------- 4.1 构建区间删失 (L,R] ----------
fprintf("正在构建患者级区间 (L,R] ...\n");
uids = unique(T.patient_id);
nP = length(uids);
rows = table('Size',[nP,5], 'VariableTypes',{'cell','double','double','double','double'}, ...
    'VariableNames',{'patient_id','GA_lower','GA_upper','event','BMI'});
for i=1:nP
    pid = uids{i};
    sub = T(strcmp(T.patient_id, pid), :);
    sub = sortrows(sub, 'gestational_weeks');
    ga = sub.gestational_weeks;
    y = sub.Y_frac;
    bmi = median(sub.BMI,'omitnan');
    if isnan(bmi)
        continue; % 跳过 BMI 全为 NaN 的患者
    end
    idx = find(y >= THRESH, 1, 'first');
    if ~isempty(idx)
        k = idx;
        if k > 1
            L = ga(k-1);
        else
            L = 0;
        end
        R = ga(k);
        ev = 1;
    else
        L = max(ga);
        R = Inf;
        ev = 0;
    end
    rows.patient_id{i} = pid;
    rows.GA_lower(i) = L;
    rows.GA_upper(i) = R;
    rows.event(i) = ev;
    rows.BMI(i) = bmi;
end
% 替换区间下界和上界为0的为极小正数
rows.GA_lower(rows.GA_lower==0) = 1e-6;
rows.GA_upper(rows.GA_upper==0) = 1e-6;
% 过滤掉 BMI 为 NaN 的行
rows = rows(~isnan(rows.BMI), :);

writetable(rows, fullfile(OUT_DIR,'intervals.csv'));

%% ---------- 4.1.2 三重敏感性 mid/low/up 数据集 ----------
fprintf("正在生成 mid/low/up 数据集 ...\n");
mid = table(); low = table(); up = table();
for i=1:height(rows)
    pid = rows.patient_id{i};
    L = rows.GA_lower(i);
    R = rows.GA_upper(i);
    ev = rows.event(i);
    bmi = rows.BMI(i);
    if isfinite(R)
        Tmid = 0.5*(L + R); evm = 1;
        Tup = R; evu = 1;
    else
        Tmid = L; evm = 0;
        Tup = L; evu = 0;
    end
    % 添加行
    mid = [mid; table({pid}, Tmid, evm, bmi, 'VariableNames', {'patient_id','T','event','BMI'})];
    low = [low; table({pid}, L, ev, bmi, 'VariableNames', {'patient_id','T','event','BMI'})];
    up = [up; table({pid}, Tup, evu, bmi, 'VariableNames', {'patient_id','T','event','BMI'})];
end

mid = mid(mid.T > 0 & ~isnan(mid.T) & ~isnan(mid.BMI), :);
low = low(low.T > 0 & ~isnan(low.T) & ~isnan(low.BMI), :);
up  = up(up.T  > 0 & ~isnan(up.T) & ~isnan(up.BMI),  :);

writetable(mid, fullfile(OUT_DIR,'GA_midpoint.csv'));
writetable(low, fullfile(OUT_DIR,'GA_low.csv'));
writetable(up, fullfile(OUT_DIR,'GA_up.csv'));

%% ---------- 4.1.2 三重敏感性分析，调用 R (survreg) ----------
% 准备传递给 R 的输入：写出 mid/low/up CSV，列为 patient_id,T,event,BMI
fprintf("调用 R 拟合 mid/low/up 的对数正态AFT模型 (survreg) ...\n");
rInputMid = fullfile(OUT_DIR,'GA_midpoint.csv');
rInputLow = fullfile(OUT_DIR,'GA_low.csv');
rInputUp  = fullfile(OUT_DIR,'GA_up.csv');
% R 脚本路径
rScript = 'icenreg.R';
% 调用 R，拟合三组数据及区间删失AFT
% 此 R 脚本将输出：aft_mid_coef.csv, aft_low_coef.csv, aft_up_coef.csv, icenreg_out.csv
cmd = sprintf('"%s" "%s" "%s" "%s" "%s" "%s"', rscriptExe, rScript, fullfile(OUT_DIR,'intervals.csv'), rInputMid, rInputLow, rInputUp);
[status, cmdout] = system(cmd);
if status ~= 0
    warning('Rscript 调用失败或 R 未安装/配置。R 输出不可用。错误信息:\n%s', cmdout);
else
    fprintf('Rscript 执行完毕，正在读取 R 输出...\n');
end

% 若 R 产生了系数文件则尝试读取
aft_mid_coef = []; aft_low_coef = []; aft_up_coef = []; icenreg_out = [];
try
    aft_mid_coef = readtable(fullfile(OUT_DIR,'aft_mid_coef.csv'));
    aft_low_coef = readtable(fullfile(OUT_DIR,'aft_low_coef.csv'));
    aft_up_coef = readtable(fullfile(OUT_DIR,'aft_up_coef.csv'));
    icenreg_out = readtable(fullfile(OUT_DIR,'icenreg_out.csv')); % R 预测的不同 BMI 分位数的 tgrid 生存率
    disp('已加载 R 输出。');
catch
    warning('无法加载部分 R 输出文件，将使用可用数据继续。');
end

%% ---------- 4.2 风险函数定义  ----------
fprintf("准备 IC-AFT (R 输出) 的生存预测函数 ...\n");
% 策略：
% - 如果 icenreg_out 可用：其中包含多个 BMI 水平的 tgrid 生存率 S(t)
% - 构建插值函数 S(bmi,t)，在 BMI 和 t 方向做线性插值
% 如果 icenreg_out 不可用，则回退为每组的 Kaplan-Meier

if ~isempty(icenreg_out)
    % 期望 icenreg_out: 列为 week, S_BMIq1, S_BMIq2, S_BMIq3（或类似）
    % 获取 tgrid 和 bmi 分位数标签（R 输出文件头部）
    t_R = icenreg_out.week;
    bmi_cols = icenreg_out.Properties.VariableNames(2:end);
    % 解析 bmi 分位数标签在表头；需要 BMI 水平；R 脚本用分位数
    % 若有补充文件则读取；否则默认索引映射
    % 构建 S_pred(bmi) 函数，在各列生存曲线间插值
    S_matrix = double(table2array(icenreg_out(:,2:end))); % 明确转换为 double 类型
    bmi_levels_file = fullfile(OUT_DIR,'icenreg_bmi_levels.csv');
    if exist(bmi_levels_file,'file')
        bmi_levels = double(csvread(bmi_levels_file)); % 确保为 double 类型
    else
        % 回退：默认用 3 个分位数：25/50/75（R 脚本默认）
        % 若 bmi 分位数文件不存在，则用 intervals.BMI 估算
        bmi_levels = double(quantile(rows.BMI, [0.25,0.5,0.75])); % 确保为 double 类型
    end
    % 用 meshgrid 构建插值网格
    t_R = double(t_R); % 确保为 double 类型
    [Bgrid, TgridR] = meshgrid(bmi_levels, t_R);
    
    % 创建安全的插值函数
    interp_with_fallback = @(bmi, t_query) safe_interp(Bgrid, TgridR, S_matrix, double(bmi), double(t_query));
    
    % 存储以备后用
    icen.Sfun = interp_with_fallback;
else
    % 回退：后续每组用 KM
    icen.Sfun = [];
    warning('IC-AFT 生存曲面不可用，将使用每组 KM 估计。');
end

% 安全的插值函数，处理各种边缘情况
function result = safe_interp(X, Y, V, xi, yi)
    % 确保所有输入都是数值型
    if ~isnumeric(xi) || ~isnumeric(yi) || ~isnumeric(X) || ~isnumeric(Y) || ~isnumeric(V)
        result = 1; % 如果任何参数不是数值型，返回安全值
        return;
    end
    
    % 确保标量输入
    if ~isscalar(xi) || ~isscalar(yi)
        result = 1;
        return;
    end
    
    % 尝试插值
    try
        result = interp2(X, Y, V, xi, yi, 'linear', NaN);
        % 处理 NaN 和边界
        if isnan(result)
            result = 1; % 保守估计
        else
            result = max(0, min(1, result)); % 确保结果在 [0,1] 范围内
        end
    catch
        % 任何插值错误，返回保守估计
        result = 1;
    end
end

% 风险函数句柄：输入 Sfun(bmi,t) 返回生存率 S(t)
risk_fun = @(t, Sval) (W(1)*(1 - Sval) + W(2)*max(t-12,0) + W(3)*max(t-20,0));
% 注：R 中风险函数用 P(T>t) = 1 - S(t)，此处同样用 1 - S

%% ---------- 4.3 聚类辅助 & DP 分箱 ----------
fprintf("对 (BMI, Tmid) 做 GMM 聚类以获得初步分组 ...\n");
% 对 sets.mid 的 {BMI, T} 做 GMM
BMI_T = [mid.BMI, mid.T];
% 用 BIC 类似方法选分量数
maxComp = 5;
bestBIC = inf; bestGM = [];
for k = 1:min(maxComp, size(BMI_T,1)-1)
    try
        gm = fitgmdist(BMI_T, k, 'RegularizationValue',1e-6, 'Options', statset('MaxIter',500));
        bic = gm.BIC;
        if bic < bestBIC
            bestBIC = bic; bestGM = gm; bestK = k;
        end
    catch
        % 跳过
    end
end
if isempty(bestGM)
    warning('GMM 聚类失败，回退为 K=4 的 kmeans。');
    bestK = min(4, size(BMI_T,1));
    idx = kmeans(BMI_T, bestK, 'Replicates',5);
else
    idx = cluster(bestGM, BMI_T);
    bestK = bestGM.NumComponents;
end
mid.group = idx;
% 将 mid.group 映射到患者级 intervals（一个患者可能多次映射，取众数）
% 通过患者在 mid 中的 group 标签的中位数确定分组
pidList = rows.patient_id;
groupAssign = nan(height(rows),1);
for i=1:height(rows)
    pid = rows.patient_id{i};
    mask = strcmp(mid.patient_id, pid);
    if any(mask)
        groupAssign(i) = round(median(mid.group(mask)));
    else
        groupAssign(i) = 1;
    end
end
rows.group = groupAssign;
writetable(rows, fullfile(OUT_DIR,'GA_intervals_with_groups.csv'));

% ---------- DP 分箱（按 BMI 排序，最小化组内 Tmid 方差） ----------
fprintf("正在对 BMI 做 DP 最优一维分箱 (k=%d) ...\n", K_BINS);
% 准备数组（用 mid.T 作为目标）
valid = ~isnan(mid.BMI) & ~isnan(mid.T);
bmi_arr = mid.BMI(valid);
t_arr = mid.T(valid);
% 按 bmi 排序
[sortBMI, sidx] = sort(bmi_arr);
sortT = t_arr(sidx);
n = numel(sortBMI);

% 计算前缀和用于组内平方和
S1 = [0; cumsum(sortT)];
S2 = [0; cumsum(sortT.^2)];
withinSS = @(i,j) ( (S2(j+1)-S2(i)) - ((S1(j+1)-S1(i)).^2)./(j-i+1) );

INF = 1e18;
dp = INF * ones(K_BINS+1, n+1);
prev = -ones(K_BINS+1, n+1);
dp(1,2:n+1) = arrayfun(@(j) withinSS(1,j), 1:n); % 单组代价
% 转移时强制每组最小样本数
for k=2:K_BINS
    for j=k:n
        best = INF; bi = -1;
        for i=k-1:j-1
            if (j - i) < MIN_PER_BIN
                continue;
            end
            val = dp(k-1, i) + withinSS(i+1, j);
            if val < best
                best = val; bi = i;
            end
        end
        dp(k,j+1) = best;
        prev(k,j+1) = bi;
    end
end
% 回溯得到分割点
cuts = zeros(K_BINS,2);
j = n;
dp_failed = false;
for k=K_BINS:-1:1
    i = prev(k,j+1);
    if i < 0
        warning('DP 回溯失败，使用分位数分箱');
        dp_failed = true;
        break;
    end
    cuts(k,:) = [i+1, j];
    j = i;
end

binEdges = zeros(K_BINS,1);
labels_sorted = zeros(n,1);

if dp_failed
    % 用分位数分箱
    thresholds = quantile(sortBMI, linspace(0,1,K_BINS+1));
    thresholds = thresholds(2:end); % 去掉最小值
    for idx = 1:n
        assigned = find(sortBMI(idx) <= thresholds,1);
        if isempty(assigned)
            assigned = K_BINS;
        end
        labels_sorted(idx) = assigned;
    end
    binEdges = thresholds;
else
    % DP分箱正常
    for k=1:K_BINS
        i = cuts(k,1); j = cuts(k,2);
        labels_sorted(i:j) = k;
        binEdges(k) = sortBMI(j);
    end
end
% 映射标签回 mid 原顺序
labels_full = nan(size(sortBMI));
labels_full(sidx) = labels_sorted;
% 按 BMI 和分箱阈值给患者分组
% 阈值四舍五入到 0.5
thresholds = round(binEdges*2)/2;
fprintf('DP 分箱阈值（四舍五入到0.5）：'); disp(thresholds');

% 按 BMI 和阈值给患者分组
rows.BMI_group = zeros(height(rows),1);
for i=1:height(rows)
    b = rows.BMI(i);
    if isnan(b)
        rows.BMI_group(i) = -1;
        continue;
    end
    assigned = find(b <= thresholds,1);
    if isempty(assigned)
        assigned = K_BINS;
    end
    rows.BMI_group(i) = assigned;
end
writetable(rows, fullfile(OUT_DIR,'GA_intervals_DPgroups.csv'));

%% ---------- 4.4 每组最优时点搜索 ----------
fprintf("正在为每个 BMI 组搜索最优 t*（网格搜索，约束 P(T<=t)>=0.9）...\n");
groups = unique(rows.BMI_group);
group_summary = table('Size',[numel(groups),5], 'VariableTypes',{'double','double','double','double','double'}, ...
    'VariableNames',{'BMI_group','BMI_center','t_star','reach_prob','min_risk'});
rowIdx = 1;
for g = groups'
    if g < 0; continue; end
    mask = rows.BMI_group == g;
    if sum(mask) < 10
        fprintf('第 %d 组样本太少，跳过\n', g); continue;
    end
    bmi_center = double(mean(rows.BMI(mask),'omitnan')); % 确保为 double 类型
    % 生存函数 S(t)：优先用 icen.Sfun，否则用该组 mid 的 KM
    if ~isempty(icen) && ~isempty(icen.Sfun)
        % 包装一个安全的函数调用
        Sfun = @(t) icen.Sfun(bmi_center, double(t));
        try
            reach_vec = 1 - arrayfun(Sfun, TGRID);
            risk_vec = arrayfun(@(t, r) risk_fun(t, r), TGRID, arrayfun(Sfun, TGRID));
        catch ME
            fprintf('插值计算出错，回退到KM估计: %s\n', ME.message);
            % 回退到KM（下面的else分支代码）
            patlist = rows.patient_id(mask);
            patlist = rows.patient_id(mask);

            % 保证都是 cellstr 类型（只保留字符型，过滤 NaN/空）
            mid_pid = mid.patient_id;
            if ~iscellstr(mid_pid)
                mid_pid = cellfun(@(x) char(x), mid_pid, 'UniformOutput', false);
            end
            patlist_c = patlist;
            if ~iscellstr(patlist_c)
                patlist_c = cellfun(@(x) char(x), patlist_c, 'UniformOutput', false);
            end

            mask_mid = ismember(mid_pid, patlist_c);
            sub_mid = mid(mask_mid,:);
            % 用 MATLAB 的 ecdf 做 KM？用 Statistics Toolbox 的 ecdf 带删失
            try
                % 如果有 Statistics Toolbox：
                [f,x] = ecdf(sub_mid.T, 'censoring', 1 - sub_mid.event);
                % ecdf 返回 F（cdf），生存率 S = 1 - F
                % 在 TGRID 上插值 S
                F_interp = interp1(x, f, TGRID, 'previous', 'extrap');
                Svec = 1 - F_interp;
            catch
                % 回退：经验近似（不删失）
                Svec = arrayfun(@(t) mean(sub_mid.T > t), TGRID);
            end
            reach_vec = 1 - Svec;
            risk_vec = arrayfun(@(i) risk_fun(TGRID(i), Svec(i)), 1:length(TGRID));
        end
    else
        % 用该组 mid 数据做 KM 拟合
        patlist = rows.patient_id(mask);
        patlist = rows.patient_id(mask);

        % 保证都是 cellstr 类型（只保留字符型，过滤 NaN/空）
        mid_pid = mid.patient_id;
        if ~iscellstr(mid_pid)
            mid_pid = cellfun(@(x) char(x), mid_pid, 'UniformOutput', false);
        end
        patlist_c = patlist;
        if ~iscellstr(patlist_c)
            patlist_c = cellfun(@(x) char(x), patlist_c, 'UniformOutput', false);
        end

        mask_mid = ismember(mid_pid, patlist_c);
        sub_mid = mid(mask_mid,:);
        % 用 MATLAB 的 ecdf 做 KM？用 Statistics Toolbox 的 ecdf 带删失
        try
            % 如果有 Statistics Toolbox：
            [f,x] = ecdf(sub_mid.T, 'censoring', 1 - sub_mid.event);
            % ecdf 返回 F（cdf），生存率 S = 1 - F
            % 在 TGRID 上插值 S
            F_interp = interp1(x, f, TGRID, 'previous', 'extrap');
            Svec = 1 - F_interp;
        catch
            % 回退：经验近似（不删失）
            Svec = arrayfun(@(t) mean(sub_mid.T > t), TGRID);
        end
        reach_vec = 1 - Svec;
        risk_vec = arrayfun(@(i) risk_fun(TGRID(i), Svec(i)), 1:length(TGRID));
    end
    % 强制 reach >= 0.9
    valid_idx = find(reach_vec >= 0.9);
    if isempty(valid_idx)
        fprintf('第 %d 组：网格内无 t 达到 reach >=0.9，放宽约束（选最小风险）\n', g);
        [minR, minI] = min(risk_vec);
        t_star = TGRID(minI);
        reach_at = reach_vec(minI);
    else
        [minR, minI_rel] = min(risk_vec(valid_idx));
        minI = valid_idx(minI_rel);
        t_star = TGRID(minI);
        reach_at = reach_vec(minI);
    end
    group_summary.BMI_group(rowIdx) = g;
    group_summary.BMI_center(rowIdx) = bmi_center;
    group_summary.t_star(rowIdx) = t_star;
    group_summary.reach_prob(rowIdx) = reach_at;
    group_summary.min_risk(rowIdx) = minR;
    rowIdx = rowIdx + 1;
end

writetable(group_summary, fullfile(OUT_DIR,'group_table.csv'));

%% ---------- 4.4 Bootstrap 抽样 ----------
fprintf("Bootstrap 抽样（B=%d）以估计 t* 的置信区间 ...\n", BOOT_B);
% 在 parfor 中写入必须是按循环切片的可切片变量，因此用矩阵存放每次迭代结果
nGroups = height(group_summary);
boot_tstars_mat = NaN(BOOT_B, nGroups);

% 创建临时目录以避免文件冲突
temp_dir = fullfile(OUT_DIR, 'temp_bootstrap');
if ~exist(temp_dir, 'dir'); mkdir(temp_dir); end

% 使用更加健壮的并行处理方法 - 拆分为较小的批次
batch_size = min(5, ceil(BOOT_B/4)); % 每批最多5个样本
num_batches = ceil(BOOT_B/batch_size);

% 创建等待进度条
progress_step = ceil(num_batches/10);
fprintf('开始 Bootstrap 模拟，共 %d 批次:\n', num_batches);
progress = '';

for batch = 1:num_batches
    % 显示进度
    if mod(batch, progress_step) == 0 || batch == num_batches
        progress = [progress, '='];
        fprintf('[%s] %d/%d\n', progress, batch, num_batches);
    end
    
    start_idx = (batch-1)*batch_size + 1;
    end_idx = min(batch*batch_size, BOOT_B);
    batch_ids = start_idx:end_idx;
    
    % 使用 parfor 处理当前批次
    parfor b = batch_ids
        try
            % 设置超时（60秒）
            timeout_cmd = '180'; % 最长允许3分钟
            
            % 在唯一临时目录中创建文件
            boot_dir = fullfile(temp_dir, sprintf('boot_%d', b));
            if ~exist(boot_dir, 'dir'); mkdir(boot_dir); end
            
            % 创建Bootstrap样本
            ridx = randi(height(rows), height(rows), 1);
            samp = rows(ridx, :);
            samp = samp(~isnan(samp.BMI), :);
            sampFile = fullfile(boot_dir, sprintf('boot_sample_%d.csv', b));
            writetable(samp(:,{'patient_id','GA_lower','GA_upper','event','BMI'}), sampFile);
            
            % 使用超时机制调用R
            if ispc
                % Windows
                cmd = sprintf('timeout %s "%s" "%s" "%s" "%s" "%s" "%s" "%d"', ...
                    timeout_cmd, rscriptExe, rScript, sampFile, 'NA', 'NA', 'NA', 1);
            else
                % Linux/Mac
                cmd = sprintf('timeout %s Rscript "%s" "%s" "%s" "%s" "%s" "%d"', ...
                    timeout_cmd, rScript, sampFile, 'NA', 'NA', 'NA', 1);
            end
            
            [st, cout] = system(cmd);
            
            % 处理输出
            outFile = fullfile(boot_dir, sprintf('icenreg_boot_out_%d.csv', b));
            if ~exist(outFile,'file')
                % 如果输出文件不存在，尝试在原始目录中查找
                outFile = fullfile(OUT_DIR, sprintf('icenreg_boot_out_%d.csv', b));
            end
            
            if exist(outFile,'file')
                % 读取R输出
                sOut = readmatrix(outFile);
                tR = sOut(:,1); 
                if size(sOut, 2) > 1
                    Smat = sOut(:,2:end);
                else
                    % 处理极端情况：只有一列
                    Smat = ones(length(tR), 1);
                end
                
                % 获取BMI水平
                bmi_levels_file = fullfile(OUT_DIR,'icenreg_bmi_levels.csv');
                if exist(bmi_levels_file,'file')
                    bmi_levels = csvread(bmi_levels_file);
                else
                    bmi_levels = linspace(min(samp.BMI), max(samp.BMI), size(Smat,2));
                end
                
                % 计算最优时间点
                bmi_ref = mean(samp.BMI,'omitnan');
                Svals = interp2(bmi_levels, tR, Smat, bmi_ref, TGRID, 'linear', NaN);
                Svals(isnan(Svals)) = 1;
                reach_vec = 1 - Svals;
                Rvals = arrayfun(@(i) risk_fun(TGRID(i), 1 - Svals(i)), 1:length(TGRID));
                valid_idx = find(reach_vec >= 0.9);
                if isempty(valid_idx)
                    [~, idxmin] = min(Rvals);
                else
                    [~, rel] = min(Rvals(valid_idx)); 
                    idxmin = valid_idx(rel);
                end
                t_star_b = TGRID(idxmin);
                
                % 保存结果
                boot_tstars_mat(b, :) = t_star_b;
            end
        catch ME
            warning('Bootstrap 样本 %d 处理失败: %s', b, ME.message);
        end
    end
end

% 并行结束后按列汇总为 cell，每组一个向量，去掉 NaN
boot_tstars = cell(nGroups,1);
for gi = 1:nGroups
    col = boot_tstars_mat(:, gi);
    boot_tstars{gi} = col(~isnan(col));
end

% 汇总 bootstrap 结果
boot_summary = table();
for gi = 1:height(group_summary)
    arr = boot_tstars{gi};
    if isempty(arr)
        boot_summary.mean(gi,1) = NaN;
        boot_summary.std(gi,1) = NaN;
        boot_summary.ci_low(gi,1) = NaN;
        boot_summary.ci_high(gi,1) = NaN;
    else
        boot_summary.mean(gi,1) = mean(arr);
        boot_summary.std(gi,1) = std(arr);
        ci = quantile(arr,[0.025,0.975]);
        boot_summary.ci_low(gi,1) = ci(1);
        boot_summary.ci_high(gi,1) = ci(2);
    end
end
writetable([group_summary(:,1:end) array2table(table2array(boot_summary))], fullfile(OUT_DIR,'group_table_with_bootstrap.csv'));

% 清理临时文件夹
if exist(temp_dir, 'dir')
    rmdir(temp_dir, 's');
end

%% ---------- 4.6 Monte Carlo 误差分析 ----------
fprintf("正在运行 Monte Carlo 仿真 (B=%d) ...\n", MC_B);
mc_tstars = zeros(MC_B,1);

% 创建临时目录以避免文件冲突
temp_dir = fullfile(OUT_DIR, 'temp_mc');
if ~exist(temp_dir, 'dir'); mkdir(temp_dir); end

% 使用更加健壮的并行处理方法 - 拆分为较小的批次
batch_size = min(5, ceil(MC_B/4)); % 每批最多5个样本
num_batches = ceil(MC_B/batch_size);

% 创建等待进度条
progress_step = ceil(num_batches/10);
fprintf('开始 Monte Carlo 模拟，共 %d 批次:\n', num_batches);
progress = '';

for batch = 1:num_batches
    % 显示进度
    if mod(batch, progress_step) == 0 || batch == num_batches
        progress = [progress, '='];
        fprintf('[%s] %d/%d\n', progress, batch, num_batches);
    end
    
    start_idx = (batch-1)*batch_size + 1;
    end_idx = min(batch*batch_size, MC_B);
    batch_ids = start_idx:end_idx;
    
    % 使用 parfor 处理当前批次
    parfor b = batch_ids
        try
            % 设置超时（60秒）
            timeout_cmd = '180'; % 最长允许3分钟
            
            % 在唯一临时目录中创建文件
            mc_dir = fullfile(temp_dir, sprintf('mc_%d', b));
            if ~exist(mc_dir, 'dir'); mkdir(mc_dir); end
            
            % 扰动原始数据
            dfmc = T;
            sdY = nanstd(T.Y_frac);
            dfmc.Y_frac = dfmc.Y_frac + normrnd(0, sdY, size(dfmc.Y_frac));
            dfmc.Y_frac(dfmc.Y_frac<0) = 0; dfmc.Y_frac(dfmc.Y_frac>1) = 1;
            dfmc.gestational_weeks = dfmc.gestational_weeks + (rand(size(dfmc.gestational_weeks)) - 0.5);
            
            % 重建区间
            uids_m = unique(dfmc.patient_id);
            rows_mc = table('Size',[numel(uids_m),5], 'VariableTypes',{'cell','double','double','double','double'}, ...
                'VariableNames',{'patient_id','GA_lower','GA_upper','event','BMI'});
            for ii=1:numel(uids_m)
                pid = uids_m{ii};
                sub = dfmc(strcmp(dfmc.patient_id, pid), :);
                sub = sortrows(sub, 'gestational_weeks');
                ga = sub.gestational_weeks; y = sub.Y_frac; bmi = median(sub.BMI,'omitnan');
                idx = find(y >= THRESH, 1, 'first');
                if ~isempty(idx)
                    k = idx;
                    if k>1
                        L = ga(k-1);
                    else
                        L = 0;
                    end
                    R = ga(k); ev = 1;
                else
                    L = max(ga); R = Inf; ev = 0;
                end
                rows_mc.patient_id{ii} = pid;
                rows_mc.GA_lower(ii) = L;
                rows_mc.GA_upper(ii) = R;
                rows_mc.event(ii) = ev;
                rows_mc.BMI(ii) = bmi;
            end
            
            % 替换区间下界和上界为0的为极小正数
            rows_mc.GA_lower(rows_mc.GA_lower==0) = 1e-6;
            rows_mc.GA_upper(rows_mc.GA_upper==0) = 1e-6;
            % 过滤 BMI 为 NaN 的行
            rows_mc = rows_mc(~isnan(rows_mc.BMI), :);
            
            sampFile = fullfile(mc_dir, sprintf('mc_sample_%d.csv', b));
            writetable(rows_mc, sampFile);
            
            % 使用超时机制调用R
            if ispc
                % Windows
                cmd = sprintf('timeout %s "%s" "%s" "%s" "%s" "%s" "%s" "%d"', ...
                    timeout_cmd, rscriptExe, rScript, sampFile, 'NA', 'NA', 'NA', 1);
            else
                % Linux/Mac
                cmd = sprintf('timeout %s Rscript "%s" "%s" "%s" "%s" "%s" "%d"', ...
                    timeout_cmd, rScript, sampFile, 'NA', 'NA', 'NA', 1);
            end
            
            [st, cout] = system(cmd);
            
            % 处理输出
            outFile = fullfile(mc_dir, sprintf('icenreg_mc_out_%d.csv', b));
            if ~exist(outFile,'file')
                % 如果输出文件不存在，尝试在原始目录中查找
                outFile = fullfile(OUT_DIR, sprintf('icenreg_mc_out_%d.csv', b));
            end
            
            if exist(outFile,'file')
                % 读取R输出
                sOut = readmatrix(outFile);
                tR = sOut(:,1); 
                if size(sOut, 2) > 1
                    Smat = sOut(:,2:end);
                else
                    % 处理极端情况：只有一列
                    Smat = ones(length(tR), 1);
                end
                
                % 获取BMI水平
                bmi_levels_file = fullfile(OUT_DIR,'icenreg_bmi_levels.csv');
                if exist(bmi_levels_file,'file')
                    bmi_levels = csvread(bmi_levels_file);
                else
                    bmi_levels = linspace(min(rows_mc.BMI), max(rows_mc.BMI), size(Smat,2));
                end
                
                % 计算最优时间点
                bmi_ref = mean(rows_mc.BMI,'omitnan');
                Svals = interp2(bmi_levels, tR, Smat, bmi_ref, TGRID, 'linear', NaN);
                Svals(isnan(Svals)) = 1; % 保守估计
                reach_vec = 1 - Svals;
                Rvals = arrayfun(@(i) risk_fun(TGRID(i), 1 - Svals(i)), 1:length(TGRID));
                valid_idx = find(reach_vec >= 0.9);
                if isempty(valid_idx)
                    [~, idxmin] = min(Rvals);
                else
                    [~, rel] = min(Rvals(valid_idx)); idxmin = valid_idx(rel);
                end
                mc_tstars(b) = TGRID(idxmin);
            end
        catch ME
            warning('Monte Carlo 样本 %d 处理失败: %s', b, ME.message);
            mc_tstars(b) = NaN;
        end
    end
end

% 汇总 Monte Carlo 结果
mc_summary = table();
arr = mc_tstars(~isnan(mc_tstars));
if ~isempty(arr)
    mc_summary.mean = mean(arr);
    mc_summary.sd = std(arr);
    ci = quantile(arr,[0.025,0.975]);
    mc_summary.ci_low = ci(1);
    mc_summary.ci_high = ci(2);
else
    mc_summary.mean = NaN;
    mc_summary.sd = NaN;
    mc_summary.ci_low = NaN;
    mc_summary.ci_high = NaN;
end
writetable(mc_summary, fullfile(OUT_DIR,'mc_results.csv'));

% 清理临时文件夹
if exist(temp_dir, 'dir')
    rmdir(temp_dir, 's');
end

fprintf('全部完成。输出已放在 %s\n', OUT_DIR);

% 删除临时文件
delete(fullfile(OUT_DIR, 'boot_sample_*.csv'));
delete(fullfile(OUT_DIR, 'mc_sample_*.csv'));
delete(fullfile(OUT_DIR, 'icenreg_boot_out_*.csv'));
delete(fullfile(OUT_DIR, 'icenreg_mc_out_*.csv'));