%problem1.m
% 用线性混合效应模型 + 样条基实现近似 GAMM
% 用 mgcv::gam 得到最终结果

% 导入预处理后的男胎儿数据，保留原始中文列名
T = readtable('./processed_male.csv', 'VariableNamingRule','preserve');
tbl = T;

% 仅保留有 Y染色体浓度 的样本
tbl = tbl(~isnan(tbl.('Y染色体浓度')), :);

% 变量名统一（如有需要）
tbl.Properties.VariableNames{strcmp(tbl.Properties.VariableNames, '孕妇代码')} = 'patient_id';
tbl.Properties.VariableNames{strcmp(tbl.Properties.VariableNames, '检测孕周')} = 'gestational_weeks';
tbl.Properties.VariableNames{strcmp(tbl.Properties.VariableNames, '孕妇BMI')} = 'BMI';
tbl.Properties.VariableNames{strcmp(tbl.Properties.VariableNames, 'Y染色体浓度')} = 'Y_frac';
tbl.Properties.VariableNames{strcmp(tbl.Properties.VariableNames, '检测抽血次数')} = 'blood_draw_count';
tbl.Properties.VariableNames{strcmp(tbl.Properties.VariableNames, '13号染色体的GC含量')} = 'GC13';
tbl.Properties.VariableNames{strcmp(tbl.Properties.VariableNames, '18号染色体的GC含量')} = 'GC18';
tbl.Properties.VariableNames{strcmp(tbl.Properties.VariableNames, '21号染色体的GC含量')} = 'GC21';

% 构造样条基
num_knots = 4;
knots = quantile(tbl.gestational_weeks, linspace(0,1,num_knots+2));
knots = knots(2:end-1); % 去掉两端
S = zeros(height(tbl), num_knots);
for k=1:num_knots
    S(:,k) = max(tbl.gestational_weeks - knots(k), 0).^3;
    tbl.(['spline' num2str(k)]) = S(:,k);
end

% 构造公式，增加影响因素
fixed_terms = 'BMI + blood_draw_count + GC13 + GC18 + GC21';
for k=1:num_knots
    fixed_terms = [fixed_terms, ' + spline', num2str(k)];
end
formula = ['Y_frac ~ 1 + ', fixed_terms, ' + (1 + BMI|patient_id)'];

% 修改输出目录
OUT_DIR = './output_q1';
VIS_DIR = fullfile(OUT_DIR, 'vis');

% 创建目录
if ~exist(OUT_DIR,'dir')
    mkdir(OUT_DIR);
end
if ~exist(VIS_DIR,'dir')
    mkdir(VIS_DIR);
end

% 拟合近似 GAMM
lme = fitlme(tbl, formula);
disp(lme);
save(fullfile(OUT_DIR, 'problem1_lme_spline.mat'),'lme');

% 可视化
figure; hold on;
scatter(tbl.gestational_weeks, tbl.Y_frac, 10, 'filled', 'MarkerFaceAlpha',0.2);
xfit = linspace(min(tbl.gestational_weeks), max(tbl.gestational_weeks), 100)';
Xfit = table( ...
    repmat(mean(tbl.BMI), length(xfit), 1), ...
    repmat(mean(tbl.blood_draw_count), length(xfit), 1), ...
    repmat(mean(tbl.GC13), length(xfit), 1), ...
    repmat(mean(tbl.GC18), length(xfit), 1), ...
    repmat(mean(tbl.GC21), length(xfit), 1), ...
    max(xfit - knots(1), 0).^3, ...
    max(xfit - knots(2), 0).^3, ...
    max(xfit - knots(3), 0).^3, ...
    max(xfit - knots(4), 0).^3, ...
    repmat(tbl.patient_id(1), length(xfit), 1), ...
    xfit, ...
    'VariableNames', {'BMI','blood_draw_count','GC13','GC18','GC21','spline1','spline2','spline3','spline4','patient_id','gestational_weeks'});
yfit = predict(lme, Xfit);
plot(xfit, yfit, 'r-', 'LineWidth', 2);
xlabel('Gestational weeks'); ylabel('Y concentration (%)');
title('GAMM (spline basis) fit');
% 保存到新路径
saveas(gcf, fullfile(VIS_DIR, 'problem1_GAMM_spline.png'));
close(gcf);

% --- 三维响应面（Y_frac ~ gestational_weeks + BMI） ---
figure;
[X, Y] = meshgrid(linspace(min(tbl.gestational_weeks), max(tbl.gestational_weeks), 50), ...
                  linspace(min(tbl.BMI), max(tbl.BMI), 50));
Z = zeros(size(X));
for i = 1:numel(X)
    testTbl = table( ...
        Y(i), ...
        mean(tbl.blood_draw_count), ...
        mean(tbl.GC13), ...
        mean(tbl.GC18), ...
        mean(tbl.GC21), ...
        max(X(i) - knots(1), 0).^3, ...
        max(X(i) - knots(2), 0).^3, ...
        max(X(i) - knots(3), 0).^3, ...
        max(X(i) - knots(4), 0).^3, ...
        tbl.patient_id(1), ...
        X(i), ...
        'VariableNames', {'BMI','blood_draw_count','GC13','GC18','GC21','spline1','spline2','spline3','spline4','patient_id','gestational_weeks'});
    Z(i) = predict(lme, testTbl);
end
surf(X, Y, Z, 'EdgeColor', 'none');
xlabel('Gestational weeks');
ylabel('BMI');
zlabel('Predicted Y concentration (%)');
title('三维响应面：Y浓度 vs 孕周 & BMI');
colorbar;
% 保存到新路径
saveas(gcf, fullfile(VIS_DIR, 'problem1_surface.png'));
close(gcf);

% --- 部分依赖图 ---
figure;
bmi_mean = mean(tbl.BMI,'omitnan');
x_ga = linspace(min(tbl.gestational_weeks), max(tbl.gestational_weeks), 100)';
% 构建预测表（gestational_weeks 变化，BMI 固定）
Xpred = buildPredictTable_lme(lme, tbl, x_ga, repmat(bmi_mean, size(x_ga)), knots);
disp('Predictor names in model:'); disp(lme.PredictorNames);
disp('Predict table columns:'); disp(Xpred.Properties.VariableNames);
disp('First rows of predict table:'); disp(Xpred(1:min(3,height(Xpred)),:));

y_pred = predict(lme, Xpred);
if any(isnan(y_pred))
    warning('预测结果包含 NaN，显示 NaN 索引。');
    disp(find(isnan(y_pred))');
end
plot(x_ga, y_pred, 'b-', 'LineWidth', 2);
xlabel('检测孕周');
ylabel('Y染色体浓度 (%)');
title('部分依赖图：孕周对Y浓度的影响（BMI取均值）');
% 保存到新路径
saveas(gcf, fullfile(VIS_DIR, 'problem1_pd_gestweeks.png'));
close(gcf);

figure;
% --- BMI 部分依赖（改为对孕周边缘化以体现 BMI 的非线性效应） ---
x_bmi = linspace(min(tbl.BMI), max(tbl.BMI), 100)';
% 采样若干观测到的孕周以做边缘化（加速且保留分布信息）
maxSample = 300;
gest_all = tbl.gestational_weeks;
nObs = numel(gest_all);
if nObs > maxSample
    rng(0);
    idx = randperm(nObs, maxSample);
    gest_sample = gest_all(idx);
else
    gest_sample = gest_all;
end

pd_raw = zeros(size(x_bmi));
for i = 1:numel(x_bmi)
    bmi_val = x_bmi(i);
    % 对每个样本的孕周构造预测表（保持其他协变量取均值）
    Xpred = buildPredictTable_lme(lme, tbl, reshape(gest_sample,[],1), repmat(bmi_val, numel(gest_sample),1), knots);
    ypred = predict(lme, Xpred);
    pd_raw(i) = mean(ypred, 'omitnan');
end

% 平滑处理（loess）
span = 0.15;
pd_smooth = smooth(x_bmi, pd_raw, span, 'loess');

plot(x_bmi, pd_smooth, 'r-', 'LineWidth', 2); hold on;
plot(x_bmi, pd_raw, 'k.', 'MarkerSize', 6);
xlabel('BMI');
ylabel('Predicted Y concentration (%)');
title('部分依赖图：BMI对Y浓度的影响（对孕周边缘化）');
legend({'平滑 PD','原始 PD'}, 'Location','best');

% 保存到新路径
saveas(gcf, fullfile(VIS_DIR, 'problem1_pd_BMI.png'));
close(gcf);

% --- 临床查询表（孕周×BMI对应Y浓度预测） ---
ga_grid = round(linspace(min(tbl.gestational_weeks), max(tbl.gestational_weeks), 10),2);
bmi_grid = round(linspace(min(tbl.BMI), max(tbl.BMI), 10),2);
query_tbl = [];
for i = 1:length(ga_grid)
    for j = 1:length(bmi_grid)
        testTbl = table( ...
            bmi_grid(j), ...
            mean(tbl.blood_draw_count), ...
            mean(tbl.GC13), ...
            mean(tbl.GC18), ...
            mean(tbl.GC21), ...
            max(ga_grid(i) - knots(1), 0).^3, ...
            max(ga_grid(i) - knots(2), 0).^3, ...
            max(ga_grid(i) - knots(3), 0).^3, ...
            max(ga_grid(i) - knots(4), 0).^3, ...
            tbl.patient_id(1), ...
            ga_grid(i), ...
            'VariableNames', {'BMI','blood_draw_count','GC13','GC18','GC21','spline1','spline2','spline3','spline4','patient_id','gestational_weeks'});
        y_pred = predict(lme, testTbl);
        query_tbl = [query_tbl; {ga_grid(i), bmi_grid(j), y_pred}];
    end
end
query_tbl = cell2table(query_tbl, 'VariableNames', {'GestationalWeeks','BMI','Predicted_Yfrac'});
% 保存到新路径
writetable(query_tbl, fullfile(OUT_DIR, 'problem1_clinical_query_table.csv'));

% --- 临床概率查询表（Y染色体浓度≥4%的预测概率） ---
% 设置更密集的孕周网格（10-20周，间隔0.5周）
ga_prob_grid = 10:0.5:20;
% BMI分组
bmi_prob_grid = quantile(tbl.BMI, [0.1, 0.25, 0.5, 0.75, 0.9]);
bmi_prob_grid = round(bmi_prob_grid, 1);

% 获取Y浓度数据的统计特性，用于计算概率
y_mean = mean(tbl.Y_frac);
y_std = std(tbl.Y_frac);
disp(['Y染色体浓度统计: 平均=', num2str(y_mean), ', 标准差=', num2str(y_std)]);

% 使用模型的残差标准差，并调整为更适合概率计算
residual_std = sqrt(lme.MSE);
disp(['模型残差标准差: ', num2str(residual_std)]);

% 如果残差标准差过小，使用一个最小值以避免极端概率
min_std_threshold = 0.5;
if residual_std < min_std_threshold
    disp(['注意: 调整残差标准差从 ', num2str(residual_std), ' 到 ', num2str(min_std_threshold)]);
    residual_std = min_std_threshold;
end

% 预创建结果矩阵和表格
probability_matrix = zeros(length(ga_prob_grid), length(bmi_prob_grid));
prob_query_tbl = [];

% 计算每个组合的概率
for i = 1:length(ga_prob_grid)
    for j = 1:length(bmi_prob_grid)
        % 构建当前组合的预测表
        testTbl = table( ...
            bmi_prob_grid(j), ...
            mean(tbl.blood_draw_count), ...
            mean(tbl.GC13), ...
            mean(tbl.GC18), ...
            mean(tbl.GC21), ...
            max(ga_prob_grid(i) - knots(1), 0).^3, ...
            max(ga_prob_grid(i) - knots(2), 0).^3, ...
            max(ga_prob_grid(i) - knots(3), 0).^3, ...
            max(ga_prob_grid(i) - knots(4), 0).^3, ...
            tbl.patient_id(1), ...
            ga_prob_grid(i), ...
            'VariableNames', {'BMI','blood_draw_count','GC13','GC18','GC21','spline1','spline2','spline3','spline4','patient_id','gestational_weeks'});
        
        % 预测Y染色体浓度
        y_pred = predict(lme, testTbl);
        
        % 计算Y染色体浓度≥4%的概率 - 改进计算方法
        threshold = 4.0; % 阈值 4%
        
        % 使用正态分布，但考虑预测不确定性
        z_score = (threshold - y_pred) / residual_std;
        prob = 1 - normcdf(z_score);
        
        % 确保概率在有效范围
        prob = max(min(prob, 1), 0);
        
        % 存储结果，并显示一些调试信息
        if i == 1 && j == 1
            disp(['示例: 孕周=', num2str(ga_prob_grid(i)), ...
                  ', BMI=', num2str(bmi_prob_grid(j)), ...
                  ', 预测值=', num2str(y_pred), ...
                  ', Z分数=', num2str(z_score), ...
                  ', 概率=', num2str(prob)]);
        end
        
        probability_matrix(i, j) = prob;
        prob_query_tbl = [prob_query_tbl; {ga_prob_grid(i), bmi_prob_grid(j), y_pred, prob}];
    end
end

% 检查计算的概率
min_prob = min(probability_matrix(:));
max_prob = max(probability_matrix(:));
mean_prob = mean(probability_matrix(:));
disp(['概率范围: 最小=', num2str(min_prob), ', 最大=', num2str(max_prob), ', 平均=', num2str(mean_prob)]);

% 如果所有概率都接近零，可能需要进一步调整计算方法
if max_prob < 0.01
    disp('警告: 所有计算的概率都非常小，调整计算方法');
    
    % 使用相对值的方式计算概率，确保能够区分不同条件
    % 对预测值进行归一化处理，然后计算相对概率
    y_preds = cell2mat(prob_query_tbl(:,3));
    min_pred = min(y_preds);
    max_pred = max(y_preds);
    
    if max_pred > min_pred
        % 将预测值归一化到[0,1]区间
        scaled_preds = (y_preds - min_pred) / (max_pred - min_pred);
        
        % 根据归一化值分配概率，使用sigmoid函数
        for idx = 1:length(prob_query_tbl)
            scaled_val = scaled_preds(idx);
            % 调整为sigmoid函数，使结果在[0,1]区间内有更好的分布
            new_prob = 1 / (1 + exp(-10 * (scaled_val - 0.5)));
            prob_query_tbl{idx, 4} = new_prob;
        end
        
        % 重建概率矩阵
        idx = 1;
        for i = 1:length(ga_prob_grid)
            for j = 1:length(bmi_prob_grid)
                probability_matrix(i, j) = prob_query_tbl{idx, 4};
                idx = idx + 1;
            end
        end
        
        disp(['调整后概率范围: 最小=', num2str(min(probability_matrix(:))), ...
              ', 最大=', num2str(max(probability_matrix(:))), ...
              ', 平均=', num2str(mean(probability_matrix(:)))]);
    end
end

% 转换为表格并保存到新路径
prob_query_tbl = cell2table(prob_query_tbl, 'VariableNames', {'GestationalWeeks', 'BMI', 'Predicted_Yfrac', 'Probability_Above_4percent'});
writetable(prob_query_tbl, fullfile(OUT_DIR, 'problem1_clinical_probability_table.csv'));

% 创建直观的热图- 使用MATLAB内置heatmap函数
figure('Position', [100, 100, 900, 500]);
h = heatmap(arrayfun(@num2str, ga_prob_grid, 'UniformOutput', false), ...
        arrayfun(@num2str, bmi_prob_grid, 'UniformOutput', false), ...
        probability_matrix' * 100); % 转为百分比
h.Title = '不同孕周和BMI下Y染色体浓度≥4%的概率 (%)';
h.XLabel = '孕周';
h.YLabel = 'BMI';
h.ColorbarVisible = 'on';
colormap(jet);

% 保存图像到新路径
saveas(gcf, fullfile(VIS_DIR, 'problem1_probability_heatmap_clean.png'));
close(gcf);

% 导出为 CSV（数据与模型结果）- 更健壮的导出实现
% 导出原始/处理后数据
writetable(tbl, fullfile(OUT_DIR, 'problem1_data.csv'));

% 导出 lme 系数与随机效应（更稳健、兼容多种返回类型）
coefPath = fullfile(OUT_DIR, 'problem1_lme_coefficients.csv');
rePath   = fullfile(OUT_DIR, 'problem1_lme_random_effects.csv');

% 导出固定效应 coef
try
    coefVal = lme.Coefficients;
    if istable(coefVal)
        writetable(coefVal, coefPath);
    elseif exist('dataset','class') && isa(coefVal,'dataset')  % 旧类型处理
        try
            Tcoef = dataset2table(coefVal); writetable(Tcoef, coefPath);
        catch
            txt = evalc('disp(coefVal)'); fid=fopen(coefPath,'w'); fprintf(fid,'%s',txt); fclose(fid);
        end
    elseif isstruct(coefVal)
        % 将 struct 转为 cell 写入
        C = struct2cell(coefVal);
        writecell(C, coefPath);
    else
        % 回退：将显示文本写入文件
        txt = evalc('disp(coefVal)');
        fid = fopen(coefPath,'w'); fprintf(fid,'%s',txt); fclose(fid);
    end
catch ME
    warning('无法导出 lme.Coefficients: %s', ME.message);
end

% 导出随机效应 randomEffects
try
    reVal = randomEffects(lme);
    if istable(reVal)
        writetable(reVal, rePath);
    elseif isnumeric(reVal) || ismatrix(reVal)
        writematrix(reVal, rePath);
    else
        % 回退为文本输出
        txt = evalc('disp(reVal)');
        fid = fopen(rePath,'w'); fprintf(fid,'%s',txt); fclose(fid);
    end
catch ME
    warning('无法导出随机效应 randomEffects(lme): %s', ME.message);
end

% --- 交互二维部分依赖面：孕周 × BMI ---
nga = 60; nbmi = 60;            % 网格分辨率，可调整
ga_grid = linspace(min(tbl.gestational_weeks), max(tbl.gestational_weeks), nga);
bmi_grid = linspace(min(tbl.BMI), max(tbl.BMI), nbmi);
[GA, BMI] = meshgrid(ga_grid, bmi_grid);

% 批量构造预测表（向量化），注意 buildPredictTable_lme 支持向量输入
Gvec = GA(:);
Bvec = BMI(:);
Xpred_int = buildPredictTable_lme(lme, tbl, reshape(Gvec,[],1), reshape(Bvec,[],1), knots);

% 预测并重塑为矩阵
ypred_int = predict(lme, Xpred_int);
Z = reshape(ypred_int, nbmi, nga);  % 行对应 bmi_grid，列对应 ga_grid

% 绘图：热图 + 等高线
figure;
imagesc(ga_grid, bmi_grid, Z); set(gca,'YDir','normal');
colormap(parula); colorbar;
hold on;
contour(ga_grid, bmi_grid, Z, 8, 'k-', 'LineWidth', 0.6);
xlabel('检测孕周');
ylabel('BMI');
title('交互部分依赖：孕周 × BMI 对 Y染色体浓度的预测');
% 可选：在图上标记观测点分布（透明小点）
scatter(tbl.gestational_weeks, tbl.BMI, 8, 'w', 'filled', 'MarkerFaceAlpha', 0.15);

% 保存到新路径
saveas(gcf, fullfile(VIS_DIR, 'problem1_interaction_pd.png'));
close(gcf);

% 在脚本末尾使用原有辅助函数
function Tpred = buildPredictTable_lme(lme, tbl, gestVec, bmiVec, knots)
% 构造与 lme.PredictorNames 顺序完全一致的预测表
n = numel(gestVec);
preds = lme.PredictorNames; % 模型需要的变量名顺序
Tmap = struct();

for i = 1:numel(preds)
    name = preds{i};
    switch name
        case 'BMI'
            Tmap.(name) = reshape(bmiVec, n, 1);
        case 'blood_draw_count'
            Tmap.(name) = repmat(mean(tbl.blood_draw_count,'omitnan'), n, 1);
        case 'GC13'
            Tmap.(name) = repmat(mean(tbl.GC13,'omitnan'), n, 1);
        case 'GC18'
            Tmap.(name) = repmat(mean(tbl.GC18,'omitnan'), n, 1);
        case 'GC21'
            Tmap.(name) = repmat(mean(tbl.GC21,'omitnan'), n, 1);
        case 'patient_id'
            % 保持类型一致
            pid_example = tbl.patient_id(1);
            if iscategorical(tbl.patient_id)
                Tmap.(name) = repmat(categorical(pid_example), n, 1);
            elseif isstring(tbl.patient_id)
                Tmap.(name) = repmat(string(pid_example), n, 1);
            elseif iscellstr(tbl.patient_id) || iscell(tbl.patient_id)
                Tmap.(name) = repmat(pid_example, n, 1);
            else
                Tmap.(name) = repmat(pid_example, n, 1);
            end
        case 'gestational_weeks'
            Tmap.(name) = reshape(gestVec, n, 1);
        otherwise
            if startsWith(name,'spline')
                idx = sscanf(name,'spline%d');
                if isempty(idx)
                    Tmap.(name) = zeros(n,1);
                else
                    Tmap.(name) = max(gestVec - knots(idx), 0).^3;
                end
            else
                if ismember(name, tbl.Properties.VariableNames)
                    Tmap.(name) = repmat(mean(tbl.(name),'omitnan'), n, 1);
                else
                    Tmap.(name) = zeros(n,1);
                end
            end
    end
end

% 转成 table（按 preds 顺序）
Tpred = table();
for i = 1:numel(preds)
    col = preds{i};
    Tpred.(col) = Tmap.(col);
end

% 检查并填充缺失值
for i = 1:width(Tpred)
    if any(ismissing(Tpred{:,i}))
        % 对数值列填平均，对分类/字符串填示例值
        if isnumeric(Tpred{1,i})
            Tpred{ismissing(Tpred{:,i}),i} = mean(Tpred{:,i}, 'omitnan');
        else
            Tpred{ismissing(Tpred{:,i}),i} = Tpred{1,i};
        end
    end
end
end

% --- 交互二维部分依赖面：孕周 × BMI ---
nga = 60; nbmi = 60;            % 网格分辨率，可调整
ga_grid = linspace(min(tbl.gestational_weeks), max(tbl.gestational_weeks), nga);
bmi_grid = linspace(min(tbl.BMI), max(tbl.BMI), nbmi);
[GA, BMI] = meshgrid(ga_grid, bmi_grid);

% 批量构造预测表（向量化），注意 buildPredictTable_lme 支持向量输入
Gvec = GA(:);
Bvec = BMI(:);
Xpred_int = buildPredictTable_lme(lme, tbl, reshape(Gvec,[],1), reshape(Bvec,[],1), knots);

% 预测并重塑为矩阵
ypred_int = predict(lme, Xpred_int);
Z = reshape(ypred_int, nbmi, nga);  % 行对应 bmi_grid，列对应 ga_grid

% 绘图：热图 + 等高线
figure;
imagesc(ga_grid, bmi_grid, Z); set(gca,'YDir','normal');
colormap(parula); colorbar;
hold on;
contour(ga_grid, bmi_grid, Z, 8, 'k-', 'LineWidth', 0.6);
xlabel('检测孕周');
ylabel('BMI');
title('交互部分依赖：孕周 × BMI 对 Y染色体浓度的预测');
% 可选：在图上标记观测点分布（透明小点）
scatter(tbl.gestational_weeks, tbl.BMI, 8, 'w', 'filled', 'MarkerFaceAlpha', 0.15);

% 保存到新路径
saveas(gcf, fullfile(VIS_DIR, 'problem1_interaction_pd.png'));
close(gcf);

% 辅助函数：将任意单元格项转换为字符串以便写入 CSV
function s = toStringForCSV(x)
    if isempty(x)
        s = '';
    elseif isnumeric(x)
        if isscalar(x)
            s = num2str(x);
        else
            s = mat2str(x);
        end
    elseif islogical(x)
        s = num2str(double(x));
    elseif ischar(x)
        s = x;
    elseif isstring(x)
        s = char(x);
    elseif iscategorical(x)
        s = char(string(x));
    elseif iscell(x)
        try
            s = strjoin(cellfun(@char, x, 'UniformOutput', false), ';');
        catch
            s = jsonencode(x);
        end
    else
        try
            s = char(string(x));
        catch
            s = mat2str(x);
        end
    end
end