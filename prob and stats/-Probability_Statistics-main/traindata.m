%% ENGI 2211 - Concrete Regression Analysis
% Robust implementation with proper data handling and validation

%% Step 1: Data Preparation
data = readtable('Concrete_Data.csv');

% Identify unique ages with sample counts
[uniqueAges, ~, idx] = unique(data.Age);
sampleCounts = accumarray(idx, 1);

% Split data using statistical validation
trainData = data(sampleCounts >= 50, :); 
testData = data(sampleCounts < 50, :);

%% Step 2: Get UNIQUE AGES FROM TRAINING SET
uniqueAges = unique(trainData.Age);  % Critical fix
logAges = log(uniqueAges);           % Now guaranteed same length as paramsCem

%% Step 3: Parameter Calculation
paramsCem = zeros(numel(uniqueAges), 2); % [b0, b1] for cement
paramsBind = zeros(numel(uniqueAges), 2); % [b0, b1] for binder

for i = 1:numel(uniqueAges)
    currentAge = uniqueAges(i);
    idx = (trainData.Age == currentAge);
    
    % Cement case regression
    X_cem = [ones(sum(idx), 1), trainData.wc_cem(idx)];
    [b_cem, ~] = regress(trainData.Comp_str_ln(idx), X_cem);
    paramsCem(i, :) = b_cem';
    
    % Binder case regression
    X_bind = [ones(sum(idx), 1), trainData.wc_binder(idx)];
    [b_bind, ~] = regress(trainData.Comp_str_ln(idx), X_bind);
    paramsBind(i, :) = b_bind';
end

%% Step 4: Store Parameters for 14-Day Model
% Find index for 14-day data
age14_idx = find(uniqueAges == 14);

% Extract parameters
b0_orig = paramsCem(age14_idx, 1);
b1_orig = paramsCem(age14_idx, 2);
b0_bind_orig = paramsBind(age14_idx, 1);
b1_bind_orig = paramsBind(age14_idx, 2);

% Define idx14 for cement and binder
idx14 = (trainData.Age == 14); % Logical index for 14-day data

%% Step 5: Parameter vs log(Age) Plots
figure('Position', [100 100 800 600])
subplot(2,2,1)
scatter(logAges, paramsCem(:,1), 40, 'b', 'filled')
hold on
plot(logAges, polyval(polyfit(logAges, paramsCem(:,1), 1), logAges), 'r--')
title('Cement: \beta_0 vs ln(Age)')
xlabel('ln(Age [days]') 
ylabel('\beta_0')

subplot(2,2,2)
scatter(logAges, paramsCem(:,2), 40, 'b', 'filled')
hold on
plot(logAges, polyval(polyfit(logAges, paramsCem(:,2), 1), logAges), 'r--')
title('Cement: \beta_1 vs ln(Age)')
xlabel('ln(Age [days]') 
ylabel('\beta_1')

subplot(2,2,3)
scatter(logAges, paramsBind(:,1), 40, 'm', 'filled')
hold on
plot(logAges, polyval(polyfit(logAges, paramsBind(:,1), 1), logAges), 'r--')
title('Binder: \beta_0 vs ln(Age)')
xlabel('ln(Age [days]') 
ylabel('\beta_0')

subplot(2,2,4)
scatter(logAges, paramsBind(:,2), 40, 'm', 'filled')
hold on
plot(logAges, polyval(polyfit(logAges, paramsBind(:,2), 1), logAges), 'r--')
title('Binder: \beta_1 vs ln(Age)')
xlabel('ln(Age [days]') 
ylabel('\beta_1')

%% Step 6: 14-Day Model Comparison
if any(trainData.Age == 14)
    figure('Position', [100 100 1000 400])
    
    % Cement case
    subplot(1,2,1)
    scatter(trainData.wc_cem(idx14), trainData.Comp_str_ln(idx14), 40, 'b', 'filled')
    hold on
    x_plot = linspace(min(trainData.wc_cem(idx14)), max(trainData.wc_cem(idx14)), 100);
    plot(x_plot, b0_orig + b1_orig * x_plot, 'k', 'LineWidth', 2)
    title('Cement Model (Age=14)')
    xlabel('Water:Cement Ratio'), ylabel('ln(Comp Strength)')
    
    % Binder case
    subplot(1,2,2)
    scatter(trainData.wc_binder(idx14), trainData.Comp_str_ln(idx14), 40, 'm', 'filled')
    hold on
    x_plot = linspace(min(trainData.wc_binder(idx14)), max(trainData.wc_binder(idx14)), 100);
    plot(x_plot, b0_bind_orig + b1_bind_orig * x_plot, 'k', 'LineWidth', 2)
    title('Binder Model (Age=14)')
    xlabel('Water:Binder Ratio'), ylabel('ln(Comp Strength)')
else
    warning('No 14-day data in training set')
end

%% Step 7: Residual Plots
figure('Position', [100 100 1200 400])
subplot(1,2,1)
histogram(trainResidualsCem, 'BinWidth', 2, 'Normalization','pdf')
hold on
histogram(testResidualsCem, 'BinWidth', 2, 'Normalization','pdf')
title('Cement Case Residuals')
xlabel('Residual (MPa)'), ylabel('Density')
legend({'Train','Test'})

subplot(1,2,2)
histogram(trainResidualsBind, 'BinWidth', 2, 'Normalization','pdf')
hold on
histogram(testResidualsBind, 'BinWidth', 2, 'Normalization','pdf')
title('Binder Case Residuals')
xlabel('Residual (MPa)'), ylabel('Density')

%% R² Table
fprintf('\nR² Values\n')
fprintf('-----------------------------------------\n')
fprintf('Case\t\tData\tTransformed\tRaw\n')
fprintf('Cement\t\tTrain\t%.4f\t\t%.4f\n', R2_cem_train_trans, R2_cem_train_raw)
fprintf('Cement\t\tTest\t%.4f\t\t%.4f\n', R2_cem_test_trans, R2_cem_test_raw)
fprintf('Binder\t\tTrain\t%.4f\t\t%.4f\n', R2_bind_train_trans, R2_bind_train_raw)
fprintf('Binder\t\tTest\t%.4f\t\t%.4f\n', R2_bind_test_trans, R2_bind_test_raw)
fprintf('-----------------------------------------\n')
