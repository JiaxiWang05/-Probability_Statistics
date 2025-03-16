%% Main Script Body
% All executable code comes BEFORE function definitions

% Common styling parameters
cmap = [0.7 0.2 0.2; 0.2 0.5 0.7]; % Red/blue color scheme
markerType = 'o';
lineStyle = {'-', '--'};
axisFont = 12;
legFont = 10;

%% Core Working Version (Simplified)
% Clear any previous settings
reset(groot)

% Step 1: Data preparation (critical for dimension matching)
% Ensure 'data' is defined and contains necessary columns
if ~exist('data', 'var') || isempty(data)
    error('Data variable is not defined or is empty.');
end

% Filter training data based on unique ages with sufficient samples
trainData = data(ismember(data.Age, uniqueAges(sampleCounts >= 50)), :);
uniqueTrainAges = unique(trainData.Age); % Must use TRAINING ages only
logAges = log(uniqueTrainAges); % Calculated from training data

% Step 2: Parameter calculation (ensure dimensions match)
paramsCem = zeros(length(uniqueTrainAges), 2);
paramsBind = zeros(length(uniqueTrainAges), 2);

for i = 1:length(uniqueTrainAges)
    age = uniqueTrainAges(i);
    idx = (trainData.Age == age);
    
    % Cement model
    X_cem = [ones(sum(idx), 1), trainData.wc_cem(idx)];
    paramsCem(i, :) = X_cem \ trainData.Comp_str_ln(idx);
    
    % Binder model
    X_bind = [ones(sum(idx), 1), trainData.wc_binder(idx)];
    paramsBind(i, :) = X_bind \ trainData.Comp_str_ln(idx);
end

% Step 3: Parameter plots (fixed dimensions)
figure('Position', [100, 100, 1200, 800]) % Set figure size
subplot(2, 2, 1)
scatter(logAges, paramsCem(:, 1), 40, 'b', 'filled')
title('Cement: β₀ vs ln(Age)')
xlabel('ln(Age)'), ylabel('β₀')
grid on

subplot(2, 2, 2)
scatter(logAges, paramsCem(:, 2), 40, 'b', 'filled')
title('Cement: β₁ vs ln(Age)')
xlabel('ln(Age)'), ylabel('β₁')
grid on

subplot(2, 2, 3)
scatter(logAges, paramsBind(:, 1), 40, 'r', 'filled')
title('Binder: β₀ vs ln(Age)')
xlabel('ln(Age)'), ylabel('β₀')
grid on

subplot(2, 2, 4)
scatter(logAges, paramsBind(:, 2), 40, 'r', 'filled')
title('Binder: β₁ vs ln(Age)')
xlabel('ln(Age)'), ylabel('β₁')
grid on

% Step 4: 14-Day Model (with existence check)
if ismember(14, uniqueTrainAges)
    figure('Position', [100, 100, 1200, 500]) % Set figure size
    idx14 = (trainData.Age == 14);
    
    % Cement model
    subplot(1, 2, 1)
    scatter(trainData.wc_cem(idx14), trainData.Comp_str_ln(idx14), 40, 'b', 'filled')
    title('Cement Model (14 Days)')
    xlabel('Water:Cement'), ylabel('ln(Strength)')
    grid on
    
    % Binder model
    subplot(1, 2, 2)
    scatter(trainData.wc_binder(idx14), trainData.Comp_str_ln(idx14), 40, 'r', 'filled')
    title('Binder Model (14 Days)')
    xlabel('Water:Binder'), ylabel('ln(Strength)')
    grid on
else
    warning('No 14-day data in training set');
end

%% Enhanced Residual Plots with KDE
figure('Position', [100 100 1200 600], 'Name', 'Residual Analysis');
tiledlayout(1,2);

% Cement residuals
nexttile(1);
plot_residuals('Cement');

% Binder residuals
nexttile(2);
plot_residuals('Binder');

%% Function Definitions - MUST COME AFTER ALL EXECUTABLE CODE
function plot_model_comparison(feature, params, caseName)
    % Shared parameters
    trainData = evalin('base', 'trainData');
    age14 = evalin('base', 'age14');
    log_age14 = evalin('base', 'log_age14');
    ensembleMdl = evalin('base', 'ensembleMdl');
    
    idx14 = trainData.Age == age14;
    x_vals = trainData.(feature)(idx14);
    y_vals = trainData.Comp_str_ln(idx14);
    
    % Create plot range
    x_plot = linspace(min(x_vals), max(x_vals), 100);
    
    % Original model
    y_orig = params(1) + params(2)*x_plot;
    
    % Estimated model
    X_second = [ones(length(trainData.logAge),1) trainData.logAge];
    b0_est = X_second\ensembleMdl.Bias;
    b1_est = X_second\ensembleMdl.Coefficients(1);
    y_est = b0_est(1) + b0_est(2)*log_age14 + ...
           (b1_est(1) + b1_est(2)*log_age14)*x_plot;

    % Plotting
    hold on;
    scatter(x_vals, y_vals, 40, 'k', 'filled');
    plot(x_plot, y_orig, 'Color', [0.8 0.2 0.2], 'LineWidth', 2);
    plot(x_plot, y_est, '--', 'Color', [0.2 0.6 0.3], 'LineWidth', 2);
    
    % Formatting
    xlabel(sprintf('Water:%s Ratio', caseName));
    ylabel('$\ln$(Compressive Strength)', 'Interpreter', 'latex');
    title(sprintf('%s Case - %d Days', caseName, age14));
    legend('Data', 'Original Model', 'Estimated Model', 'Location', 'best');
    grid on;
    set(gca, 'FontSize', 11);
end

function plot_residuals(caseType)
    trainRes = evalin('base', 'trainData.Comp_strength - trainPredRaw');
    testRes = evalin('base', 'testData.Comp_strength - testPredRaw');
    
    % Kernel density estimation
    [f_train,xi_train] = ksdensity(trainRes);
    [f_test,xi_test] = ksdensity(testRes);
    
    hold on;
    area(xi_train, f_train, 'FaceColor', [0.7 0.2 0.2], 'FaceAlpha', 0.5);
    area(xi_test, f_test, 'FaceColor', [0.2 0.4 0.7], 'FaceAlpha', 0.5);
    
    title(sprintf('%s Case Residuals', caseType));
    xlabel('Residual (MPa)');
    ylabel('Probability Density');
    legend('Training', 'Testing');
    grid on;
    xlim([-25 25]);
    set(gca, 'FontSize', 11);
end
