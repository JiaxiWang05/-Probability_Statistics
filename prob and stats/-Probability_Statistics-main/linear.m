%% Step 1: Load data and split into training/testing sets
% Load the data
data = readtable("Concrete_Data.csv");

% Find unique ages
unique_ages = unique(data.Age);
num_unique_ages = length(unique_ages);

% Initialize training and testing indices
train_idx = false(height(data), 1);
test_idx = false(height(data), 1);

% Split data based on sample count per age
fprintf('Unique age values: %d\n', num_unique_ages);
fprintf('Age (days)\tSamples\tAssignment\n');
for i = 1:num_unique_ages
    age = unique_ages(i);
    age_idx = (data.Age == age);
    num_samples = sum(age_idx);
    
    if num_samples > 50
        train_idx = train_idx | age_idx;
        assignment = 'Training';
    else
        test_idx = test_idx | age_idx;
        assignment = 'Testing';
    end
    
    fprintf('%.0f\t\t%d\t%s\n', age, num_samples, assignment);
end

% Create training and testing datasets
train_data = data(train_idx, :);
test_data = data(test_idx, :);
fprintf('Training set: %d samples, Testing set: %d samples\n', height(train_data), height(test_data));

%% Step 2: Transform the data
% Log transform of compressive strength
Comp_str_ln = log(data.Comp_strength);

% Calculate water-to-cement ratio
wc_cem = data.Water ./ data.Cement; 

% Calculate water-to-binder ratio (cement + slag + ash)
wc_binder = data.Water ./ sum([data.Cement data.Slag data.Ash], 2);

% Add transformed variables to the data table
data.Comp_str_ln = Comp_str_ln;
data.wc_cem = wc_cem;
data.wc_binder = wc_binder;

% Split transformed variables for training and testing
train_Comp_str_ln = Comp_str_ln(train_idx);
train_wc_cem = wc_cem(train_idx);
train_wc_binder = wc_binder(train_idx);
train_Age = data.Age(train_idx);

test_Comp_str_ln = Comp_str_ln(test_idx);
test_wc_cem = wc_cem(test_idx);
test_wc_binder = wc_binder(test_idx);
test_Age = data.Age(test_idx);

%% Step 3: Perform first regressions for each unique age
% Get unique ages in training data
unique_train_ages = unique(train_Age);
num_unique_train_ages = length(unique_train_ages);

% Initialize arrays for regression parameters
cement_b0 = zeros(num_unique_train_ages, 1);
cement_b1 = zeros(num_unique_train_ages, 1);
binder_b0 = zeros(num_unique_train_ages, 1);
binder_b1 = zeros(num_unique_train_ages, 1);
log_ages = log(unique_train_ages);

% Perform regression for each age
fprintf('\nRegression parameters for each age:\n');
fprintf('Age (days)\tCement Case (b0, b1)\t\tBinder Case (b0, b1)\n');

for i = 1:num_unique_train_ages
    age = unique_train_ages(i);
    age_idx = (train_Age == age);
    
    % Regression for cement case
    X_cem = [ones(sum(age_idx), 1), train_wc_cem(age_idx)];
    Y = train_Comp_str_ln(age_idx);
    
    cement_params = X_cem \ Y;
    cement_b0(i) = cement_params(1);
    cement_b1(i) = cement_params(2);
    
    % Regression for binder case
    X_binder = [ones(sum(age_idx), 1), train_wc_binder(age_idx)];
    binder_params = X_binder \ Y;
    binder_b0(i) = binder_params(1);
    binder_b1(i) = binder_params(2);
    
    fprintf('%.0f\t\t(%.4f, %.4f)\t\t(%.4f, %.4f)\n', age, cement_b0(i), cement_b1(i), binder_b0(i), binder_b1(i));
end

% Plot b0 and b1 vs. log(Age)
figure;
subplot(2, 2, 1);
plot(log_ages, cement_b0, 'o-', 'LineWidth', 1.5);
xlabel('log(Age)'); ylabel('b0 (Intercept)');
title('Cement Case: b0 vs. log(Age)'); grid on;

subplot(2, 2, 3);
plot(log_ages, cement_b1, 'o-', 'LineWidth', 1.5);
xlabel('log(Age)'); ylabel('b1 (Slope)');
title('Cement Case: b1 vs. log(Age)'); grid on;

subplot(2, 2, 2);
plot(log_ages, binder_b0, 'o-', 'LineWidth', 1.5);
xlabel('log(Age)'); ylabel('b0 (Intercept)');
title('Binder Case: b0 vs. log(Age)'); grid on;

subplot(2, 2, 4);
plot(log_ages, binder_b1, 'o-', 'LineWidth', 1.5);
xlabel('log(Age)'); ylabel('b1 (Slope)');
title('Binder Case: b1 vs. log(Age)'); grid on;

sgtitle('Regression Parameters vs. log(Age)');

%% Step 4: Perform second regressions (modeling b0 and b1 as functions of Age)
% Prepare design matrix for log(Age)
X_age_log = [ones(num_unique_train_ages, 1), log_ages];

% Cement case regressions
cement_b0_params = X_age_log \ cement_b0;
cement_b1_params = X_age_log \ cement_b1;

% Binder case regressions
binder_b0_params = X_age_log \ binder_b0;
binder_b1_params = X_age_log \ binder_b1;

fprintf('\nSecond regression parameters (modeling b0 and b1 as functions of log(Age)):\n');
fprintf('Parameter\tCement Case\t\tBinder Case\n');
fprintf('b0 model\t(%.4f, %.4f)\t\t(%.4f, %.4f)\n', cement_b0_params(1), cement_b0_params(2), binder_b0_params(1), binder_b0_params(2));
fprintf('b1 model\t(%.4f, %.4f)\t\t(%.4f, %.4f)\n', cement_b1_params(1), cement_b1_params(2), binder_b1_params(1), binder_b1_params(2));

% Compare parameters for Age = 14 days
age_14 = 14;
log_age_14 = log(age_14);

% Calculate estimated parameters for Age = 14 days
cement_b0_est = cement_b0_params(1) + cement_b0_params(2) * log_age_14;
cement_b1_est = cement_b1_params(1) + cement_b1_params(2) * log_age_14;
binder_b0_est = binder_b0_params(1) + binder_b0_params(2) * log_age_14;
binder_b1_est = binder_b1_params(1) + binder_b1_params(2) * log_age_14;

% Find original parameters for Age = 14 days
age_14_idx = find(unique_train_ages == age_14);
if ~isempty(age_14_idx)
    cement_b0_orig = cement_b0(age_14_idx);
    cement_b1_orig = cement_b1(age_14_idx);
    binder_b0_orig = binder_b0(age_14_idx);
    binder_b1_orig = binder_b1(age_14_idx);
    
    fprintf('\nParameters for Age = 14 days:\n');
    fprintf('Parameter\tCement Case\t\t\tBinder Case\n');
    fprintf('\t\tOriginal\tEstimated\tOriginal\tEstimated\n');
    fprintf('b0\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', cement_b0_orig, cement_b0_est, binder_b0_orig, binder_b0_est);
    fprintf('b1\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', cement_b1_orig, cement_b1_est, binder_b1_orig, binder_b1_est);
    
    % Get data for Age = 14 days
    data_14_idx = (data.Age == age_14);
    x_cem_14 = data.wc_cem(data_14_idx);
    x_binder_14 = data.wc_binder(data_14_idx);
    y_ln_14 = data.Comp_str_ln(data_14_idx);
    
    % Plot for cement case (Age = 14 days)
    figure;
    subplot(1, 2, 1);
    scatter(x_cem_14, y_ln_14, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    
    x_range = linspace(min(x_cem_14), max(x_cem_14), 100)';
    y_orig = cement_b0_orig + cement_b1_orig * x_range;
    y_est = cement_b0_est + cement_b1_est * x_range;
    
    plot(x_range, y_orig, 'r-', 'LineWidth', 2);
    plot(x_range, y_est, 'g--', 'LineWidth', 2);
    
    xlabel('Water-to-Cement Ratio');
    ylabel('log(Compressive Strength)');
    title('Cement Case: Age = 14 days');
    legend('Data', 'Original Model', 'Estimated Model');
    grid on;
    
    % Plot for binder case (Age = 14 days)
    subplot(1, 2, 2);
    scatter(x_binder_14, y_ln_14, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    
    x_range = linspace(min(x_binder_14), max(x_binder_14), 100)';
    y_orig = binder_b0_orig + binder_b1_orig * x_range;
    y_est = binder_b0_est + binder_b1_est * x_range;
    
    plot(x_range, y_orig, 'r-', 'LineWidth', 2);
    plot(x_range, y_est, 'g--', 'LineWidth', 2);
    
    xlabel('Water-to-Binder Ratio');
    ylabel('log(Compressive Strength)');
    title('Binder Case: Age = 14 days');
    legend('Data', 'Original Model', 'Estimated Model');
    grid on;
else
    fprintf('\nAge = 14 days not found in training data\n');
end

%% Step 5: Assess the full regression model
% Functions for parameter prediction and strength prediction
predict_params = @(age, b0_params, b1_params) deal(b0_params(1) + b0_params(2) * log(age), b1_params(1) + b1_params(2) * log(age));
predict_log_strength = @(x, b0, b1) b0 + b1 * x;

% Initialize prediction arrays
train_cement_pred_ln = zeros(size(train_Comp_str_ln));
train_binder_pred_ln = zeros(size(train_Comp_str_ln));
test_cement_pred_ln = zeros(size(test_Comp_str_ln));
test_binder_pred_ln = zeros(size(test_Comp_str_ln));

% Generate predictions for training data
for i = 1:height(train_data)
    age = train_Age(i);
    [b0_cem, b1_cem] = predict_params(age, cement_b0_params, cement_b1_params);
    [b0_bind, b1_bind] = predict_params(age, binder_b0_params, binder_b1_params);
    
    train_cement_pred_ln(i) = predict_log_strength(train_wc_cem(i), b0_cem, b1_cem);
    train_binder_pred_ln(i) = predict_log_strength(train_wc_binder(i), b0_bind, b1_bind);
end

% Generate predictions for testing data
for i = 1:height(test_data)
    age = test_Age(i);
    [b0_cem, b1_cem] = predict_params(age, cement_b0_params, cement_b1_params);
    [b0_bind, b1_bind] = predict_params(age, binder_b0_params, binder_b1_params);
    
    test_cement_pred_ln(i) = predict_log_strength(test_wc_cem(i), b0_cem, b1_cem);
    test_binder_pred_ln(i) = predict_log_strength(test_wc_binder(i), b0_bind, b1_bind);
end

% Convert log predictions back to original scale
train_cement_pred = exp(train_cement_pred_ln);
train_binder_pred = exp(train_binder_pred_ln);
test_cement_pred = exp(test_cement_pred_ln);
test_binder_pred = exp(test_binder_pred_ln);

% Calculate residuals (in original scale)
train_cement_residuals = data.Comp_strength(train_idx) - train_cement_pred;
train_binder_residuals = data.Comp_strength(train_idx) - train_binder_pred;
test_cement_residuals = data.Comp_strength(test_idx) - test_cement_pred;
test_binder_residuals = data.Comp_strength(test_idx) - test_binder_pred;

% Plot residual densities
figure;
subplot(1, 2, 1);
hold on;
[f_train, xi_train] = ksdensity(train_cement_residuals);
[f_test, xi_test] = ksdensity(test_cement_residuals);
plot(xi_train, f_train, 'b-', 'LineWidth', 2);
plot(xi_test, f_test, 'r--', 'LineWidth', 2);
xlabel('Residuals (MPa)');
ylabel('Density');
title('Cement Case: Residual Distributions');
legend('Training', 'Testing');
grid on;

subplot(1, 2, 2);
hold on;
[f_train, xi_train] = ksdensity(train_binder_residuals);
[f_test, xi_test] = ksdensity(test_binder_residuals);
plot(xi_train, f_train, 'b-', 'LineWidth', 2);
plot(xi_test, f_test, 'r--', 'LineWidth', 2);
xlabel('Residuals (MPa)');
ylabel('Density');
title('Binder Case: Residual Distributions');
legend('Training', 'Testing');
grid on;

% Calculate R² values
calc_r2 = @(y_true, y_pred) 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2);

% Transformed data R²
r2_train_cement_transformed = calc_r2(train_Comp_str_ln, train_cement_pred_ln);
r2_train_binder_transformed = calc_r2(train_Comp_str_ln, train_binder_pred_ln);
r2_test_cement_transformed = calc_r2(test_Comp_str_ln, test_cement_pred_ln);
r2_test_binder_transformed = calc_r2(test_Comp_str_ln, test_binder_pred_ln);

% Raw data R²
r2_train_cement_raw = calc_r2(data.Comp_strength(train_idx), train_cement_pred);
r2_train_binder_raw = calc_r2(data.Comp_strength(train_idx), train_binder_pred);
r2_test_cement_raw = calc_r2(data.Comp_strength(test_idx), test_cement_pred);
r2_test_binder_raw = calc_r2(data.Comp_strength(test_idx), test_binder_pred);

% Display R² results
fprintf('\nR² Results Table:\n');
fprintf('%-20s %-30s %-30s\n', 'R² calculation', 'Cement case', 'Binder case');
fprintf('%-20s %-15s %-15s %-15s %-15s\n', '', 'Transformed', 'Raw data', 'Transformed', 'Raw data');
fprintf('%-20s %-15.4f %-15.4f %-15.4f %-15.4f\n', 'Training data', r2_train_cement_transformed, r2_train_cement_raw, r2_train_binder_transformed, r2_train_binder_raw);
fprintf('%-20s %-15.4f %-15.4f %-15.4f %-15.4f\n', 'Testing data', r2_test_cement_transformed, r2_test_cement_raw, r2_test_binder_transformed, r2_test_binder_raw);
