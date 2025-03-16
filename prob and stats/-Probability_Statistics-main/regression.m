% Step 1: Load and analyze data
data = readtable('Concrete_Data.csv');

%Cement Case, Moderate performance, explaining about 67% of variance in transformed space
%Lower performance in raw space suggests non-linear relationships remain
%Cement Case
%Good performance, explaining about 78% of variance
%Similar performance in both spaces suggests better model stability

%Cement Case:
 %compare the predicted data with the actual data , x a- y b0b1,
%Negative R² values (-0.2724 and -0.8717) indicate serious problems
%The model performs worse than a horizontalline (mean value)
%Suggests significant overfitting to traininag data
%
%The negative R² values in the Cement Case testing data are particularly concerning
%Large gap between training and testing performance suggests overfitting
%The model struggles to generalize to new age values


% Find unique ages and count samples for each age

ages = unique(data.Age);
numSamples = histc(data.Age, ages);

% Select ages with 50+ samples for training
trainingAges = ages(numSamples >= 50);

% Split data into training and testing sets
trainingIdx = ismember(data.Age, trainingAges);
trainData = data(trainingIdx, :);
testData = data(~trainingIdx, :);

% Step 2: Transform the data
Comp_str_ln = log(data.Comp_strength);

% Calculate water-to-cement and water-to-binder ratios for training data
wc_cem = trainData.Water ./ trainData.Cement;
wc_binder = trainData.Water ./ sum([trainData.Cement trainData.Slag trainData.Ash],2);

% Get log-transformed strength for training data
Comp_str_ln = log(trainData.Comp_strength);

% Step 3: Age-specific regressions
uniqueTrainAges = unique(trainData.Age);
n_ages = length(uniqueTrainAges);

% Initialize arrays to store regression parameters
b0_cem = zeros(n_ages, 1);
b1_cem = zeros(n_ages, 1);
b0_binder = zeros(n_ages, 1);
b1_binder = zeros(n_ages, 1);

% Perform regressions for each age
for i = 1:n_ages
    current_age = uniqueTrainAges(i);
    idx_age = (trainData.Age == current_age);
    
    % Case 1: Water-cement ratio regression
    mdl_cem_age = fitlm(wc_cem(idx_age), Comp_str_ln(idx_age));
    b0_cem(i) = mdl_cem_age.Coefficients.Estimate(1);
    b1_cem(i) = mdl_cem_age.Coefficients.Estimate(2);
    
    % Case 2: Water-binder ratio regression
    mdl_binder_age = fitlm(wc_binder(idx_age), Comp_str_ln(idx_age));
    b0_binder(i) = mdl_binder_age.Coefficients.Estimate(1);
    b1_binder(i) = mdl_binder_age.Coefficients.Estimate(2);
end

% Create plots for Step 3
figure('Position', [100 100 1200 400])

% Plot b0 vs ln(Age)
subplot(2,2,1)
plot(log(uniqueTrainAges), b0_cem, 'o-', 'LineWidth', 2)
xlabel('ln(Age)')
ylabel('b0 (Water-Cement)')
title('Intercept vs ln(Age) for Water-Cement Ratio')
grid on

subplot(2,2,2)
plot(log(uniqueTrainAges), b0_binder, 'o-', 'LineWidth', 2)
xlabel('ln(Age)')
ylabel('b0 (Water-Binder)')
title('Intercept vs ln(Age) for Water-Binder Ratio')
grid on

% Plot b1 vs ln(Age)
subplot(2,2,3)
plot(log(uniqueTrainAges), b1_cem, 'o-', 'LineWidth', 2)
xlabel('ln(Age)')
ylabel('b1 (Water-Cement)')
title('Slope vs ln(Age) for Water-Cement Ratio')
grid on

subplot(2,2,4)
plot(log(uniqueTrainAges), b1_binder, 'o-', 'LineWidth', 2)
xlabel('ln(Age)')
ylabel('b1 (Water-Binder)')
title('Slope vs ln(Age) for Water-Binder Ratio')
grid on

% Step 4: Second regressions
ln_age = log(uniqueTrainAges);

% Compute linear regressions between log(Age) and parameters
% For water-cement ratio
mdl_b0_cem = fitlm(ln_age, b0_cem);      % log(Age) vs b0
mdl_b1_cem = fitlm(ln_age, b1_cem);      % log(Age) vs b1

% For water-binder ratio
mdl_b0_binder = fitlm(ln_age, b0_binder); % log(Age) vs b0
mdl_b1_binder = fitlm(ln_age, b1_binder); % log(Age) vs b1

% First figure: Show the parameter regressions
figure('Position', [100 100 1200 800])

% b0 regressions
subplot(2,2,1)
scatter(ln_age, b0_cem, 'o')
hold on
plot(ln_age, predict(mdl_b0_cem, ln_age), 'r-', 'LineWidth', 2)
xlabel('ln(Age)')
ylabel('b0 (Water-Cement)')
title('b0 vs ln(Age) - Water-Cement')
grid on

subplot(2,2,2)
scatter(ln_age, b0_binder, 'o')
hold on
plot(ln_age, predict(mdl_b0_binder, ln_age), 'r-', 'LineWidth', 2)
xlabel('ln(Age)')
ylabel('b0 (Water-Binder)')
title('b0 vs ln(Age) - Water-Binder')
grid on

% b1 regressions
subplot(2,2,3)
scatter(ln_age, b1_cem, 'o')
hold on
plot(ln_age, predict(mdl_b1_cem, ln_age), 'r-', 'LineWidth', 2)
xlabel('ln(Age)')
ylabel('b1 (Water-Cement)')
title('b1 vs ln(Age) - Water-Cement')
grid on

subplot(2,2,4)
scatter(ln_age, b1_binder, 'o')
hold on
plot(ln_age, predict(mdl_b1_binder, ln_age), 'r-', 'LineWidth', 2)
xlabel('ln(Age)')
ylabel('b1 (Water-Binder)')
title('b1 vs ln(Age) - Water-Binder')
grid on

% Second figure: 14-day comparisons
% Get data for Age = 14 days
idx_14 = (trainData.Age == 14);
wc_cem_14 = wc_cem(idx_14);
wc_binder_14 = wc_binder(idx_14);
strength_ln_14 = Comp_str_ln(idx_14);

% Get estimated parameters for Age = 14 days
ln_age_14 = log(14);
b0_cem_est = predict(mdl_b0_cem, ln_age_14);
b1_cem_est = predict(mdl_b1_cem, ln_age_14);
b0_binder_est = predict(mdl_b0_binder, ln_age_14);
b1_binder_est = predict(mdl_b1_binder, ln_age_14);

% Create fine grid for plotting regression lines
wc_fine = linspace(min([wc_cem_14; wc_binder_14]), max([wc_cem_14; wc_binder_14]), 100)';

% Calculate regression lines for 14-day data
strength_cem_actual = b0_cem(uniqueTrainAges == 14) + b1_cem(uniqueTrainAges == 14) * wc_fine;
strength_cem_est = b0_cem_est + b1_cem_est * wc_fine;
strength_binder_actual = b0_binder(uniqueTrainAges == 14) + b1_binder(uniqueTrainAges == 14) * wc_fine;
strength_binder_est = b0_binder_est + b1_binder_est * wc_fine;

% Create plots for 14-day comparisons
figure('Position', [100 100 1200 400])

% Water-cement ratio plot
subplot(1,2,1)
scatter(wc_cem_14, strength_ln_14, 'o')
hold on
plot(wc_fine, strength_cem_actual, 'r-', 'LineWidth', 2, 'DisplayName', 'Actual Parameters')
plot(wc_fine, strength_cem_est, 'b--', 'LineWidth', 2, 'DisplayName', 'Estimated Parameters')
xlabel('Water-Cement Ratio')
ylabel('ln(Strength)')
title('14-Day Strength vs Water-Cement Ratio')
legend('Data', 'Actual Parameters', 'Estimated Parameters')
grid on

% Water-binder ratio plot
subplot(1,2,2)
scatter(wc_binder_14, strength_ln_14, 'o')
hold on
plot(wc_fine, strength_binder_actual, 'r-', 'LineWidth', 2, 'DisplayName', 'Actual Parameters')
plot(wc_fine, strength_binder_est, 'b--', 'LineWidth', 2, 'DisplayName', 'Estimated Parameters')
xlabel('Water-Binder Ratio')
ylabel('ln(Strength)')
title('14-Day Strength vs Water-Binder Ratio')
legend('Data', 'Actual Parameters', 'Estimated Parameters')
grid on

% Step 5: Assessing the full regression
% Use existing models from previous steps - no new implementations

% Calculate predicted values for both training and test data
ln_age_train = log(trainData.Age);
ln_age_test = log(testData.Age);

% Water-cement predictions
wc_cem_test = testData.Water ./ testData.Cement;
wc_binder_test = testData.Water ./ sum([testData.Cement testData.Slag testData.Ash], 2);

% Calculate predicted ln(strength) for training data
b0_cem_train = predict(mdl_b0_cem, ln_age_train);
b1_cem_train = predict(mdl_b1_cem, ln_age_train);
pred_ln_str_cem_train = b0_cem_train + b1_cem_train .* wc_cem;

b0_binder_train = predict(mdl_b0_binder, ln_age_train);
b1_binder_train = predict(mdl_b1_binder, ln_age_train);
pred_ln_str_binder_train = b0_binder_train + b1_binder_train .* wc_binder;

% Calculate predicted ln(strength) for test data
b0_cem_test = predict(mdl_b0_cem, ln_age_test);
b1_cem_test = predict(mdl_b1_cem, ln_age_test);
pred_ln_str_cem_test = b0_cem_test + b1_cem_test .* wc_cem_test;

b0_binder_test = predict(mdl_b0_binder, ln_age_test);
b1_binder_test = predict(mdl_b1_binder, ln_age_test);
pred_ln_str_binder_test = b0_binder_test + b1_binder_test .* wc_binder_test;

% Convert predictions to original scale
pred_str_cem_train = exp(pred_ln_str_cem_train);
pred_str_cem_test = exp(pred_ln_str_cem_test);
pred_str_binder_train = exp(pred_ln_str_binder_train);
pred_str_binder_test = exp(pred_ln_str_binder_test);

% Calculate residuals in original scale
res_cem_train = trainData.Comp_strength - pred_str_cem_train;
res_cem_test = testData.Comp_strength - pred_str_cem_test;
res_binder_train = trainData.Comp_strength - pred_str_binder_train;
res_binder_test = testData.Comp_strength - pred_str_binder_test;

% Plot residual densities
figure('Position', [100 100 1200 400])

% Water-cement residuals
subplot(1,2,1)
histogram(res_cem_train, 'Normalization', 'pdf', 'DisplayName', 'Training')
hold on
histogram(res_cem_test, 'Normalization', 'pdf', 'DisplayName', 'Testing')
xlabel('Residuals (MPa)')
ylabel('Density')
title('Residual Distribution - Water-Cement Model')
legend
grid on

% Water-binder residuals
subplot(1,2,2)
histogram(res_binder_train, 'Normalization', 'pdf', 'DisplayName', 'Training')
hold on
histogram(res_binder_test, 'Normalization', 'pdf', 'DisplayName', 'Testing')
xlabel('Residuals (MPa)')
ylabel('Density')
title('Residual Distribution - Water-Binder Model')
legend
grid on

% Calculate R² values
% For transformed data
R2_cem_trans_train = 1 - sum((log(trainData.Comp_strength) - pred_ln_str_cem_train).^2) / ...
    sum((log(trainData.Comp_strength) - mean(log(trainData.Comp_strength))).^2);
R2_cem_trans_test = 1 - sum((log(testData.Comp_strength) - pred_ln_str_cem_test).^2) / ...
    sum((log(testData.Comp_strength) - mean(log(testData.Comp_strength))).^2);
R2_binder_trans_train = 1 - sum((log(trainData.Comp_strength) - pred_ln_str_binder_train).^2) / ...
    sum((log(trainData.Comp_strength) - mean(log(trainData.Comp_strength))).^2);
R2_binder_trans_test = 1 - sum((log(testData.Comp_strength) - pred_ln_str_binder_test).^2) / ...
    sum((log(testData.Comp_strength) - mean(log(testData.Comp_strength))).^2);

% For raw data
%estimated data  
R2_cem_raw_train = 1 - sum((trainData.Comp_strength - pred_str_cem_train).^2) / ...
    sum((trainData.Comp_strength - mean(trainData.Comp_strength)).^2);
R2_cem_raw_test = 1 - sum((testData.Comp_strength - pred_str_cem_test).^2) / ...
    sum((testData.Comp_strength - mean(testData.Comp_strength)).^2);
R2_binder_raw_train = 1 - sum((trainData.Comp_strength - pred_str_binder_train).^2) / ...
    sum((trainData.Comp_strength - mean(trainData.Comp_strength)).^2);
R2_binder_raw_test = 1 - sum((testData.Comp_strength - pred_str_binder_test).^2) / ...
    sum((testData.Comp_strength - mean(testData.Comp_strength)).^2);

% Step 3 Plotting
figure('Name', 'Parameter b0 vs log(Age)');
subplot(1,2,1);
plot(ln_age, b0_cem, 'o', 'DisplayName', 'Cement');
xlabel('log(Age)');
ylabel('b0');
title('Water-Cement Ratio');
legend;
grid on;

subplot(1,2,2);
plot(ln_age, b0_binder, 'o', 'DisplayName', 'Binder');
xlabel('log(Age)');
ylabel('b0');
title('Water-Binder Ratio');
legend;
grid on;

figure('Name', 'Parameter b1 vs log(Age)');
subplot(1,2,1);
plot(ln_age, b1_cem, 'o', 'DisplayName', 'Cement');
xlabel('log(Age)');
ylabel('b1');
title('Water-Cement Ratio');
legend;
grid on;

subplot(1,2,2);
plot(ln_age, b1_binder, 'o', 'DisplayName', 'Binder');
xlabel('log(Age)');
ylabel('b1');
title('Water-Binder Ratio');
legend;
grid on;

% Step 4 Plotting - 14-day comparison
figure('Name', '14-day Comparison');
subplot(1,2,1);
scatter(wc_cem_14, strength_ln_14, 'DisplayName', 'Data');
hold on;
x_range = linspace(min(wc_cem_14), max(wc_cem_14), 100);
y_calc = b0_cem(uniqueTrainAges == 14) + b1_cem(uniqueTrainAges == 14) * x_range;
y_est = b0_cem_est + b1_cem_est * x_range;
plot(x_range, y_calc, 'DisplayName', 'Calculated Parameters');
plot(x_range, y_est, '--', 'DisplayName', 'Estimated Parameters');
xlabel('Water-Cement Ratio');
ylabel('ln(Strength)');
title('Water-Cement Model (14 days)');
legend;
grid on;
hold off;

subplot(1,2,2);
scatter(wc_binder_14, strength_ln_14, 'DisplayName', 'Data');
hold on;
x_range = linspace(min(wc_binder_14), max(wc_binder_14), 100);
y_calc = b0_binder(uniqueTrainAges == 14) + b1_binder(uniqueTrainAges == 14) * x_range;
y_est = b0_binder_est + b1_binder_est * x_range;
plot(x_range, y_calc, 'DisplayName', 'Calculated Parameters');
plot(x_range, y_est, '--', 'DisplayName', 'Estimated Parameters');
xlabel('Water-Binder Ratio');
ylabel('ln(Strength)');
title('Water-Binder Model (14 days)');
legend;
grid on;
hold off;

% Step 5 Plotting - Residual densities
figure('Name', 'Residual Densities');
subplot(1,2,1);
histogram(trainData.Comp_strength - pred_str_cem_train, 'Normalization', 'pdf', ...
    'DisplayName', 'Training', 'FaceAlpha', 0.5);
hold on;
histogram(testData.Comp_strength - pred_str_cem_test, 'Normalization', 'pdf', ...
    'DisplayName', 'Testing', 'FaceAlpha', 0.5);
xlabel('Residuals');
ylabel('Density');
title('Water-Cement Model Residuals');
legend;
grid on;
hold off;

subplot(1,2,2);
histogram(trainData.Comp_strength - pred_str_binder_train, 'Normalization', 'pdf', ...
    'DisplayName', 'Training', 'FaceAlpha', 0.5);
hold on;
histogram(testData.Comp_strength - pred_str_binder_test, 'Normalization', 'pdf', ...
    'DisplayName', 'Testing', 'FaceAlpha', 0.5);
xlabel('Residuals');
ylabel('Density');
title('Water-Binder Model Residuals');
legend;
grid on;
hold off;

%log-get rid off to find the original data for r^2, and use estimate data 
% Display R² results
fprintf('\nR² Results:\n');
fprintf('                    Cement Case                 Binder Case\n');
fprintf('                Transformed    Raw         Transformed    Raw\n');
fprintf('Training data  %10.4f  %10.4f    %10.4f  %10.4f\n', ...
    R2_cem_trans_train, R2_cem_raw_train, R2_binder_trans_train, R2_binder_raw_train);
fprintf('Testing data   %10.4f  %10.4f    %10.4f  %10.4f\n', ...
    R2_cem_trans_test, R2_cem_raw_test, R2_binder_trans_test, R2_binder_raw_test); 