%% Step 1: Load and split data with improved stratification
% Load the data
data = readtable("Concrete_Data.csv");

% Find unique ages and do stratified sampling
unique_ages = unique(data.Age);
num_unique_ages = length(unique_ages);
fprintf('Unique age values: %d\n', num_unique_ages);

% Improved splitting - stratified by age
rng(42); % For reproducibility
train_idx = false(height(data), 1);
test_idx = false(height(data), 1);

for i = 1:num_unique_ages
    age = unique_ages(i);
    age_idx = (data.Age == age);
    num_samples = sum(age_idx);
    
    if num_samples >= 10
        % Stratified sampling within age group (80% train, 20% test)
        cv = cvpartition(sum(age_idx), 'HoldOut', 0.2);
        age_rows = find(age_idx);
        train_idx(age_rows(cv.training)) = true;
        test_idx(age_rows(cv.test)) = true;
    else
        % Small groups entirely to training
        train_idx = train_idx | age_idx;
    end
    
    fprintf('Age %d days: %d samples (%d train, %d test)\n', age, num_samples, ...
        sum(train_idx & age_idx), sum(test_idx & age_idx));
end

fprintf('Training set: %d samples, Testing set: %d samples\n\n', ...
    sum(train_idx), sum(test_idx));

%% Step 2: Transform the data with polynomial features
% Base transformations
Comp_str_ln = log(data.Comp_strength);
wc_cem = data.Water ./ data.Cement;
wc_binder = data.Water ./ sum([data.Cement, data.Slag, data.Ash], 2);
log_age = log(data.Age);

% Create polynomial and interaction terms
wc_cem_sq = wc_cem.^2;
wc_binder_sq = wc_binder.^2;
log_age_sq = log_age.^2;
wc_cem_log_age = wc_cem .* log_age;
wc_binder_log_age = wc_binder .* log_age;
wc_cem_log_age_sq = wc_cem .* log_age_sq;
wc_binder_log_age_sq = wc_binder .* log_age_sq;

% Add to data table
data.Comp_str_ln = Comp_str_ln;
data.wc_cem = wc_cem;
data.wc_binder = wc_binder;
data.log_age = log_age;
data.wc_cem_sq = wc_cem_sq;
data.wc_binder_sq = wc_binder_sq;
data.log_age_sq = log_age_sq;
data.wc_cem_log_age = wc_cem_log_age;
data.wc_binder_log_age = wc_binder_log_age;
data.wc_cem_log_age_sq = wc_cem_log_age_sq;
data.wc_binder_log_age_sq = wc_binder_log_age_sq;

%% Step 3: Create polynomial regression models
% Prepare feature matrices for cement case
X_cem = [ones(height(data),1), wc_cem, log_age, wc_cem_sq, log_age_sq, ...
         wc_cem_log_age, wc_cem_log_age_sq];

% Prepare feature matrices for binder case
X_binder = [ones(height(data),1), wc_binder, log_age, wc_binder_sq, log_age_sq, ...
            wc_binder_log_age, wc_binder_log_age_sq];

% Target variable
Y = Comp_str_ln;

% Split features and target into training and testing sets
X_cem_train = X_cem(train_idx, :);
X_cem_test = X_cem(test_idx, :);
X_binder_train = X_binder(train_idx, :);
X_binder_test = X_binder(test_idx, :);
Y_train = Y(train_idx);
Y_test = Y(test_idx);

% Train models with regularization to prevent overfitting
lambda = 0.01;  % Regularization parameter
I = eye(size(X_cem_train, 2));
I(1,1) = 0;  % Don't regularize intercept

% Ridge regression models
cem_coeffs = (X_cem_train' * X_cem_train + lambda * I) \ (X_cem_train' * Y_train);
binder_coeffs = (X_binder_train' * X_binder_train + lambda * I) \ (X_binder_train' * Y_train);

fprintf('Polynomial model coefficients:\n');
fprintf('Cement model: Intercept=%.4f, wc_cem=%.4f, log_age=%.4f, wc_cem²=%.4f, log_age²=%.4f, wc_cem×log_age=%.4f, wc_cem×log_age²=%.4f\n', ...
    cem_coeffs(1), cem_coeffs(2), cem_coeffs(3), cem_coeffs(4), cem_coeffs(5), cem_coeffs(6), cem_coeffs(7));

fprintf('Binder model: Intercept=%.4f, wc_binder=%.4f, log_age=%.4f, wc_binder²=%.4f, log_age²=%.4f, wc_binder×log_age=%.4f, wc_binder×log_age²=%.4f\n\n', ...
    binder_coeffs(1), binder_coeffs(2), binder_coeffs(3), binder_coeffs(4), binder_coeffs(5), binder_coeffs(6), binder_coeffs(7));

%% Step 4: Cross-validation to assess model robustness
% 5-fold cross-validation
cv = cvpartition(sum(train_idx), 'KFold', 5);
cem_cv_rmse = zeros(5, 1);
binder_cv_rmse = zeros(5, 1);

for i = 1:5
    % Get train/val indices for this fold
    fold_train = cv.training(i);
    fold_val = cv.test(i);
    
    % Cement model for this fold
    X_fold_train = X_cem_train(fold_train, :);
    Y_fold_train = Y_train(fold_train);
    X_fold_val = X_cem_train(fold_val, :);
    Y_fold_val = Y_train(fold_val);
    
    % Calculate coefficients for this fold
    fold_coeffs = (X_fold_train' * X_fold_train + lambda * I) \ (X_fold_train' * Y_fold_train);
    Y_fold_pred = X_fold_val * fold_coeffs;
    cem_cv_rmse(i) = sqrt(mean((Y_fold_val - Y_fold_pred).^2));
    
    % Binder model for this fold
    X_fold_train = X_binder_train(fold_train, :);
    Y_fold_train = Y_train(fold_train);
    X_fold_val = X_binder_train(fold_val, :);
    Y_fold_val = Y_train(fold_val);
    
    fold_coeffs = (X_fold_train' * X_fold_train + lambda * I) \ (X_fold_train' * Y_fold_train);
    Y_fold_pred = X_fold_val * fold_coeffs;
    binder_cv_rmse(i) = sqrt(mean((Y_fold_val - Y_fold_pred).^2));
end

fprintf('Cross-validation results (RMSE):\n');
fprintf('Cement model: %.4f (±%.4f)\n', mean(cem_cv_rmse), std(cem_cv_rmse));
fprintf('Binder model: %.4f (±%.4f)\n\n', mean(binder_cv_rmse), std(binder_cv_rmse));

%% Step 5: Make predictions and calculate performance metrics
% Training set predictions
train_pred_cem_ln = X_cem_train * cem_coeffs;
train_pred_binder_ln = X_binder_train * binder_coeffs;

% Testing set predictions
test_pred_cem_ln = X_cem_test * cem_coeffs;
test_pred_binder_ln = X_binder_test * binder_coeffs;

% Back-transform to original scale
train_pred_cem = exp(train_pred_cem_ln);
train_pred_binder = exp(train_pred_binder_ln);
test_pred_cem = exp(test_pred_cem_ln);
test_pred_binder = exp(test_pred_binder_ln);

% Calculate R² values
calc_r2 = @(y_true, y_pred) 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2);

% R² for transformed data
r2_train_cem_ln = calc_r2(Y_train, train_pred_cem_ln);
r2_test_cem_ln = calc_r2(Y_test, test_pred_cem_ln);
r2_train_binder_ln = calc_r2(Y_train, train_pred_binder_ln);
r2_test_binder_ln = calc_r2(Y_test, test_pred_binder_ln);

% R² for raw data
r2_train_cem = calc_r2(data.Comp_strength(train_idx), train_pred_cem);
r2_test_cem = calc_r2(data.Comp_strength(test_idx), test_pred_cem);
r2_train_binder = calc_r2(data.Comp_strength(train_idx), train_pred_binder);
r2_test_binder = calc_r2(data.Comp_strength(test_idx), test_pred_binder);

% Display R² results
fprintf('R² Results with Polynomial Models:\n');
fprintf('%-20s %-30s %-30s\n', 'R² calculation', 'Cement case', 'Binder case');
fprintf('%-20s %-15s %-15s %-15s %-15s\n', '', 'Transformed', 'Raw data', 'Transformed', 'Raw data');
fprintf('%-20s %-15.4f %-15.4f %-15.4f %-15.4f\n', 'Training data', r2_train_cem_ln, r2_train_cem, r2_train_binder_ln, r2_train_binder);
fprintf('%-20s %-15.4f %-15.4f %-15.4f %-15.4f\n\n', 'Testing data', r2_test_cem_ln, r2_test_cem, r2_test_binder_ln, r2_test_binder);

%% Step 6: Visualize results
% Set larger font size (20pt) for all plots
set(0, 'DefaultAxesFontSize', 20);
set(0, 'DefaultTextFontSize', 20);

% Close any existing figures and create a new one
close all
figure('Position', [100 100 1400 700]);
set(gcf, 'Color', 'w'); % White background

% CEMENT MODEL - LEFT SUBPLOT
subplot(1,2,1);

% Training data - larger dots with transparency for density
scatter(data.Comp_strength(train_idx), train_pred_cem, 100, [0.4 0.6 0.8], 'filled', ...
    'MarkerFaceAlpha', 0.5, 'DisplayName', 'Training');
hold on;

% Testing data - larger dots with different color
scatter(data.Comp_strength(test_idx), test_pred_cem, 100, [0.9 0.4 0.3], 'filled', ...
    'MarkerFaceAlpha', 0.5, 'DisplayName', 'Testing');

% Add perfect prediction line
max_val = max([data.Comp_strength; train_pred_cem; test_pred_cem]) * 1.05;
plot([0, max_val], [0, max_val], 'k--', 'LineWidth', 2, 'DisplayName', 'Perfect Prediction');

% Add regression line for combined data
combined_actual = [data.Comp_strength(train_idx); data.Comp_strength(test_idx)];
combined_pred = [train_pred_cem; test_pred_cem];
p = polyfit(combined_actual, combined_pred, 1);
x_line = linspace(0, max_val, 100);
y_line = polyval(p, x_line);
plot(x_line, y_line, '-', 'Color', [0.2 0.2 0.8], 'LineWidth', 3, 'DisplayName', 'Regression Line');

% Create custom legend with large font
legend('Training Data', 'Testing Data', 'Perfect Prediction', 'Regression Line', ...
    'Location', 'northwest', 'FontSize', 20);

% Add info box using text() - POSITION IS ADJUSTABLE
r2_train = 0.6765; % From the data you provided
r2_test = 0.6770;  % From the data you provided
box_text = sprintf('Cement Model:\nR²_{train} = %.3f\nR²_{test} = %.3f\nRMSE = 0.294 (±0.011)', r2_train, r2_test);

% Create box with text
text(0.05*max_val, 0.85*max_val, box_text, 'FontSize', 20, 'FontWeight', 'bold', ...
    'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'k', 'LineWidth', 1.5, ...
    'Margin', 5, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');

% Enhance appearance
xlabel('Actual Compressive Strength (MPa)', 'FontSize', 25, 'FontWeight', 'bold');
ylabel('Predicted Compressive Strength (MPa)', 'FontSize', 25, 'FontWeight', 'bold');
title('Water-Cement Ratio Model', 'FontSize', 25, 'FontWeight', 'bold');
axis([0 max_val 0 max_val]);
axis square; % Make plot square
grid on;
set(gca, 'FontSize', 20, 'LineWidth', 1.5, 'GridLineStyle', ':');
box on;

% BINDER MODEL - RIGHT SUBPLOT
subplot(1,2,2);

% Training data - larger dots with transparency
scatter(data.Comp_strength(train_idx), train_pred_binder, 100, [0.4 0.6 0.8], 'filled', ...
    'MarkerFaceAlpha', 0.5, 'DisplayName', 'Training');
hold on;

% Testing data - larger dots with different color
scatter(data.Comp_strength(test_idx), test_pred_binder, 100, [0.9 0.4 0.3], 'filled', ...
    'MarkerFaceAlpha', 0.5, 'DisplayName', 'Testing');

% Add perfect prediction line
max_val = max([data.Comp_strength; train_pred_binder; test_pred_binder]) * 1.05;
plot([0, max_val], [0, max_val], 'k--', 'LineWidth', 2, 'DisplayName', 'Perfect Prediction');

% Add regression line for combined data
combined_actual = [data.Comp_strength(train_idx); data.Comp_strength(test_idx)];
combined_pred = [train_pred_binder; test_pred_binder];
p = polyfit(combined_actual, combined_pred, 1);
x_line = linspace(0, max_val, 100);
y_line = polyval(p, x_line);
plot(x_line, y_line, '-', 'Color', [0.2 0.2 0.8], 'LineWidth', 3, 'DisplayName', 'Regression Line');

% Create custom legend with large font
legend('Training Data', 'Testing Data', 'Perfect Prediction', 'Regression Line', ...
    'Location', 'northwest', 'FontSize', 20);

% Add info box using text() - POSITION IS ADJUSTABLE
r2_train = 0.7916; % From the data you provided
r2_test = 0.8032;  % From the data you provided
box_text = sprintf('Binder Model:\nR²_{train} = %.3f\nR²_{test} = %.3f\nRMSE = 0.252 (±0.020)', r2_train, r2_test);

% Create box with text
text(0.05*max_val, 0.85*max_val, box_text, 'FontSize', 20, 'FontWeight', 'bold', ...
    'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'k', 'LineWidth', 1.5, ...
    'Margin', 5, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');

% Enhance appearance
xlabel('Actual Compressive Strength (MPa)', 'FontSize', 25, 'FontWeight', 'bold');
ylabel('Predicted Compressive Strength (MPa)', 'FontSize', 25, 'FontWeight', 'bold');
title('Water-Binder Ratio Model', 'FontSize', 25, 'FontWeight', 'bold');
axis([0 max_val 0 max_val]);
axis square; % Make plot square
grid on;
set(gca, 'FontSize', 20, 'LineWidth', 1.5, 'GridLineStyle', ':');
box on;

% Add overall title
sgtitle('Concrete Strength Prediction Models: Actual vs. Predicted', 'FontSize', 28, 'FontWeight', 'bold');

% Save the figure
full_path = 'J:\prob and stats\-Probability_Statistics-main\Probability\Problem1_regression\13.03';
set(gcf, 'Renderer', 'painters');
print([full_path, '\actual_vs_predicted_improved.svg'], '-dsvg', '-painters');
print([full_path, '\actual_vs_predicted_improved.png'], '-dpng', '-r400');

% Surface plots to visualize polynomial regression
figure('Position', [100 100 1200 600]);
[X1, X2] = meshgrid(linspace(min(wc_cem), max(wc_cem), 30), linspace(min(log_age), max(log_age), 30));
Z = zeros(size(X1));

% Calculate predicted values across the grid
for i = 1:size(X1, 1)
    for j = 1:size(X1, 2)
        x = [1, X1(i,j), X2(i,j), X1(i,j)^2, X2(i,j)^2, X1(i,j)*X2(i,j), X1(i,j)*X2(i,j)^2];
        Z(i,j) = x * cem_coeffs;
    end
end

subplot(1,2,1);
surf(X1, X2, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.6);
hold on;
scatter3(wc_cem(train_idx), log_age(train_idx), Y_train, 50, 'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'r', 'MarkerFaceAlpha', 0.6);
xlabel('Water-Cement Ratio', 'FontSize', 20);
ylabel('log(Age)', 'FontSize', 20);
zlabel('log(Compressive Strength)', 'FontSize', 20);
title('Water-Cement Ratio vs. Age: Polynomial Regression Surface', 'FontSize', 25, 'FontWeight', 'bold');
colormap('jet');
colorbar;
view(45, 30);

% Repeat for binder model
[X1, X2] = meshgrid(linspace(min(wc_binder), max(wc_binder), 30), linspace(min(log_age), max(log_age), 30));
Z = zeros(size(X1));

for i = 1:size(X1, 1)
    for j = 1:size(X1, 2)
        x = [1, X1(i,j), X2(i,j), X1(i,j)^2, X2(i,j)^2, X1(i,j)*X2(i,j), X1(i,j)*X2(i,j)^2];
        Z(i,j) = x * binder_coeffs;
    end
end

subplot(1,2,2);
surf(X1, X2, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.6);
hold on;
scatter3(wc_binder(train_idx), log_age(train_idx), Y_train, 50, 'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'r', 'MarkerFaceAlpha', 0.6);
xlabel('Water-Binder Ratio', 'FontSize', 20);
ylabel('log(Age)', 'FontSize', 20);
zlabel('log(Compressive Strength)', 'FontSize', 20);
title('Water-Binder Ratio vs. Age: Polynomial Regression Surface', 'FontSize', 25, 'FontWeight', 'bold');
colormap('jet');
colorbar;
view(45, 30);
 
% Calculate residuals from raw (untransformed) data
train_residuals_cem = data.Comp_strength(train_idx) - train_pred_cem;
test_residuals_cem = data.Comp_strength(test_idx) - test_pred_cem;
train_residuals_binder = data.Comp_strength(train_idx) - train_pred_binder;
test_residuals_binder = data.Comp_strength(test_idx) - test_pred_binder;

% Find the overall min and max for both models to set consistent limits
all_residuals = [train_residuals_cem; test_residuals_cem; 
                train_residuals_binder; test_residuals_binder];
x_min = min(all_residuals);
x_max = max(all_residuals);

% Add some padding to the limits
x_range = x_max - x_min;
x_min = x_min - 0.1*x_range;
x_max = x_max + 0.1*x_range;

% Create figure with two subplots (one for cement, one for binder)
figure('Position', [100 100 1400 600]);
set(gcf, 'Color', 'w'); % White background

% CEMENT CASE - SUBPLOT 1
subplot(1,2,1);

% Create histograms for density visualization
histogram(train_residuals_cem, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.8], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.7, 'DisplayName', 'Training', 'BinWidth', 2);
hold on;
histogram(test_residuals_cem, 'Normalization', 'pdf', 'FaceColor', [0.9 0.4 0.3], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.7, 'DisplayName', 'Testing', 'BinWidth', 2);

% Add KDE curves
[f_train_cem, xi_train_cem] = ksdensity(train_residuals_cem);
[f_test_cem, xi_test_cem] = ksdensity(test_residuals_cem);
plot(xi_train_cem, f_train_cem, 'Color', [0 0.4470 0.7410], 'LineWidth', 4, 'DisplayName', 'Training KDE');
plot(xi_test_cem, f_test_cem, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 4, 'DisplayName', 'Testing KDE');

% Add mean lines
xline(mean(train_residuals_cem), '--', 'Color', [0 0.4470 0.7410], 'LineWidth', 3);
xline(mean(test_residuals_cem), '--', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 3);
xline(0, 'k-', 'LineWidth', 2); % Zero reference line

% Set fixed x-limits to be the same for both plots
xlim([x_min x_max]);

% Find max density for y-limit consistency
y_max_cem = max([max(f_train_cem), max(f_test_cem)]);

% Add labels and formatting
xlabel('Residual (MPa)', 'FontSize', 25, 'FontWeight', 'bold');
ylabel('Density', 'FontSize', 25, 'FontWeight', 'bold');
title('Cement-Based Model Residual Distribution', 'FontSize', 25, 'FontWeight', 'bold');
legend('Training', 'Testing', 'Training KDE', 'Testing KDE', 'FontSize', 25, 'Location', 'best');
grid on;
set(gca, 'FontSize', 25, 'LineWidth', 1.5, 'GridLineStyle', ':');

% Add statistics text
text(0.05, 0.95, sprintf('Train: μ=%.2f, σ=%.2f', mean(train_residuals_cem), std(train_residuals_cem)), ...
    'Units', 'normalized', 'FontSize', 25, 'Color', [0 0.4470 0.7410], 'FontWeight', 'bold');
text(0.05, 0.87, sprintf('Test: μ=%.2f, σ=%.2f', mean(test_residuals_cem), std(test_residuals_cem)), ...
    'Units', 'normalized', 'FontSize', 25, 'Color', [0.8500 0.3250 0.0980], 'FontWeight', 'bold');

% BINDER CASE - SUBPLOT 2
subplot(1,2,2);

% Create histograms for density visualization
histogram(train_residuals_binder, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.8], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.7, 'DisplayName', 'Training', 'BinWidth', 2);
hold on;
histogram(test_residuals_binder, 'Normalization', 'pdf', 'FaceColor', [0.9 0.4 0.3], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.7, 'DisplayName', 'Testing', 'BinWidth', 2);

% Add KDE curves
[f_train_binder, xi_train_binder] = ksdensity(train_residuals_binder);
[f_test_binder, xi_test_binder] = ksdensity(test_residuals_binder);
plot(xi_train_binder, f_train_binder, 'Color', [0 0.4470 0.7410], 'LineWidth', 4, 'DisplayName', 'Training KDE');
plot(xi_test_binder, f_test_binder, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 4, 'DisplayName', 'Testing KDE');

% Add mean lines
xline(mean(train_residuals_binder), '--', 'Color', [0 0.4470 0.7410], 'LineWidth', 3);
xline(mean(test_residuals_binder), '--', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 3);
xline(0, 'k-', 'LineWidth', 2); % Zero reference line

% Set fixed x-limits to be the same for both plots
xlim([x_min x_max]);

% Find max density for y-limit consistency
y_max_binder = max([max(f_train_binder), max(f_test_binder)]);

% Set consistent y-limits for both plots
y_max = max([y_max_cem, y_max_binder]) * 1.1;  % Add 10% margin
subplot(1,2,1); ylim([0 y_max]);
subplot(1,2,2); ylim([0 y_max]);

% Add labels and formatting
xlabel('Residual (MPa)', 'FontSize', 25, 'FontWeight', 'bold');
ylabel('Density', 'FontSize', 25, 'FontWeight', 'bold');
title('Binder-Based Model Residual Distribution', 'FontSize', 25, 'FontWeight', 'bold');
legend('Training', 'Testing', 'Training KDE', 'Testing KDE', 'FontSize', 25, 'Location', 'best');
grid on;
set(gca, 'FontSize', 25, 'LineWidth', 1.5, 'GridLineStyle', ':');

% Add statistics text
text(0.05, 0.95, sprintf('Train: μ=%.2f, σ=%.2f', mean(train_residuals_binder), std(train_residuals_binder)), ...
    'Units', 'normalized', 'FontSize', 25, 'Color', [0 0.4470 0.7410], 'FontWeight', 'bold');
text(0.05, 0.87, sprintf('Test: μ=%.2f, σ=%.2f', mean(test_residuals_binder), std(test_residuals_binder)), ...
    'Units', 'normalized', 'FontSize', 25, 'Color', [0.8500 0.3250 0.0980], 'FontWeight', 'bold');

% Save the figure as SVG
full_path = 'J:\prob and stats\-Probability_Statistics-main\Probability\Problem1_regression\13.03';
set(gcf, 'Renderer', 'painters');
print([full_path, '\residual_densities.svg'], '-dsvg', '-painters');
 
 
