%% Step 1: Data Loading and Splitting
data = readtable('Concrete_Data.csv');
unique_ages = unique(data.Age);
train_idx = [];
test_idx = [];

for age = unique_ages'
    age_data = find(data.Age == age);
    n = length(age_data);
    
    if n > 50
        train_idx = [train_idx; age_data(1:50)];
        test_idx = [test_idx; age_data(51:end)];
    else
        train_idx = [train_idx; age_data];
    end
end

% Create training and testing tables
train_data = data(train_idx,:);
test_data = data(test_idx,:);

% Report counts
fprintf('Training samples: %d\n', height(train_data));
fprintf('Testing samples: %d\n', height(test_data));

%% Step 2: Data Transformation
% Create transformed variables
train_data.Comp_str_ln = log(train_data.Comp_strength);
test_data.Comp_str_ln = log(test_data.Comp_strength);

% Calculate water-cement and water-binder ratios
train_data.wc_cem = train_data.Water ./ train_data.Cement;
train_data.wc_binder = train_data.Water ./ sum([train_data.Cement, ...
                       train_data.Slag, train_data.Ash], 2);

test_data.wc_cem = test_data.Water ./ test_data.Cement;
test_data.wc_binder = test_data.Water ./ sum([test_data.Cement, ...
                      test_data.Slag, test_data.Ash], 2);

%% Step 3: First Regression Analysis
% Initialize storage for coefficients
ages_train = unique(train_data.Age);
coefficients_cem = zeros(length(ages_train), 2);
coefficients_binder = zeros(length(ages_train), 2);

% Perform regressions for each age
for i = 1:length(ages_train)
    age = ages_train(i);
    idx = (train_data.Age == age);
    
    % Cement case
    X = [ones(sum(idx),1) train_data.wc_cem(idx)];
    b = X \ train_data.Comp_str_ln(idx);
    coefficients_cem(i,:) = b';
    
    % Binder case
    X = [ones(sum(idx),1) train_data.wc_binder(idx)];
    b = X \ train_data.Comp_str_ln(idx);
    coefficients_binder(i,:) = b';
end

% Plot coefficients vs log(age)
figure;
subplot(2,2,1);
plot(log(ages_train), coefficients_cem(:,1), 'o');
title('Cement Case: b₀ vs log(Age)');
xlabel('log(Age)'); ylabel('b₀');

subplot(2,2,2);
plot(log(ages_train), coefficients_cem(:,2), 'o');
title('Cement Case: b₁ vs log(Age)');
xlabel('log(Age)'); ylabel('b₁');

subplot(2,2,3);
plot(log(ages_train), coefficients_binder(:,1), 'o');
title('Binder Case: b₀ vs log(Age)');
xlabel('log(Age)'); ylabel('b₀');

subplot(2,2,4);
plot(log(ages_train), coefficients_binder(:,2), 'o');
title('Binder Case: b₁ vs log(Age)');
xlabel('log(Age)'); ylabel('b₁');

%% Step 4: Second Regression Analysis
% Fit relationships between coefficients and log(age)
log_ages = log(ages_train);

% Cement case
X = [ones(size(log_ages)) log_ages];
b0_cem_coeff = X \ coefficients_cem(:,1);
b1_cem_coeff = X \ coefficients_cem(:,2);

% Binder case
b0_bind_coeff = X \ coefficients_binder(:,1);
b1_bind_coeff = X \ coefficients_binder(:,2);

% Age=14 analysis
age14_idx = (train_data.Age == 14);
log_age14 = log(14);

% Calculate predicted coefficients
b0_cem_pred = [1 log_age14] * b0_cem_coeff;
b1_cem_pred = [1 log_age14] * b1_cem_coeff;
b0_bind_pred = [1 log_age14] * b0_bind_coeff;
b1_bind_pred = [1 log_age14] * b1_bind_coeff;

% Plot comparisons
figure;
subplot(1,2,1);
plot(train_data.wc_cem(age14_idx), train_data.Comp_str_ln(age14_idx), 'o');
hold on;
x_vals = linspace(min(train_data.wc_cem(age14_idx)), max(train_data.wc_cem(age14_idx)), 100);
plot(x_vals, coefficients_cem(ages_train==14,1) + coefficients_cem(ages_train==14,2)*x_vals, 'r');
plot(x_vals, b0_cem_pred + b1_cem_pred*x_vals, 'g--');
title('Cement Case: Age=14 Comparison');
legend('Data', 'Original Model', 'Estimated Model');

subplot(1,2,2);
plot(train_data.wc_binder(age14_idx), train_data.Comp_str_ln(age14_idx), 'o');
hold on;
x_vals = linspace(min(train_data.wc_binder(age14_idx)), max(train_data.wc_binder(age14_idx)), 100);
plot(x_vals, coefficients_binder(ages_train==14,1) + coefficients_binder(ages_train==14,2)*x_vals, 'r');
plot(x_vals, b0_bind_pred + b1_bind_pred*x_vals, 'g--');
title('Binder Case: Age=14 Comparison');
legend('Data', 'Original Model', 'Estimated Model');

%% Step 5: Model Assessment
% Prediction function
predict_strength = @(wc, age, b0_coeff, b1_coeff) ...
    exp(b0_coeff(1) + b0_coeff(2)*log(age) + ...
    (b1_coeff(1) + b1_coeff(2)*log(age)) .* wc);

% Calculate predictions and residuals
% Cement case
train_pred_cem = predict_strength(train_data.wc_cem, train_data.Age, b0_cem_coeff, b1_cem_coeff);
test_pred_cem = predict_strength(test_data.wc_cem, test_data.Age, b0_cem_coeff, b1_cem_coeff);

% Binder case
train_pred_bind = predict_strength(train_data.wc_binder, train_data.Age, b0_bind_coeff, b1_bind_coeff);
test_pred_bind = predict_strength(test_data.wc_binder, test_data.Age, b0_bind_coeff, b1_bind_coeff);

% Residual calculations
train_res_cem = train_data.Comp_strength - train_pred_cem;
test_res_cem = test_data.Comp_strength - test_pred_cem;
train_res_bind = train_data.Comp_strength - train_pred_bind;
test_res_bind = test_data.Comp_strength - test_pred_bind;

% Residual density plots
figure;
subplot(1,2,1);
histfit(train_res_cem, 50, 'kernel');
hold on;
histfit(test_res_cem, 50, 'kernel');
title('Cement Case Residuals');
legend('Train', '', 'Test', '');

subplot(1,2,2);
histfit(train_res_bind, 50, 'kernel');
hold on;
histfit(test_res_bind, 50, 'kernel');
title('Binder Case Residuals');
legend('Train', '', 'Test', '');

% R² calculations
ss_total_train = sum((train_data.Comp_strength - mean(train_data.Comp_strength)).^2);
ss_total_test = sum((test_data.Comp_strength - mean(test_data.Comp_strength)).^2);

r2_cem_train = 1 - sum(train_res_cem.^2)/ss_total_train;
r2_cem_test = 1 - sum(test_res_cem.^2)/ss_total_test;
r2_bind_train = 1 - sum(train_res_bind.^2)/ss_total_train;
r2_bind_test = 1 - sum(test_res_bind.^2)/ss_total_test;

% Display results
fprintf('R² Values:\n');
fprintf('%-20s %-10s %-10s\n', '', 'Cement', 'Binder');
fprintf('%-20s %-10.3f %-10.3f\n', 'Training (Raw):', r2_cem_train, r2_bind_train);
fprintf('%-20s %-10.3f %-10.3f\n', 'Testing (Raw):', r2_cem_test, r2_bind_test);
