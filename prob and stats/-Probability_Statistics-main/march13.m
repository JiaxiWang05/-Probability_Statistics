%% Step 1: Data Loading and Splitting
data = readtable('Concrete_Data.csv');
unique_ages = unique(data.Age);
fprintf('Unique ages: %s\n', mat2str(unique_ages'));

% Split into training/testing
train_data = [];
test_data = [];
for age = unique_ages'
    age_data = data(data.Age == age, :);
    if height(age_data) > 50
        train_data = [train_data; age_data];
    else
        test_data = [test_data; age_data];
    end
end
fprintf('Training data size: %d\n', height(train_data));
fprintf('Testing data size: %d\n', height(test_data));

%% Step 2: Data Transformation
% Common transformations
Comp_str_ln = log(data.Comp_strength);

% Case 1: Cement only
wc_cem = data.Water ./ data.Cement;

% Case 2: Binder (Cement + Slag + Ash)
binder = sum([data.Cement data.Slag data.Ash], 2);
wc_binder = data.Water ./ binder;

%% Step 3: First Regression Stage
train_ages = unique(train_data.Age);
coefficients_cem = [];
coefficients_binder = [];

for age = train_ages'
    idx = (train_data.Age == age);
    
    % Cement case
    X_cem = wc_cem(idx);
    y = Comp_str_ln(idx);
    mdl_cem = fitlm(X_cem, y);
    coefficients_cem = [coefficients_cem; [mdl_cem.Coefficients.Estimate', age]];
    
    % Binder case
    X_binder = wc_binder(idx);
    mdl_binder = fitlm(X_binder, y);
    coefficients_binder = [coefficients_binder; [mdl_binder.Coefficients.Estimate', age]];
end

% Plot parameters vs log(Age)
figure;
subplot(2,2,1)
plot(log(coefficients_cem(:,3)), coefficients_cem(:,1), 'o')
title('Cement case: b₀ vs log(Age)')
xlabel('log(Age)'), ylabel('b₀')

subplot(2,2,2)
plot(log(coefficients_cem(:,3)), coefficients_cem(:,2), 'o')
title('Cement case: b₁ vs log(Age)')
xlabel('log(Age)'), ylabel('b₁')

subplot(2,2,3)
plot(log(coefficients_binder(:,3)), coefficients_binder(:,1), 'o')
title('Binder case: b₀ vs log(Age)')
xlabel('log(Age)'), ylabel('b₀')

subplot(2,2,4)
plot(log(coefficients_binder(:,3)), coefficients_binder(:,2), 'o')
title('Binder case: b₁ vs log(Age)')
xlabel('log(Age)'), ylabel('b₁')

%% Step 4: Second Regression Stage
% Cement case
mdl_b0_cem = fitlm(log(coefficients_cem(:,3)), coefficients_cem(:,1));
mdl_b1_cem = fitlm(log(coefficients_cem(:,3)), coefficients_cem(:,2));

% Binder case
mdl_b0_binder = fitlm(log(coefficients_binder(:,3)), coefficients_binder(:,1));
mdl_b1_binder = fitlm(log(coefficients_binder(:,3)), coefficients_binder(:,2));

% Age=14 analysis
age_14 = 14;
idx_14 = (data.Age == age_14);

% Cement case plots
figure;
subplot(1,2,1)
scatter(wc_cem(idx_14), Comp_str_ln(idx_14))
hold on
plot(wc_cem(idx_14), coefficients_cem(coefficients_cem(:,3)==age_14,1) + ...
    coefficients_cem(coefficients_cem(:,3)==age_14,2)*wc_cem(idx_14))
plot(wc_cem(idx_14), mdl_b0_cem.predict(log(age_14)) + ...
    mdl_b1_cem.predict(log(age_14))*wc_cem(idx_14))
title('Cement case: Age=14')
legend('Data','Original Model','Estimated Model')

% Binder case plots
subplot(1,2,2)
scatter(wc_binder(idx_14), Comp_str_ln(idx_14))
hold on
plot(wc_binder(idx_14), coefficients_binder(coefficients_binder(:,3)==age_14,1) + ...
    coefficients_binder(coefficients_binder(:,3)==age_14,2)*wc_binder(idx_14))
plot(wc_binder(idx_14), mdl_b0_binder.predict(log(age_14)) + ...
    mdl_b1_binder.predict(log(age_14))*wc_binder(idx_14))
title('Binder case: Age=14')
legend('Data','Original Model','Estimated Model')

%% Step 5: Model Evaluation
% Prediction function
predict_strength = @(wc, age, b0_model, b1_model) ...
    exp(b0_model.predict(log(age)) + b1_model.predict(log(age)).*wc);

% Calculate predictions
% Training data
train_pred_cem = predict_strength(wc_cem(ismember(data,train_data)), ...
    train_data.Age, mdl_b0_cem, mdl_b1_cem);
train_pred_binder = predict_strength(wc_binder(ismember(data,train_data)), ...
    train_data.Age, mdl_b0_binder, mdl_b1_binder);

% Testing data
test_pred_cem = predict_strength(wc_cem(ismember(data,test_data)), ...
    test_data.Age, mdl_b0_cem, mdl_b1_cem);
test_pred_binder = predict_strength(wc_binder(ismember(data,test_data)), ...
    test_data.Age, mdl_b0_binder, mdl_b1_binder);

% Calculate R² values
ss_tot_train = sum((train_data.Comp_strength - mean(train_data.Comp_strength)).^2);
ss_tot_test = sum((test_data.Comp_strength - mean(test_data.Comp_strength)).^2);

r2_table = struct();
[r2_table.Train.Cement.raw, r2_table.Train.Cement.transformed] = ...
    calculate_r2(train_data.Comp_strength, train_pred_cem, Comp_str_ln(ismember(data,train_data)));
[r2_table.Test.Cement.raw, r2_table.Test.Cement.transformed] = ...
    calculate_r2(test_data.Comp_strength, test_pred_cem, Comp_str_ln(ismember(data,test_data)));
[r2_table.Train.Binder.raw, r2_table.Train.Binder.transformed] = ...
    calculate_r2(train_data.Comp_strength, train_pred_binder, Comp_str_ln(ismember(data,train_data)));
[r2_table.Test.Binder.raw, r2_table.Test.Binder.transformed] = ...
    calculate_r2(test_data.Comp_strength, test_pred_binder, Comp_str_ln(ismember(data,test_data)));

% Display results
fprintf('R² Results:\n')
fprintf('%-20s | %-15s | %-15s\n', 'Dataset', 'Cement Case', 'Binder Case')
fprintf('%-20s | %-7.3f/%-7.3f | %-7.3f/%-7.3f\n', ...
    'Training (Raw/Trans)', ...
    r2_table.Train.Cement.raw, r2_table.Train.Cement.transformed, ...
    r2_table.Train.Binder.raw, r2_table.Train.Binder.transformed)
fprintf('%-20s | %-7.3f/%-7.3f | %-7.3f/%-7.3f\n', ...
    'Testing (Raw/Trans)', ...
    r2_table.Test.Cement.raw, r2_table.Test.Cement.transformed, ...
    r2_table.Test.Binder.raw, r2_table.Test.Binder.transformed)

% Residual plots
figure;
subplot(1,2,1)
histogram(train_data.Comp_strength - train_pred_cem, 'Normalization','pdf')
hold on
histogram(test_data.Comp_strength - test_pred_cem, 'Normalization','pdf')
title('Cement Case Residuals')
legend('Training','Testing')

subplot(1,2,2)
histogram(train_data.Comp_strength - train_pred_binder, 'Normalization','pdf')
hold on
histogram(test_data.Comp_strength - test_pred_binder, 'Normalization','pdf')
title('Binder Case Residuals')
legend('Training','Testing')

function [r2_raw, r2_trans] = calculate_r2(true_raw, pred_raw, true_trans)
    ss_res_raw = sum((true_raw - pred_raw).^2);
    ss_tot_raw = sum((true_raw - mean(true_raw)).^2);
    r2_raw = 1 - (ss_res_raw/ss_tot_raw);
    
    ss_res_trans = sum((true_trans - log(pred_raw)).^2);
    ss_tot_trans = sum((true_trans - mean(true_trans)).^2);
    r2_trans = 1 - (ss_res_trans/ss_tot_trans);
end
