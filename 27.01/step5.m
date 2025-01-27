% --- Step 5: Assess Model Performance ---

% First, prepare test data with same transformations
testData.Comp_str_ln = log(testData.Comp_strength);
testData.wc_cem = testData.Water ./ testData.Cement;
testData.wc_binder = testData.Water ./ (testData.Cement + testData.Slag + testData.Ash);

% Calculate predictions and residuals for training data
train_pred_trans_cem = zeros(height(trainData), 1);
train_pred_trans_bind = zeros(height(trainData), 1);
train_pred_raw_cem = zeros(height(trainData), 1);
train_pred_raw_bind = zeros(height(trainData), 1);

for i = 1:height(trainData)
    [train_pred_trans_cem(i), train_pred_raw_cem(i)] = predict_strength(trainData.Age(i), trainData.wc_cem(i), gamma_b0_cem, gamma_b1_cem);
    [train_pred_trans_bind(i), train_pred_raw_bind(i)] = predict_strength(trainData.Age(i), trainData.wc_binder(i), gamma_b0_bind, gamma_b1_bind);
end

% Calculate predictions and residuals for test data
test_pred_trans_cem = zeros(height(testData), 1);
test_pred_trans_bind = zeros(height(testData), 1);
test_pred_raw_cem = zeros(height(testData), 1);
test_pred_raw_bind = zeros(height(testData), 1);

for i = 1:height(testData)
    [test_pred_trans_cem(i), test_pred_raw_cem(i)] = predict_strength(testData.Age(i), testData.wc_cem(i), gamma_b0_cem, gamma_b1_cem);
    [test_pred_trans_bind(i), test_pred_raw_bind(i)] = predict_strength(testData.Age(i), testData.wc_binder(i), gamma_b0_bind, gamma_b1_bind);
end

% Calculate residuals
train_res_trans_cem = trainData.Comp_str_ln - train_pred_trans_cem;
train_res_raw_cem = trainData.Comp_strength - train_pred_raw_cem;
train_res_trans_bind = trainData.Comp_str_ln - train_pred_trans_bind;
train_res_raw_bind = trainData.Comp_strength - train_pred_raw_bind;

test_res_trans_cem = testData.Comp_str_ln - test_pred_trans_cem;
test_res_raw_cem = testData.Comp_strength - test_pred_raw_cem;
test_res_trans_bind = testData.Comp_str_ln - test_pred_trans_bind;
test_res_raw_bind = testData.Comp_strength - test_pred_raw_bind;

% Calculate R² values using training means as reference
train_mean_trans = mean(trainData.Comp_str_ln);
train_mean_raw = mean(trainData.Comp_strength);

% Training R²
R2_trans_cem_train = 1 - sum(train_res_trans_cem.^2) / sum((trainData.Comp_str_ln - train_mean_trans).^2);
R2_raw_cem_train = 1 - sum(train_res_raw_cem.^2) / sum((trainData.Comp_strength - train_mean_raw).^2);
R2_trans_bind_train = 1 - sum(train_res_trans_bind.^2) / sum((trainData.Comp_str_ln - train_mean_trans).^2);
R2_raw_bind_train = 1 - sum(train_res_raw_bind.^2) / sum((trainData.Comp_strength - train_mean_raw).^2);

% Testing R² (using training means as reference)
R2_trans_cem_test = 1 - sum(test_res_trans_cem.^2) / sum((testData.Comp_str_ln - train_mean_trans).^2);
R2_raw_cem_test = 1 - sum(test_res_raw_cem.^2) / sum((testData.Comp_strength - train_mean_raw).^2);
R2_trans_bind_test = 1 - sum(test_res_trans_bind.^2) / sum((testData.Comp_str_ln - train_mean_trans).^2);
R2_raw_bind_test = 1 - sum(test_res_raw_bind.^2) / sum((testData.Comp_strength - train_mean_raw).^2);

% Create R² table
R2_table = table(...
    [R2_trans_cem_train; R2_trans_cem_test], ...
    [R2_raw_cem_train; R2_raw_cem_test], ...
    [R2_trans_bind_train; R2_trans_bind_test], ...
    [R2_raw_bind_train; R2_raw_bind_test], ...
    'VariableNames', {'Cement_Transformed', 'Cement_Raw', 'Binder_Transformed', 'Binder_Raw'}, ...
    'RowNames', {'Training Data', 'Testing Data'});

disp('R² Results:')
disp(R2_table)

% Plot residual densities (raw scale only)
figure('Position', [100, 100, 1000, 400]);

% Cement Case
subplot(1,2,1);
histogram(train_res_raw_cem, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5);
hold on;
histogram(test_res_raw_cem, 'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.5);
title('Residual Distribution - Cement Case');
xlabel('Residual (MPa)');
ylabel('Density');
legend('Training', 'Testing');
grid on;

% Binder Case
subplot(1,2,2);
histogram(train_res_raw_bind, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5);
hold on;
histogram(test_res_raw_bind, 'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.5);
title('Residual Distribution - Binder Case');
xlabel('Residual (MPa)');
ylabel('Density');
legend('Training', 'Testing');
grid on;

% Function definition must be at the end of the script
function [pred_trans, pred_raw] = predict_strength(age, wc_ratio, gamma_b0, gamma_b1)
    log_age = log(age);
    b0 = gamma_b0(1) + gamma_b0(2) * log_age;
    b1 = gamma_b1(1) + gamma_b1(2) * log_age;
    pred_trans = b0 + b1 * wc_ratio;
    pred_raw = exp(pred_trans);
end