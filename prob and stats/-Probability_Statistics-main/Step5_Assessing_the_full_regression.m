% Step 5: Calculate R² and residuals
% Training data predictions
log_age_train = log(trainData.Age);
wc_cem_train = trainData.Water ./ trainData.Cement;
wc_binder_train = trainData.Water ./ (trainData.Cement + trainData.Slag + trainData.Ash);

% Testing data predictions
log_age_test = log(testData.Age);
wc_cem_test = testData.Water ./ testData.Cement;
wc_binder_test = testData.Water ./ (testData.Cement + testData.Slag + testData.Ash);

% Cement case predictions
% Training
b0_pred_train = b0_params_cem(1) + b0_params_cem(2)*log_age_train;
b1_pred_train = b1_params_cem(1) + b1_params_cem(2)*log_age_train;
ln_strength_pred_train_cem = b0_pred_train + b1_pred_train.*wc_cem_train;
strength_pred_train_cem = exp(ln_strength_pred_train_cem);

% Testing
b0_pred_test = b0_params_cem(1) + b0_params_cem(2)*log_age_test;
b1_pred_test = b1_params_cem(1) + b1_params_cem(2)*log_age_test;
ln_strength_pred_test_cem = b0_pred_test + b1_pred_test.*wc_cem_test;
strength_pred_test_cem = exp(ln_strength_pred_test_cem);

% Binder case predictions
% Training
b0_pred_train = b0_params_binder(1) + b0_params_binder(2)*log_age_train;
b1_pred_train = b1_params_binder(1) + b1_params_binder(2)*log_age_train;
ln_strength_pred_train_binder = b0_pred_train + b1_pred_train.*wc_binder_train;
strength_pred_train_binder = exp(ln_strength_pred_train_binder);

% Testing
b0_pred_test = b0_params_binder(1) + b0_params_binder(2)*log_age_test;
b1_pred_test = b1_params_binder(1) + b1_params_binder(2)*log_age_test;
ln_strength_pred_test_binder = b0_pred_test + b1_pred_test.*wc_binder_test;
strength_pred_test_binder = exp(ln_strength_pred_test_binder);

% Calculate residuals
residuals_train_cem = trainData.Comp_strength - strength_pred_train_cem;
residuals_test_cem = testData.Comp_strength - strength_pred_test_cem;
residuals_train_binder = trainData.Comp_strength - strength_pred_train_binder;
residuals_test_binder = testData.Comp_strength - strength_pred_test_binder;

% Plot residual densities
figure(3)
subplot(2,1,1)
histogram(residuals_train_cem, 'Normalization', 'pdf', 'DisplayName', 'Training')
hold on
histogram(residuals_test_cem, 'Normalization', 'pdf', 'DisplayName', 'Testing')
title('Residual Distribution - Cement Model')
xlabel('Residual')
ylabel('Density')
legend('Location', 'best')
grid on
hold off

subplot(2,1,2)
histogram(residuals_train_binder, 'Normalization', 'pdf', 'DisplayName', 'Training')
hold on
histogram(residuals_test_binder, 'Normalization', 'pdf', 'DisplayName', 'Testing')
title('Residual Distribution - Binder Model')
xlabel('Residual')
ylabel('Density')
legend('Location', 'best')
grid on
hold off

% Calculate and display R² values
% Training data - Cement
SST_trans_train = sum((log(trainData.Comp_strength) - mean(log(trainData.Comp_strength))).^2);
SSE_trans_train_cem = sum((log(trainData.Comp_strength) - ln_strength_pred_train_cem).^2);
R2_train_cem_trans = 1 - SSE_trans_train_cem/SST_trans_train;

SST_raw_train = sum((trainData.Comp_strength - mean(trainData.Comp_strength)).^2);
SSE_raw_train_cem = sum((trainData.Comp_strength - strength_pred_train_cem).^2);
R2_train_cem_raw = 1 - SSE_raw_train_cem/SST_raw_train;

% Testing data - Cement
SST_trans_test = sum((log(testData.Comp_strength) - mean(log(testData.Comp_strength))).^2);
SSE_trans_test_cem = sum((log(testData.Comp_strength) - ln_strength_pred_test_cem).^2);
R2_test_cem_trans = 1 - SSE_trans_test_cem/SST_trans_test;

SST_raw_test = sum((testData.Comp_strength - mean(testData.Comp_strength)).^2);
SSE_raw_test_cem = sum((testData.Comp_strength - strength_pred_test_cem).^2);
R2_test_cem_raw = 1 - SSE_raw_test_cem/SST_raw_test;

% Display results
fprintf('\nR² Results:\n');
fprintf('Cement case:\n');
fprintf('Training: Transformed = %.3f, Raw = %.3f\n', R2_train_cem_trans, R2_train_cem_raw);
fprintf('Testing:  Transformed = %.3f, Raw = %.3f\n', R2_test_cem_trans, R2_test_cem_raw);