% Load and prepare data
data = readtable('Concrete_Data.csv');

% Transform the data
Comp_str_ln = log(data.Comp_strength);
wc_cem = data.Water ./ data.Cement;

% Split data into training and testing
ages = unique(data.Age);
numSamples = histc(data.Age, ages);
trainingAges = ages(numSamples >= 50);
trainingIdx = ismember(data.Age, trainingAges);
trainData = data(trainingIdx, :);
testData = data(~trainingIdx, :);

% Polynomial features for cement model
poly_degree = 2;
X_train = [ones(size(trainData, 1), 1), ...
           wc_cem(trainingIdx), ...
           wc_cem(trainingIdx).^2, ...
           log(trainData.Age)];
X_test = [ones(size(testData, 1), 1), ...
          wc_cem(~trainingIdx), ...
          wc_cem(~trainingIdx).^2, ...
          log(testData.Age)];

% Regularization (Ridge Regression)
lambda = 1; % Regularization parameter
b_cem = (X_train' * X_train + lambda * eye(size(X_train, 2))) \ (X_train' * Comp_str_ln(trainingIdx));

% Predictions
ln_strength_pred_train_cem = X_train * b_cem;
ln_strength_pred_test_cem = X_test * b_cem;
strength_pred_train_cem = exp(ln_strength_pred_train_cem);
strength_pred_test_cem = exp(ln_strength_pred_test_cem);

% Calculate residuals
residuals_train_cem = trainData.Comp_strength - strength_pred_train_cem;
residuals_test_cem = testData.Comp_strength - strength_pred_test_cem;

% Plot residual distributions
figure
subplot(2,1,1)
histogram(residuals_train_cem, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'Training')
hold on
histogram(residuals_test_cem, 'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'Testing')
title('Residual Distribution - Cement Model with Polynomial Features')
xlabel('Residual (MPa)')
ylabel('Density')
legend('Location', 'best')
grid on
hold off

% Calculate R-squared values
% For transformed data (logarithmic scale)
SST_train_trans = sum((Comp_str_ln(trainingIdx) - mean(Comp_str_ln(trainingIdx))).^2);
SSE_train_trans = sum((Comp_str_ln(trainingIdx) - ln_strength_pred_train_cem).^2);
R2_train_trans = 1 - SSE_train_trans/SST_train_trans;

SST_test_trans = sum((Comp_str_ln(~trainingIdx) - mean(Comp_str_ln(~trainingIdx))).^2);
SSE_test_trans = sum((Comp_str_ln(~trainingIdx) - ln_strength_pred_test_cem).^2);
R2_test_trans = 1 - SSE_test_trans/SST_test_trans;

% For raw data
SST_train_raw = sum((trainData.Comp_strength - mean(trainData.Comp_strength)).^2);
SSE_train_raw = sum(residuals_train_cem.^2);
R2_train_raw = 1 - SSE_train_raw/SST_train_raw;

SST_test_raw = sum((testData.Comp_strength - mean(testData.Comp_strength)).^2);
SSE_test_raw = sum(residuals_test_cem.^2);
R2_test_raw = 1 - SSE_test_raw/SST_test_raw;

% Display results
fprintf('\nR-squared Results for Cement Model:\n');
fprintf('Training Data - Transformed: %.4f, Raw: %.4f\n', R2_train_trans, R2_train_raw);
fprintf('Testing Data - Transformed: %.4f, Raw: %.4f\n', R2_test_trans, R2_test_raw);

% Add subplot for Q-Q plot to check normality of residuals
subplot(2,1,2)
qqplot(residuals_train_cem)
title('Q-Q Plot of Training Residuals - Cement Model')
grid on

%