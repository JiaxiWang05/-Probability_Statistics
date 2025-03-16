% Step 1: Load and analyze data
data = readtable('Concrete_Data.csv');

% Find unique ages and count samples
ages = unique(data.Age);
numSamples = histc(data.Age, ages);

% Select ages with 50+ samples for training
trainingAges = ages(numSamples >= 50);

% Split data into training and testing
trainingIdx = ismember(data.Age, trainingAges);
trainData = data(trainingIdx, :);
testData = data(~trainingIdx, :);

% Step 2: Transform the data (use training data only)
% Calculate water-cement and water-binder ratios
wc_cem = trainData.Water ./ trainData.Cement;
wc_binder = trainData.Water ./ sum([trainData.Cement trainData.Slag trainData.Ash], 2);
Comp_str_ln = log(trainData.Comp_strength);

% Step 3: Perform age-specific regressions
uniqueTrainAges = unique(trainData.Age);
n_ages = length(uniqueTrainAges);

% Initialize parameter arrays
b0_cem = zeros(n_ages, 1);
b1_cem = zeros(n_ages, 1);
b0_binder = zeros(n_ages, 1);
b1_binder = zeros(n_ages, 1);

% Perform regressions for each age
for i = 1:n_ages
    current_age = uniqueTrainAges(i);
    idx_age = (trainData.Age == current_age);
    
    % Water-cement ratio regression
    mdl_cem = fitlm(wc_cem(idx_age), Comp_str_ln(idx_age));
    b0_cem(i) = mdl_cem.Coefficients.Estimate(1);
    b1_cem(i) = mdl_cem.Coefficients.Estimate(2);
    
    % Water-binder ratio regression
    mdl_binder = fitlm(wc_binder(idx_age), Comp_str_ln(idx_age));
    b0_binder(i) = mdl_binder.Coefficients.Estimate(1);
    b1_binder(i) = mdl_binder.Coefficients.Estimate(2);
end

% Step 4: Second level regressions
ln_age = log(uniqueTrainAges);

% Regression for b0 and b1 parameters
mdl_b0_cem = fitlm(ln_age, b0_cem);
mdl_b1_cem = fitlm(ln_age, b1_cem);
mdl_b0_binder = fitlm(ln_age, b0_binder);
mdl_b1_binder = fitlm(ln_age, b1_binder);

% For 14-day comparison
idx_14 = (trainData.Age == 14);
wc_cem_14 = wc_cem(idx_14);
wc_binder_14 = wc_binder(idx_14);
strength_ln_14 = Comp_str_ln(idx_14);

% Get parameters for 14-day predictions
ln_age_14 = log(14);
b0_cem_est = predict(mdl_b0_cem, ln_age_14);
b1_cem_est = predict(mdl_b1_cem, ln_age_14);
b0_binder_est = predict(mdl_b0_binder, ln_age_14);
b1_binder_est = predict(mdl_b1_binder, ln_age_14);

% Step 5: Full regression assessment
% Prepare test data transformations
wc_cem_test = testData.Water ./ testData.Cement;
wc_binder_test = testData.Water ./ sum([testData.Cement testData.Slag testData.Ash], 2);
ln_age_test = log(testData.Age);
ln_age_train = log(trainData.Age);

% Predict for training data
b0_cem_train = predict(mdl_b0_cem, ln_age_train);
b1_cem_train = predict(mdl_b1_cem, ln_age_train);
pred_ln_str_cem_train = b0_cem_train + b1_cem_train .* wc_cem;

b0_binder_train = predict(mdl_b0_binder, ln_age_train);
b1_binder_train = predict(mdl_b1_binder, ln_age_train);
pred_ln_str_binder_train = b0_binder_train + b1_binder_train .* wc_binder;

% Predict for test data
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
R2_cem_raw_train = 1 - sum((trainData.Comp_strength - pred_str_cem_train).^2) / ...
    sum((trainData.Comp_strength - mean(trainData.Comp_strength)).^2);
R2_cem_raw_test = 1 - sum((testData.Comp_strength - pred_str_cem_test).^2) / ...
    sum((testData.Comp_strength - mean(testData.Comp_strength)).^2);
R2_binder_raw_train = 1 - sum((trainData.Comp_strength - pred_str_binder_train).^2) / ...
    sum((trainData.Comp_strength - mean(trainData.Comp_strength)).^2);
R2_binder_raw_test = 1 - sum((testData.Comp_strength - pred_str_binder_test).^2) / ...
    sum((testData.Comp_strength - mean(testData.Comp_strength)).^2);

% Display R² values
disp('R² values for Cement case:');
disp(['Transformed Training: ', num2str(R2_cem_trans_train)]);
disp(['Transformed Testing: ', num2str(R2_cem_trans_test)]);
disp(['Raw Training: ', num2str(R2_cem_raw_train)]);
disp(['Raw Testing: ', num2str(R2_cem_raw_test)]);

disp('R² values for Binder case:');
disp(['Transformed Training: ', num2str(R2_binder_trans_train)]);
disp(['Transformed Testing: ', num2str(R2_binder_trans_test)]);
disp(['Raw Training: ', num2str(R2_binder_raw_train)]);
disp(['Raw Testing: ', num2str(R2_binder_raw_test)]);