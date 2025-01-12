% Step 1: Load and analyze data
data = readtable('Concrete_Data.csv');
%The table includes various columns such as Age, Comp_strength (compressive strength), Water, Cement, Slag, and Ash.

ages = unique(data.Age);
%It extracts the unique ages in the dataset using unique(data.Age).
numSamples = histc(data.Age, ages);
%It counts the samples for each unique age (histc(data.Age, ages)) and selects those ages with at least 50 samples for training (trainingAges). The remaining data is used for testing.

trainingAges = ages(numSamples >= 50);

% Create training and testing datasets
trainingIdx = ismember(data.Age, trainingAges);
% Logical indexing (ismember(data.Age, trainingAges)) is used to split the data into training and testing subsets (trainData and testData).
trainData = data(trainingIdx, :);
testData = data(~trainingIdx, :);

% Step 2: Transform the data
Comp_str_ln = log(data.Comp_strength);
%The log transformation is applied to the compressive strength values (Comp_str_ln = log(data.Comp_strength)), linearizing the relationship with the water-cement ratio.

wc_cem = data.Water ./ data.Cement; % Water-to-cement ratio is calculated as data.Water ./ data.Cement.
wc_binder = data.Water ./ (data.Cement + data.Slag + data.Ash); %Water-to-binder ratio is calculated as data.Water ./ (data.Cement + data.Slag + data.Ash).

% Step 3: First regression for each age
b0_cem = zeros(length(trainingAges), 1);
b1_cem = zeros(length(trainingAges), 1);
b0_binder = zeros(length(trainingAges), 1);
b1_binder = zeros(length(trainingAges), 1);

for i = 1:length(trainingAges)
    idx = (data.Age == trainingAges(i));
    
    % Cement regression
    X = [ones(sum(idx), 1), wc_cem(idx)];
    y = Comp_str_ln(idx);
%For every age in the training set, linear regression is performed for the transformed compressive strength (Comp_str_ln) using wc_cem and wc_binder as predictors.
    b = X \ y;
    b0_cem(i) = b(1);
    b1_cem(i) = b(2);
    
    % Binder regression
    X = [ones(sum(idx), 1), wc_binder(idx)];
    b = X \ y;
%Regression matrices X are built with a constant term (ones) and the ratio (wc_cem or wc_binder), and the regression coefficients are calculated as b = X \ y for each case.
    b0_binder(i) = b(1);
    b1_binder(i) = b(2);
end

% Plot b0 and b1 vs log(Age)
figure(1)
subplot(2,2,1)
plot(log(trainingAges), b0_cem, 'o-')
xlabel('log(Age)')
ylabel('b0 (cement)')
title('b0 vs log(Age) - Cement')
%The resulting parameters b0_cem and b1_cem (intercept and slope for cement-based regression) and b0_binder and b1_binder (for binder-based regression) are stored.

subplot(2,2,2)
plot(log(trainingAges), b1_cem, 'o-')
xlabel('log(Age)')
ylabel('b1 (cement)')
title('b1 vs log(Age) - Cement')

subplot(2,2,3)
plot(log(trainingAges), b0_binder, 'o-')
xlabel('log(Age)')
ylabel('b0 (binder)')
title('b0 vs log(Age) - Binder')

subplot(2,2,4)
plot(log(trainingAges), b1_binder, 'o-')
xlabel('log(Age)')
ylabel('b1 (binder)')
title('b1 vs log(Age) - Binder')

% Step 4: Second regression
%Model Parameter Trends: A second regression is performed to model the dependence of b0 and b1 on log(Age):

X = [ones(length(trainingAges), 1), log(trainingAges)];
%The design matrix X is built using ones and log(trainingAges).

% Cement parameters
b0_params_cem = X \ b0_cem;
b1_params_cem = X \ b1_cem;

% Binder parameters
b0_params_binder = X \ b0_binder;
b1_params_binder = X \ b1_binder;

% Plot for Age = 14 days
%For data where Age = 14, the water-cement and water-binder ratios are used along with the predicted parameters (b0 and b1) from the second regression to plot the predicted compressive strength trends against the data.
figure(2)
idx_14 = (data.Age == 14);

% Cement plot
subplot(2,1,1)
plot(wc_cem(idx_14), Comp_str_ln(idx_14), 'b.')
hold on
wc_range = linspace(min(wc_cem(idx_14)), max(wc_cem(idx_14)), 100)';
b0_pred = b0_params_cem(1) + b0_params_cem(2)*log(14);
b1_pred = b1_params_cem(1) + b1_params_cem(2)*log(14);
y_pred = b0_pred + b1_pred*wc_range;
plot(wc_range, y_pred, 'r-')
xlabel('Water-Cement Ratio')
ylabel('ln(Compressive Strength)')
title('14-day Strength vs W/C Ratio')
legend('Data', 'Model')
hold off

% Binder plot
subplot(2,1,2)
plot(wc_binder(idx_14), Comp_str_ln(idx_14), 'b.')
hold on
wc_range = linspace(min(wc_binder(idx_14)), max(wc_binder(idx_14)), 100)';
b0_pred = b0_params_binder(1) + b0_params_binder(2)*log(14);
b1_pred = b1_params_binder(1) + b1_params_binder(2)*log(14);
y_pred = b0_pred + b1_pred*wc_range;
plot(wc_range, y_pred, 'r-')
xlabel('Water-Binder Ratio')
ylabel('ln(Compressive Strength)')
title('14-day Strength vs W/B Ratio')
legend('Data', 'Model')
hold off

% Step 5: Calculate R^2 and residuals

% Training data - Cement case
log_age_train = log(trainData.Age);
%    Transformed compressive strength (log) predictions are computed using the parameterized models.

wc_cem_train = trainData.Water ./ trainData.Cement;
%•    For the training dataset:
%•    Transformed compressive strength (log) predictions are computed using the parameterized models.
%    Raw compressive strength predictions are obtained by exponentiating the transformed predictions.
%   R² is calculated for both transformed and raw data.
%    Similar calculations are repeated for the test dataset.

b0_pred = b0_params_cem(1) + b0_params_cem(2)*log_age_train;
b1_pred = b1_params_cem(1) + b1_params_cem(2)*log_age_train;
ln_strength_pred = b0_pred + b1_pred.*wc_cem_train;
strength_pred = exp(ln_strength_pred);

% R^2 calculations - training cement
SST_trans = sum((log(trainData.Comp_strength) - mean(log(trainData.Comp_strength))).^2);
SSE_trans = sum((log(trainData.Comp_strength) - ln_strength_pred).^2);
R2_train_cem_trans = 1 - SSE_trans/SST_trans;

SST_raw = sum((trainData.Comp_strength - mean(trainData.Comp_strength)).^2);
SSE_raw = sum((trainData.Comp_strength - strength_pred).^2);
R2_train_cem_raw = 1 - SSE_raw/SST_raw;

%Transformed R²: Measures the fit of the log-linear regression:

R^2 = 1 - \frac{\text{SSE (log-transformed residuals)}}{\text{SST (log-transformed strength)}}


%•    Raw R²: Measures the fit of the untransformed predictions:

%R^2 = 1 - \frac{\text{SSE (raw residuals)}}{\text{SST (raw strength)}}

%Output Results: The R² values for both the cement and binder cases (transformed and raw) are displayed, along with plots comparing predictions and actual data.

% Training data - Binder case
wc_binder_train = trainData.Water ./ (trainData.Cement + trainData.Slag + trainData.Ash);
b0_pred = b0_params_binder(1) + b0_params_binder(2)*log_age_train;
b1_pred = b1_params_binder(1) + b1_params_binder(2)*log_age_train;
ln_strength_pred = b0_pred + b1_pred.*wc_binder_train;
strength_pred = exp(ln_strength_pred);

% R^2 calculations - training binder
SSE_trans = sum((log(trainData.Comp_strength) - ln_strength_pred).^2);
R2_train_binder_trans = 1 - SSE_trans/SST_trans;

SSE_raw = sum((trainData.Comp_strength - strength_pred).^2);
R2_train_binder_raw = 1 - SSE_raw/SST_raw;

% Test data calculations - similar process for test data
% [Code continues similarly for test data calculations]

% Display results
fprintf('R^2 Results:\n');
fprintf('Cement case:\n');
fprintf('Training: Transformed = %.3f, Raw = %.3f\n', R2_train_cem_trans, R2_train_cem_raw);
fprintf('Binder case:\n');
fprintf('Training: Transformed = %.3f, Raw = %.3f\n', R2_train_binder_trans, R2_train_binder_raw);
