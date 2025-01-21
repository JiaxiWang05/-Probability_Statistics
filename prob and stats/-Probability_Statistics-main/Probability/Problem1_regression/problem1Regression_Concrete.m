% Step 1: Load and analyze data
data = readtable('Concrete_Data.csv');
%The table includes various columns such as Age, Comp_strength (compressive strength), Water, Cement, Slag, and Ash.

% Find unique ages and count samples for each age
ages = unique(data.Age);
%It extracts the unique ages in the dataset using unique(data.Age).
numSamples = histc(data.Age, ages);
%It counts the samples for each unique age (histc(data.Age, ages)) and selects those ages with at least 50 samples for training (trainingAges). The remaining data is used for testing.

% Select ages with 50+ samples for training
trainingAges = ages(numSamples >= 50);
%It creates an array of ages that have at least 50 samples for training.

% Split data into training and testing sets
trainingIdx = ismember(data.Age, trainingAges);
%Logical indexing (ismember(data.Age, trainingAges)) is used to split the data into training and testing subsets (trainData and testData).
trainData = data(trainingIdx, :);
testData = data(~trainingIdx, :);
% Step 2: Transform the data
Comp_str_ln = log(data.Comp_strength);
%The log transformation is applied to the compressive strength values (Comp_str_ln = log(data.Comp_strength)), linearizing the relationship with the water-cement ratio.

wc_cem = data.Water ./ data.Cement; % Water-to-cement ratio is calculated as data.Water ./ data.Cement.
wc_binder = data.Water ./ (data.Cement + data.Slag + data.Ash); %Water-to-binder ratio is calculated as data.Water ./ (data.Cement + data.Slag + data.Ash).

% Step 3: First regressions for each age
b0_cem = zeros(length(trainingAges), 1);     % Initialize arrays
b1_cem = zeros(length(trainingAges), 1);
b0_binder = zeros(length(trainingAges), 1);
b1_binder = zeros(length(trainingAges), 1);

for i = 1:length(trainingAges)
    idx = (data.Age == trainingAges(i));    % Filter for current age
    
    % Cement regression
    X = [ones(sum(idx), 1), wc_cem(idx)];
    y = Comp_str_ln(idx);
    b = X \ y;
    b0_cem(i) = b(1);
    b1_cem(i) = b(2);
    
    % Binder regression
    X = [ones(sum(idx), 1), wc_binder(idx)];
    y = Comp_str_ln(idx);
    b = X \ y;
    b0_binder(i) = b(1);
    b1_binder(i) = b(2);
end

% Plot all four required plots in a single figure
figure(1)
% Top left: b0 vs log(Age) for cement case
subplot(2,2,1)
plot(log(trainingAges), b0_cem, 'o')
xlabel('log(Age)')
ylabel('b0')
title('b0 vs log(Age) - Cement')

% Top right: b1 vs log(Age) for cement case
subplot(2,2,2)
plot(log(trainingAges), b1_cem, 'o')
xlabel('log(Age)')
ylabel('b1')
title('b1 vs log(Age) - Cement')

% Bottom left: b0 vs log(Age) for binder case
subplot(2,2,3)
plot(log(trainingAges), b0_binder, 'o')
xlabel('log(Age)')
ylabel('b0')
title('b0 vs log(Age) - Binder')

% Bottom right: b1 vs log(Age) for binder case
subplot(2,2,4)
plot(log(trainingAges), b1_binder, 'o')
xlabel('log(Age)')
ylabel('b1')
title('b1 vs log(Age) - Binder')

% Step 4: Second regression with polynomial terms
% Add polynomial terms for both age and water-cement ratio
X_age = [ones(length(trainingAges), 1), log(trainingAges), (log(trainingAges)).^2];
b0_params_cem = X_age \ b0_cem_smooth;
b1_params_cem = X_age \ b1_cem_smooth;

% For 14-day strength prediction
log_age_14 = log(14);
b0_14 = b0_params_cem(1) + b0_params_cem(2)*log_age_14 + b0_params_cem(3)*log_age_14^2;
b1_14 = b1_params_cem(1) + b1_params_cem(2)*log_age_14 + b1_params_cem(3)*log_age_14^2;

% Create finer grid for smoother curves but limit to data range
wc_fine = linspace(min(wc_cem), max(wc_cem), 100)';  % Only use actual data range
idx_14 = data.Age == 14;  % Get 14-day data

% Linear prediction (remove quadratic term for stability)
strength_pred = b0_14 + b1_14*wc_fine;

% Update plotting
figure(2)
% Top: Cement case
subplot(2,1,1)
idx_14 = data.Age == 14;
plot(wc_cem(idx_14), Comp_str_ln(idx_14), 'b.', 'DisplayName', 'Data')
hold on
plot(wc_fine, strength_pred_cem, 'r-', 'LineWidth', 2, 'DisplayName', 'Model')
xlabel('Water-Cement Ratio')
ylabel('ln(Compressive Strength)')
title('14-day Strength vs W/C Ratio')
legend('Location', 'best')
hold off

% Bottom: Binder case
subplot(2,1,2)
plot(wc_binder(idx_14), Comp_str_ln(idx_14), 'b.', 'DisplayName', 'Data')
hold on
plot(wc_fine, strength_pred_binder, 'r-', 'LineWidth', 2, 'DisplayName', 'Model')
xlabel('Water-Binder Ratio')
ylabel('ln(Compressive Strength)')
title('14-day Strength vs W/B Ratio')
legend('Location', 'best')
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

%R^2 = 1 - SSE(log-transformed residuals) / SST(log-transformed strength)

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

% Consider adding additional terms for better fit
% For example, add quadratic terms to the model
X_quad = [ones(length(trainingAges), 1), log(trainingAges), log(trainingAges).^2];
b0_params_cem_quad = X_quad \ b0_cem_smooth;
b1_params_cem_quad = X_quad \ b1_cem_smooth;
b0_params_binder_quad = X_quad \ b0_binder_smooth;
b1_params_binder_quad = X_quad \ b1_binder_smooth;

% Update the prediction calculations using quadratic model
log_age_train = log(trainData.Age);
b0_pred = b0_params_cem_quad(1) + b0_params_cem_quad(2)*log_age_train + b0_params_cem_quad(3)*log_age_train.^2;
b1_pred = b1_params_cem_quad(1) + b1_params_cem_quad(2)*log_age_train + b1_params_cem_quad(3)*log_age_train.^2;

% Add polynomial fit for strength prediction
wc_range = linspace(min(wc_cem(idx_14)), max(wc_cem(idx_14)), 100)';
X_strength = [ones(length(wc_range), 1), wc_range, wc_range.^2];  % Quadratic terms for W/C ratio

% Update strength predictions with polynomial model
b0_pred = b0_params_cem_quad(1) + b0_params_cem_quad(2)*log(14) + b0_params_cem_quad(3)*log(14).^2;
b1_pred = b1_params_cem_quad(1) + b1_params_cem_quad(2)*log(14) + b1_params_cem_quad(3)*log(14).^2;

