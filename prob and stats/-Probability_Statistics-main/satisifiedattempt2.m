 %% ENGI 2211 - Optimized Regression Solution
% Robust implementation using regularized regression and feature engineering

%% Step 1: Data Preparation
data = readtable('Concrete_Data.csv');
%Purpose: Load the concrete dataset from a CSV file into a table structure (data).
%Why? The dataset contains age, water content, cement content, slag, ash, and compressive strength, which are needed for modeling.

% Identify unique ages with sample counts
[uniqueAges, ~, idx] = unique(data.Age);
sampleCounts = accumarray(idx, 1);
%Purpose: Find unique ages in the dataset and count how many times each age appears.
%Why? We need to split the dataset into training and testing based on the number of samples available for each age.

% Split data using statistical validation
trainAges = uniqueAges(sampleCounts >= 50);
testAges = uniqueAges(sampleCounts < 50);
%Purpose: If an age appears 50 times or more, it is used for training. Otherwise, it is for testing.

trainData = data(ismember(data.Age, trainAges), :);
testData = data(ismember(data.Age, testAges), :);
%Purpose: Split the dataset based on age into training (trainData) and testing (testData).

fprintf('Training samples: %d (Ages with ≥50 samples)\n', height(trainData));
fprintf('Testing samples: %d (Ages with <50 samples)\n', height(testData));
%Purpose: Print the number of training and testing samples for verification.

%% Step 1: Enhanced Data Cleaning
% Check for missing values in original data
disp('Missing values in raw data:');
disp(sum(ismissing(data)));

% Handle missing values before feature engineering
data.Cement = fillmissing(data.Cement, 'constant', 0.001); % Prevent zero cement
data.Slag = fillmissing(data.Slag, 'constant', 0.001);
data.Ash = fillmissing(data.Ash, 'constant', 0.001);

%% Step 2: Advanced Feature Engineering
epsilon = 1e-6;
%Purpose: A small constant (epsilon) is added to prevent division by zero when normalizing values.

% Safe division with zero protection
trainData.wc_cem = trainData.Water ./ max(trainData.Cement, epsilon);
trainData.wc_binder = trainData.Water ./ max(trainData.Cement + trainData.Slag + trainData.Ash, epsilon);

% Validate feature matrices
assert(all(isfinite(trainData.wc_cem)), 'NaN/Inf in wc_cem');
assert(all(isfinite(trainData.wc_binder)), 'NaN/Inf in wc_binder');

% Core transformations
trainData.Comp_str_ln = log(trainData.Comp_strength + epsilon);
%Purpose: Apply a log transformation to compressive strength to make the data more linear.

% Enhanced features
trainData.logAge = log(trainData.Age);
trainData.wc_cem_sq = trainData.wc_cem.^2;
trainData.wc_binder_sq = trainData.wc_binder.^2;
%Purpose: Add nonlinear features (logAge, squared ratios) to improve prediction accuracy.

% Apply same transformations to test data
testData.Comp_str_ln = log(testData.Comp_strength + epsilon);
testData.wc_cem = testData.Water ./ max(testData.Cement, epsilon);
testData.wc_binder = testData.Water ./ max(testData.Cement + testData.Slag + testData.Ash, epsilon);
testData.logAge = log(testData.Age);
testData.wc_cem_sq = testData.wc_cem.^2;
testData.wc_binder_sq = testData.wc_binder.^2;

% Add variable validation checks
assert(istable(testData), 'Testing data not in table format');
assert(all(ismember({'Water','Cement','Slag','Ash'}, testData.Properties.VariableNames)), 'Missing required columns in testData');

% Consider renaming for clarity
trainTbl = trainData;  % Instead of trainData
testTbl = testData;    % Instead of testData

%% Step 3: Regularized Regression with Interaction Terms
predictors = [trainTbl.wc_cem, trainTbl.wc_binder, trainTbl.logAge,...
              trainTbl.wc_cem_sq, trainTbl.wc_binder_sq];
response = trainTbl.Comp_str_ln;

% Bayesian optimized regression ensemble
ensembleMdl = fitrensemble(predictors, response,...
    'Method', 'LSBoost',...
    'Learners', templateTree('MaxNumSplits', 20),...
    'NumLearningCycles', 500,...
    'LearnRate', 0.01,...
    'OptimizeHyperparameters', {'NumLearningCycles','LearnRate'},...
    'HyperparameterOptimizationOptions', struct(...
        'AcquisitionFunctionName', 'expected-improvement-plus',...
        'MaxObjectiveEvaluations', 50,...
        'ShowPlots', false));

%% Step 4: Model Evaluation
% Prepare test data with the same features as training data
testPredictors = [testTbl.wc_cem, testTbl.wc_binder, testTbl.logAge,...
                  testTbl.wc_cem_sq, testTbl.wc_binder_sq];

% Generate predictions
trainPred = predict(ensembleMdl, predictors);
testPred = predict(ensembleMdl, testPredictors);

% Convert back to original scale
trainPredRaw = exp(trainPred);
testPredRaw = exp(testPred);

% Performance metrics
r2 = @(y,yhat) 1 - sum((y - yhat).^2)/sum((y - mean(y)).^2);

fprintf('\nRegression Performance\n');
fprintf('-----------------------\n');
fprintf('Case\t\tTraining R²\tTesting R²\n');
fprintf('Cement+Binder\t%.4f\t\t%.4f\n',...
    r2(trainTbl.Comp_strength, trainPredRaw),...
    r2(testTbl.Comp_strength, testPredRaw));

%% Revised Model Separation
% Cement-specific model
predictorsCement = [trainTbl.wc_cem, trainTbl.logAge, trainTbl.wc_cem_sq];
ensembleMdlCement = fitrensemble(predictorsCement, response,...
    'Method', 'LSBoost',...
    'Learners', templateTree('MaxNumSplits', 20),...
    'NumLearningCycles', 500,...
    'LearnRate', 0.01,...
    'OptimizeHyperparameters', {'NumLearningCycles','LearnRate'},...
    'HyperparameterOptimizationOptions', struct(...
        'AcquisitionFunctionName', 'expected-improvement-plus',...
        'MaxObjectiveEvaluations', 50,...
        'ShowPlots', false));

% Binder-specific model
predictorsBinder = [trainTbl.wc_binder, trainTbl.logAge, trainTbl.wc_binder_sq];
ensembleMdlBinder = fitrensemble(predictorsBinder, response,...
    'Method', 'LSBoost',...
    'Learners', templateTree('MaxNumSplits', 20),...
    'NumLearningCycles', 500,...
    'LearnRate', 0.01,...
    'OptimizeHyperparameters', {'NumLearningCycles','LearnRate'},...
    'HyperparameterOptimizationOptions', struct(...
        'AcquisitionFunctionName', 'expected-improvement-plus',...
        'MaxObjectiveEvaluations', 50,...
        'ShowPlots', false));

% Prepare test data for separate models
testPredictorsCement = [testTbl.wc_cem, testTbl.logAge, testTbl.wc_cem_sq];
testPredictorsBinder = [testTbl.wc_binder, testTbl.logAge, testTbl.wc_binder_sq];

% Generate predictions for both models
trainPredCement = predict(ensembleMdlCement, predictorsCement);
testPredCement = predict(ensembleMdlCement, testPredictorsCement);

trainPredBinder = predict(ensembleMdlBinder, predictorsBinder);
testPredBinder = predict(ensembleMdlBinder, testPredictorsBinder);

% Convert back to original scale
trainPredCementRaw = exp(trainPredCement);
testPredCementRaw = exp(testPredCement);
trainPredBinderRaw = exp(trainPredBinder);
testPredBinderRaw = exp(testPredBinder);

% Performance metrics for separate models
fprintf('\nSeparate Models Performance\n');
fprintf('-----------------------\n');
fprintf('Model\t\tTraining R²\tTesting R²\n');
fprintf('Cement\t\t%.4f\t\t%.4f\n',...
    r2(trainTbl.Comp_strength, trainPredCementRaw),...
    r2(testTbl.Comp_strength, testPredCementRaw));
fprintf('Binder\t\t%.4f\t\t%.4f\n',...
    r2(trainTbl.Comp_strength, trainPredBinderRaw),...
    r2(testTbl.Comp_strength, testPredBinderRaw));

%% Q-Q Plot Generation
figure;
subplot(1,2,1)
qqplot(trainTbl.Comp_strength - trainPredCementRaw);
title('Cement Model Residuals Q-Q');
grid on;

subplot(1,2,2) 
qqplot(trainTbl.Comp_strength - trainPredBinderRaw);
title('Binder Model Residuals Q-Q');
grid on;

%% Bias Verification
fprintf('\nResidual Bias Analysis\n');
fprintf('-----------------------\n');
fprintf('Cement residual bias: μ = %.2f\n', mean(trainTbl.Comp_strength - trainPredCementRaw));
fprintf('Binder residual bias: μ = %.2f\n', mean(trainTbl.Comp_strength - trainPredBinderRaw));

% Test set bias verification
fprintf('Cement test residual bias: μ = %.2f\n', mean(testTbl.Comp_strength - testPredCementRaw));
fprintf('Binder test residual bias: μ = %.2f\n', mean(testTbl.Comp_strength - testPredBinderRaw));

%% Step 5: Parameter-Age Relationships
% Revised Parameter-Age Relationship Setup
[uniqueTrainAges, ~, ageIDs] = unique(trainTbl.Age);
numUniqueAges = length(uniqueTrainAges);

% Preallocate with NaN to detect missing ages
paramsCem = nan(numUniqueAges, 2);
paramsBind = nan(numUniqueAges, 2);

for i = 1:numUniqueAges
    ageMask = (trainTbl.Age == uniqueTrainAges(i));
    
    % Cement case
    X_cem = [ones(sum(ageMask),1), trainTbl.wc_cem(ageMask)];
    paramsCem(i,:) = X_cem\trainTbl.Comp_str_ln(ageMask);
    
    % Binder case
    X_bind = [ones(sum(ageMask),1), trainTbl.wc_binder(ageMask)];
    paramsBind(i,:) = X_bind\trainTbl.Comp_str_ln(ageMask);
end

%% Safe 14-Day Plotting
age14 = 14;
[hasAge14, age14Idx] = ismember(age14, uniqueTrainAges);

if hasAge14
    % Get logical index within training data
    age14Mask = (trainTbl.Age == age14);
    
    % Plot with parameter access using age14Idx
    b0_orig = paramsCem(age14Idx, 1);
    b1_orig = paramsCem(age14Idx, 2);
else
    warning('Age 14 not present in training data - cannot plot comparison');
    return;
end

%% Step 6: 14-Day Model Comparison
age14 = 14;
log_age14 = log(age14);

% Get 14-day data
idx14_train = (trainTbl.Age == age14);
idx14_test = (testTbl.Age == age14);

% Cement case plot
figure;
scatter(trainTbl.wc_cem(idx14_train), trainTbl.Comp_str_ln(idx14_train), 'k', 'filled');
hold on;
x_plot = linspace(min(trainTbl.wc_cem(idx14_train)), max(trainTbl.wc_cem(idx14_train)), 100);

% Original model
b0_orig = paramsCem(idx14_train,1);
b1_orig = paramsCem(idx14_train,2);
plot(x_plot, b0_orig + b1_orig*x_plot, 'r', 'LineWidth', 2);

% Estimated model
b0_est = ensembleMdl.Bias + ensembleMdl.Coefficients(1) * log_age14; % Example estimation
b1_est = ensembleMdl.Coefficients(2); % Example estimation
plot(x_plot, b0_est + b1_est*x_plot, 'g--', 'LineWidth', 2);

xlabel('Water:Cement Ratio');
ylabel('log(Compressive Strength)');
title('Cement Case - Age = 14 Days');
legend('Data', 'Original Model', 'Estimated Model');
grid on;

%% Step 7: Residual Density Plots
% Use the models and predictions already defined in the Revised Model Separation section
% Calculate residuals for Cement
residualsCement = testTbl.Comp_strength - testPredCementRaw;

% Calculate residuals for Binder
residualsBinder = testTbl.Comp_strength - testPredBinderRaw;

% Plot Cement residuals
figure;
histogram(residualsCement, 'BinWidth', 2, 'Normalization', 'pdf', 'FaceColor', 'b', 'DisplayName', 'Cement Test Residuals');
hold on;
% Plot Binder residuals
histogram(residualsBinder, 'BinWidth', 2, 'Normalization', 'pdf', 'FaceColor', 'r', 'DisplayName', 'Binder Test Residuals');
xlabel('Residual (MPa)');
ylabel('Probability Density');
title('Residual Distribution Comparison');
legend('show');
grid on;

%% Step 8: SHAP Value Analysis (Model Interpretation)
% Compute SHAP values using KernelSHAP
% Convert to matrix with validation
predictors = [trainTbl.wc_cem, trainTbl.wc_binder, trainTbl.logAge,...
              trainTbl.wc_cem_sq, trainTbl.wc_binder_sq];

% Final missing value check
if any(ismissing(predictors(:)))
    warning('Missing values detected - imputing with column medians');
    predictors = fillmissing(predictors, 'constant', median(predictors, 'omitnan'));
end

% Create validated table
featureNames = {'wc_cem', 'wc_binder', 'logAge', 'wc_cem_sq', 'wc_binder_sq'};
predictorsTable = array2table(predictors, 'VariableNames', featureNames);

% Create explainer with validated data
explainer = shapley(ensembleMdl, predictorsTable, ...
    'Method', 'interventional', ...  % Better for missing data robustness
    'UseParallel', true);            % Accelerate computation

% Verify explainer creation
if isempty(explainer.X)
    error('SHAP explainer failed - check predictor data integrity');
end

% Plot feature importance
figure;
bar(explainer.ShapleyValues);
title('SHAP Feature Importance');
ylabel('Mean Absolute SHAP Value');
xticklabels({'wc_{cem}', 'wc_{binder}', 'Age', 'wc_{cem}^2', 'wc_{binder}^2', 'Age*wc_{cem}', 'Age*wc_{binder}'});
xtickangle(45);

%% Combined Model Comparison
% Combine predictions from both models
combinedPredTrain = [trainPredCementRaw; trainPredBinderRaw];
combinedPredTest = [testPredCementRaw; testPredBinderRaw];
combinedActualTrain = [trainTbl.Comp_strength; trainTbl.Comp_strength];
combinedActualTest = [testTbl.Comp_strength; testTbl.Comp_strength];

% Calculate combined R²
r2_combined_train = r2(combinedActualTrain, combinedPredTrain);
r2_combined_test = r2(combinedActualTest, combinedPredTest);

fprintf('\nCombined Model Performance\n');
fprintf('-----------------------\n');
fprintf('Combined R² (Training): %.4f\n', r2_combined_train);
fprintf('Combined R² (Testing): %.4f\n', r2_combined_test);

% Create a figure for combined residuals
figure;
hold on;

% Plot residuals for Cement
histogram(trainTbl.Comp_strength - trainPredCementRaw, 'BinWidth', 2, 'Normalization', 'pdf', 'FaceColor', 'b', 'DisplayName', 'Cement Training');
histogram(testTbl.Comp_strength - testPredCementRaw, 'BinWidth', 2, 'Normalization', 'pdf', 'FaceColor', 'cyan', 'DisplayName', 'Cement Testing');

% Plot residuals for Binder
histogram(trainTbl.Comp_strength - trainPredBinderRaw, 'BinWidth', 2, 'Normalization', 'pdf', 'FaceColor', 'r', 'DisplayName', 'Binder Training');
histogram(testTbl.Comp_strength - testPredBinderRaw, 'BinWidth', 2, 'Normalization', 'pdf', 'FaceColor', 'magenta', 'DisplayName', 'Binder Testing');

% Add labels and title
xlabel('Residual (MPa)');
ylabel('Probability Density');
title(sprintf('Combined Residuals (Train R² = %.4f, Test R² = %.4f)', r2_combined_train, r2_combined_test));
legend('show');
grid on;
