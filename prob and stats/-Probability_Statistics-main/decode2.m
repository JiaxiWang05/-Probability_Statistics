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


%% Step 2: Advanced Feature Engineering
epsilon = 1e-6;
%Purpose: A small constant (epsilon) is added to prevent division by zero when normalizing values.

% Core transformations
trainData.Comp_str_ln = log(trainData.Comp_strength + epsilon);
%Purpose: Apply a log transformation to compressive strength to make the data more linear.

trainData.wc_cem = trainData.Water ./ (trainData.Cement + epsilon);
%Purpose: Compute the water-to-cement ratio, an important feature for concrete strength.


binderSum = trainData.Cement + trainData.Slag + trainData.Ash + epsilon;
%Purpose: Compute the water-to-binder ratio (cement + slag + ash). 
trainData.wc_binder = trainData.Water ./ binderSum;

% Enhanced features
trainData.logAge = log(trainData.Age);
trainData.wc_cem_sq = trainData.wc_cem.^2;
trainData.wc_binder_sq = trainData.wc_binder.^2;
%Purpose: Add nonlinear features (logAge, squared ratios) to improve prediction accuracy.

% Apply same transformations to test data
testData.Comp_str_ln = log(testData.Comp_strength + epsilon);
testData.wc_cem = testData.Water ./ (testData.Cement + epsilon);
testData.wc_binder = testData.Water ./ (testData.Cement + testData.Slag + testData.Ash + epsilon);
testData.logAge = log(testData.Age);
testData.wc_cem_sq = testData.wc_cem.^2;
testData.wc_binder_sq = testData.wc_binder.^2;


%% Step 3: Regularized Regression with Interaction Terms
predictors = [trainData.wc_cem, trainData.wc_binder, trainData.logAge,...
              trainData.wc_cem_sq, trainData.wc_binder_sq];
response = trainData.Comp_str_ln;

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
testPredictors = [testData.wc_cem, testData.wc_binder, testData.logAge,...
                  testData.wc_cem_sq, testData.wc_binder_sq];

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
    r2(trainData.Comp_strength, trainPredRaw),...
    r2(testData.Comp_strength, testPredRaw));

%% Step 5: Parameter-Age Relationships
% Revised Parameter-Age Relationship Setup
[uniqueTrainAges, ~, ageIDs] = unique(trainData.Age);
numUniqueAges = length(uniqueTrainAges);

% Preallocate with NaN to detect missing ages
paramsCem = nan(numUniqueAges, 2);
paramsBind = nan(numUniqueAges, 2);

for i = 1:numUniqueAges
    ageMask = (trainData.Age == uniqueTrainAges(i));
    
    % Cement case
    X_cem = [ones(sum(ageMask),1), trainData.wc_cem(ageMask)];
    paramsCem(i,:) = X_cem\trainData.Comp_str_ln(ageMask);
    
    % Binder case
    X_bind = [ones(sum(ageMask),1), trainData.wc_binder(ageMask)];
    paramsBind(i,:) = X_bind\trainData.Comp_str_ln(ageMask);
end

%% Safe 14-Day Plotting
age14 = 14;
[hasAge14, age14Idx] = ismember(age14, uniqueTrainAges);

if hasAge14
    % Get parameters using the unique age index
    b0_orig = paramsCem(age14Idx, 1);  % Fixed indexing
    b1_orig = paramsCem(age14Idx, 2);
else
    warning('Age 14 not present in training data - cannot plot comparison');
    return;
end

%% Step 6: 14-Day Model Comparison
age14 = 14;
log_age14 = log(age14);

% Get 14-day data
idx14_train = (trainData.Age == age14);
idx14_test = (testData.Age == age14);

% Cement case plot
figure;
scatter(trainData.wc_cem(idx14_train), trainData.Comp_str_ln(idx14_train), 'k', 'filled');
hold on;
x_plot = linspace(min(trainData.wc_cem(idx14_train)), max(trainData.wc_cem(idx14_train)), 100);

% Generate predictions for original model
x_plot_table = table(x_plot', log_age14*ones(size(x_plot')), x_plot'.^2, ...
    'VariableNames', {'wc_cem', 'logAge', 'wc_cem_sq'});
origPred = b0_orig + b1_orig*x_plot;

% Create properly dimensioned predictor matrix
x_plot = linspace(min(trainData.wc_cem), max(trainData.wc_cem), 100)';

predictor_matrix = [
    x_plot, ...                        % wc_cem (100x1)
    zeros(size(x_plot)), ...           % wc_binder placeholder (100x1)
    log_age14*ones(size(x_plot)), ...  % logAge (100x1)
    x_plot.^2, ...                     % wc_cem_sq (100x1)
    zeros(size(x_plot))                % wc_binder_sq placeholder (100x1)
];

% Verify matrix dimensions before prediction
assert(size(predictor_matrix,2) == 5, 'Matrix must have 5 predictor columns');
assert(size(predictor_matrix,1) == 100, 'Row count mismatch with x_plot');

ensemblePred = predict(ensembleMdl, predictor_matrix);

% Original model
b0_orig = paramsCem(age14Idx, 1);  % Fixed indexing
b1_orig = paramsCem(age14Idx, 2);
plot(x_plot, origPred, 'r', 'LineWidth', 2);
plot(x_plot, ensemblePred, 'g--', 'LineWidth', 2);

xlabel('Water:Cement Ratio');
ylabel('log(Compressive Strength)');
title('Cement Case - Age = 14 Days');
legend('Data', 'Original Model', 'Estimated Model');
grid on;

%% Step 7: Residual Density Plots
residuals = testData.Comp_strength - testPredRaw;

% Kernel density estimation
[f,xi] = ksdensity(residuals);
figure('Position', [100 100 1200 600]);
tiledlayout(1,2);

% Cement residuals
nexttile;
hold on;
histogram(trainData.Comp_strength - trainPredRaw, 'BinWidth', 2, 'Normalization', 'pdf');
histogram(testData.Comp_strength - testPredRaw, 'BinWidth', 2, 'Normalization', 'pdf');
title('Cement Case Residuals');
xlabel('Residual (MPa)'); ylabel('Probability Density');
legend('Training', 'Testing');
grid on;

% Binder residuals
nexttile;
hold on;
histogram(trainData.Comp_strength - trainPredRaw, 'BinWidth', 2, 'Normalization', 'pdf');
histogram(testData.Comp_strength - testPredRaw, 'BinWidth', 2, 'Normalization', 'pdf');
title('Binder Case Residuals');
xlabel('Residual (MPa)'); ylabel('Probability Density');
legend('Training', 'Testing');
grid on;

%% Step 8: SHAP Value Analysis (Model Interpretation)
%% Revised SHAP Value Analysis Section
% Convert predictors matrix to table with proper feature names
featureNames = {'wc_cem', 'wc_binder', 'logAge', 'wc_cem_sq', 'wc_binder_sq'};
predictorsTable = array2table(predictors(1:100,:), 'VariableNames', featureNames);

% Create explainer with training data reference
explainer = shapley(ensembleMdl, predictorsTable);

% Preallocate SHAP values storage
shapValues = cell(100, 1);

% Compute SHAP values for each query point individually
for i = 1:100
    % Get single-row table query point
    queryPoint = predictorsTable(i,:);
    
    % Compute Shapley values for this observation
    explainer = fit(explainer, queryPoint);
    
    % Store SHAP values with observation index
    shapValues{i} = explainer.Shapley;
    shapValues{i}.Observation = repmat(i, height(shapValues{i}), 1);
end

% Combine all SHAP values into one table
shapTable = vertcat(shapValues{:});

% Create summary plot using aggregated data
figure;
grpstats(shapTable, 'Predictor', {'mean', 'std'}, 'DataVars', 'Value');
title('SHAP Feature Importance Summary');
ylabel('Mean |SHAP Value|');
xlabel('Predictors');

%% Step 1: Enhanced Data Validation
% Convert to table with explicit data types
predictorsTable = varfun(@double, trainData(:, featureNames));

% Identify problematic values
missing_mask = ismissing(predictorsTable);
inf_mask = varfun(@(x)isinf(x), predictorsTable);

% Handle edge cases
predictorsTable = standardizeMissing(predictorsTable, {NaN, Inf, -Inf});
predictorsTable = fillmissing(predictorsTable, 'constant', 0.001); % Replace missing
predictorsTable = filloutliers(predictorsTable, 'clip'); % Handle infinite values

%% Step 2: Model-Data Alignment Check
% Verify feature compatibility
requiredFeatures = ensembleMdl.PredictorNames;
assert(isequal(sort(requiredFeatures), sort(featureNames)),...
    'Feature mismatch between model and explainer data');

%% Step 3: Robust SHAP Configuration
% Create explainer with validation
try
    explainer = shapley(ensembleMdl, predictorsTable,...
        'Method', 'interventional',...
        'UseParallel', true,...
        'Verbose', 2); % Show diagnostic messages
catch ME
    % Fallback: Matrix input with explicit feature names
    predictorsMatrix = table2array(predictorsTable);
    explainer = shapley(ensembleMdl, predictorsMatrix,...
        'PredictorNames', featureNames,...
        'Method', 'kernel');
end

%% Step 4: SHAP Value Computation with Validation
% Compute baseline SHAP values
shapValues = fit(explainer, predictorsTable(1:100,:));

% Verify additive property
delta = abs(sum(shapValues.ShapleyValues.Value, 2) + shapValues.ShapleyValues.Baseline - predict(ensembleMdl, predictorsTable(1:100,:)));
assert(all(delta < 1e-6), 'Additivity check failed');

%% Diagnostic Visualization
% Data Quality Report
figure;
subplot(2,1,1)
heatmap(double(ismissing(predictorsTable(1:100,:))));
title('Missing Values Pattern (First 100 Samples)');

subplot(2,1,2)
boxplot(table2array(predictorsTable), 'Labels', featureNames);
title('Feature Value Distributions');
extickangle(45);

%% Version Compatibility Check
% Verify SHAP implementation version
if ~contains(which('shapley'), '2023b')
    warning('SHAP implementation differs from recommended version');
end

%% Fallback: Custom Prediction Wrapper
% Create wrapper function to handle data inconsistencies
function pred = modelWrapper(model, X)
    X = fillmissing(X, 'constant', 0.001);
    X = max(X, 0); % Ensure non-negative values
    pred = predict(model, X);
end
