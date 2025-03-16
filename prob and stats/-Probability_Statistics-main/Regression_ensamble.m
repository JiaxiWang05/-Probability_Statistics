%% ENGI 2211 - Coursework-Compliant Regression Solution
% Combines required linear regression with optimized ensemble extension

%% Step 1: Data Preparation (Strictly Follow Coursework Requirements)
data = readtable('Concrete_Data.csv');

% Split data by age as specified
[uniqueAges, ~, idx] = unique(data.Age);
sampleCounts = accumarray(idx, 1);
trainAges = uniqueAges(sampleCounts >= 50);
testAges = uniqueAges(sampleCounts < 50);

trainData = data(ismember(data.Age, trainAges), :);
testData = data(ismember(data.Age, testAges), :);

fprintf('Training samples: %d (Ages with ≥50 samples)\n', height(trainData));
fprintf('Testing samples: %d (Ages with <50 samples)\n\n', height(testData));

%% Step 2: Transformations (Coursework-Mandated)
epsilon = 1e-6;
trainData.Comp_str_ln = log(trainData.Comp_strength + epsilon);
trainData.wc_cem = trainData.Water ./ (trainData.Cement + epsilon);
trainData.wc_binder = trainData.Water ./ (trainData.Cement + trainData.Slag + trainData.Ash + epsilon);

testData.Comp_str_ln = log(testData.Comp_strength + epsilon);
testData.wc_cem = testData.Water ./ (testData.Cement + epsilon);
testData.wc_binder = testData.Water ./ (testData.Cement + testData.Slag + testData.Ash + epsilon);

%% Step 3: First-Stage Regression (Core Requirement)
% Modified to store all regression outputs for coursework validation
uniqueTrainAges = unique(trainData.Age);
numAges = length(uniqueTrainAges);

% Preallocate matrices for regression parameters
paramsCem = zeros(numAges, 2);  % [intercept, slope]
paramsBind = zeros(numAges, 2);
ageList = zeros(numAges, 1);

fprintf('First-Stage Linear Regression Results:\n');
fprintf('Age\tCement R²\tBinder R²\n');

for i = 1:numAges
    currentAge = uniqueTrainAges(i);
    ageMask = (trainData.Age == currentAge);
    ageList(i) = currentAge;
    
    % Cement case
    X_cem = [ones(sum(ageMask),1), trainData.wc_cem(ageMask)];
    y = trainData.Comp_str_ln(ageMask);
    b_cem = X_cem\y;
    paramsCem(i,:) = b_cem;
    y_pred_cem = X_cem*b_cem;
    r2_cem = 1 - sum((y - y_pred_cem).^2)/sum((y - mean(y)).^2);
    
    % Binder case
    X_bind = [ones(sum(ageMask),1), trainData.wc_binder(ageMask)];
    b_bind = X_bind\y;
    paramsBind(i,:) = b_bind;
    y_pred_bind = X_bind*b_bind;
    r2_bind = 1 - sum((y - y_pred_bind).^2)/sum((y - mean(y)).^2);
    
    fprintf('%d\t%.4f\t\t%.4f\n', currentAge, r2_cem, r2_bind);
end

%% Step 4: Second-Stage Regression (Coursework Requirement)
% Relationship between parameters and log(age)
logAge = log(ageList);

% For β₀
X_second = [ones(numAges,1), logAge];
b0_cem = X_second\paramsCem(:,1);
b0_bind = X_second\paramsBind(:,1);

% For β₁
b1_cem = X_second\paramsCem(:,2);
b1_bind = X_second\paramsBind(:,2);

% Store second-stage parameters
secondStageParams = struct(...
    'b0_cem', b0_cem,...
    'b0_bind', b0_bind,...
    'b1_cem', b1_cem,...
    'b1_bind', b1_bind);

%% Step 5: Full Model Predictions (Coursework Requirement)
function [y_trans, y_raw] = predictStrength(X, age, params, secondParams, isCement)
    log_age = log(age);
    if isCement
        b0 = [1 log_age] * secondParams.b0_cem;
        b1 = [1 log_age] * secondParams.b1_cem;
    else
        b0 = [1 log_age] * secondParams.b0_bind;
        b1 = [1 log_age] * secondParams.b1_bind;
    end
    y_trans = b0 + b1*X;
    y_raw = exp(y_trans) - epsilon;
end

% Generate predictions for coursework table
% Cement case
[~, trainPredCem] = predictStrength(trainData.wc_cem, trainData.Age, paramsCem, secondStageParams, true);
[~, testPredCem] = predictStrength(testData.wc_cem, testData.Age, paramsCem, secondStageParams, true);

% Binder case
[~, trainPredBind] = predictStrength(trainData.wc_binder, trainData.Age, paramsBind, secondStageParams, false);
[~, testPredBind] = predictStrength(testData.wc_binder, testData.Age, paramsBind, secondStageParams, false);

%% Step 6: Performance Evaluation (Coursework Table)
% Modified to include both transformed and raw metrics
r2 = @(y,yhat) 1 - sum((y - yhat).^2)/sum((y - mean(y)).^2);

% Transformed space predictions
trainTransCem = predictStrength(trainData.wc_cem, trainData.Age, paramsCem, secondStageParams, true);
testTransCem = predictStrength(testData.wc_cem, testData.Age, paramsCem, secondStageParams, true);

trainTransBind = predictStrength(trainData.wc_binder, trainData.Age, paramsBind, secondStageParams, false);
testTransBind = predictStrength(testData.wc_binder, testData.Age, paramsBind, secondStageParams, false);

% Compile results table
resultsTable = array2table(zeros(2,4),...
    'VariableNames', {'Cement_Trans', 'Cement_Raw', 'Binder_Trans', 'Binder_Raw'},...
    'RowNames', {'Training', 'Testing'});

resultsTable{'Training','Cement_Trans'} = r2(trainData.Comp_str_ln, trainTransCem);
resultsTable{'Training','Cement_Raw'} = r2(trainData.Comp_strength, trainPredCem);
resultsTable{'Training','Binder_Trans'} = r2(trainData.Comp_str_ln, trainTransBind);
resultsTable{'Training','Binder_Raw'} = r2(trainData.Comp_strength, trainPredBind);

resultsTable{'Testing','Cement_Trans'} = r2(testData.Comp_str_ln, testTransCem);
resultsTable{'Testing','Cement_Raw'} = r2(testData.Comp_strength, testPredCem);
resultsTable{'Testing','Binder_Trans'} = r2(testData.Comp_str_ln, testTransBind);
resultsTable{'Testing','Binder_Raw'} = r2(testData.Comp_strength, testPredBind);

disp('R² Results Table:');
disp(resultsTable);

%% Step 7: Enhanced Model (For Higher Marks)
% Bayesian-optimized ensemble as extension
predictors = [trainData.wc_cem, trainData.wc_binder, log(trainData.Age),...
              trainData.wc_cem.^2, trainData.wc_binder.^2];
response = trainData.Comp_str_ln;

ensembleMdl = fitrensemble(predictors, response,...
    'Method', 'LSBoost',...
    'Learners', templateTree('MaxNumSplits',20),...
    'OptimizeHyperparameters', {'NumLearningCycles','LearnRate'},...
    'HyperparameterOptimizationOptions', struct(...
        'MaxObjectiveEvaluations',50,...
        'ShowPlots',false));

% Generate enhanced predictions
trainPredEns = exp(predict(ensembleMdl, predictors)) - epsilon;
testPredEns = exp(predict(ensembleMdl, [testData.wc_cem, testData.wc_binder, log(testData.Age),...
                  testData.wc_cem.^2, testData.wc_binder.^2])) - epsilon;

% Compare with coursework model
fprintf('\nEnhanced Model Performance:\n');
fprintf('Case\t\tTraining R²\tTesting R²\n');
fprintf('Cement+Binder\t%.4f\t\t%.4f\n',...
    r2(trainData.Comp_strength, trainPredEns),...
    r2(testData.Comp_strength, testPredEns));

%% Step 8: Required Visualizations (Coursework)
% 14-day comparison plot
age14 = 14;
idx14 = (trainData.Age == age14);

figure;
scatter(trainData.wc_cem(idx14), trainData.Comp_str_ln(idx14), 'k', 'filled');
hold on;

% Coursework model prediction
x_plot = linspace(min(trainData.wc_cem(idx14)), max(trainData.wc_cem(idx14)), 100);
y_course = predictStrength(x_plot', age14, paramsCem, secondStageParams, true);

% Enhanced model prediction
X_enhanced = [x_plot', repmat([mean(trainData.wc_binder(idx14)), log(age14),...
               mean(trainData.wc_cem(idx14).^2), mean(trainData.wc_binder(idx14).^2)], length(x_plot),1)];
y_enhanced = predict(ensembleMdl, X_enhanced);

plot(x_plot, y_course, 'r', 'LineWidth', 2);
plot(x_plot, y_enhanced, 'g--', 'LineWidth', 2);
xlabel('Water:Cement Ratio');
ylabel('log(Compressive Strength)');
title('Age = 14 Days Comparison');
legend('Data', 'Coursework Model', 'Enhanced Model', 'Location', 'best');
grid on;

%% Step 9: Residual Analysis (Coursework Requirement)
figure;
subplot(1,2,1);
histogram(trainData.Comp_strength - trainPredCem, 'BinWidth', 2, 'Normalization', 'pdf');
hold on;
histogram(testData.Comp_strength - testPredCem, 'BinWidth', 2, 'Normalization', 'pdf');
title('Cement Case Residuals');
xlabel('Residual (MPa)'); 
ylabel('Probability Density');
legend('Training', 'Testing');

subplot(1,2,2);
histogram(trainData.Comp_strength - trainPredBind, 'BinWidth', 2, 'Normalization', 'pdf');
hold on;
histogram(testData.Comp_strength - testPredBind, 'BinWidth', 2, 'Normalization', 'pdf');
title('Binder Case Residuals');
xlabel('Residual (MPa)');
legend('Training', 'Testing');

%% Step 10: SHAPley Values (For Higher Marks)
% Check for missing values in predictors
if any(any(isnan(predictors)))
    error('Predictors contain missing values. Please handle them before proceeding.');
end

% Remove rows with missing values
predictorsTable = rmmissing(predictors);

% Call the shapley function
explainer = shapley(ensembleMdl, predictorsTable);
