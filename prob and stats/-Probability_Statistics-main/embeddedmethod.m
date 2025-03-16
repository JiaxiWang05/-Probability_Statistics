%% Step 1: Refined Feature Selection and Engineering
% Load data and split as before
% [Previous data loading and initial splitting code]

% Add more specialized concrete-specific features
data.cement_content = data.Cement; 
data.total_binder = data.Cement + data.Slag + data.Ash;
data.fine_agg_ratio = data.Fine_Agg ./ (data.Fine_Agg + data.Coarse_Agg);
data.wc_cem_sqrt = sqrt(data.wc_cem);
data.wc_binder_sqrt = sqrt(data.wc_binder);
data.age_sqrt = sqrt(data.Age);
data.log_age_cubed = log_age.^3;
data.fine_coarse_ratio = data.Fine_Agg ./ data.Coarse_Agg;

%% Step 2: Feature Selection using Stepwise Regression
% Create a large pool of potential features
feature_vars = {'wc_cem', 'wc_binder', 'log_age', 'wc_cem_sq', 'wc_binder_sq', ...
                'log_age_sq', 'wc_cem_log_age', 'wc_binder_log_age', ...
                'wc_cem_log_age_sq', 'wc_binder_log_age_sq', 'wc_cem_sqrt', ...
                'wc_binder_sqrt', 'age_sqrt', 'log_age_cubed', 'fine_agg_ratio', ...
                'fine_coarse_ratio', 'cement_content', 'total_binder'};

% Create model formulas
cement_formula = 'Comp_str_ln ~ 1';
binder_formula = 'Comp_str_ln ~ 1';

% Perform stepwise regression to select best features
train_data = data(train_idx, :);
cement_mdl = stepwiselm(train_data, cement_formula, 'PredictorVars', feature_vars, ...
                         'Criterion', 'aic', 'Verbose', 0);
binder_mdl = stepwiselm(train_data, binder_formula, 'PredictorVars', feature_vars, ...
                         'Criterion', 'aic', 'Verbose', 0);

fprintf('Selected features for cement model:\n');
disp(cement_mdl.Formula);
fprintf('Selected features for binder model:\n');
disp(binder_mdl.Formula);

%% Step 3: Ensemble Modeling for Improved Robustness
% Create an ensemble of models with different polynomial degrees
ensemble_size = 5;
cement_ensemble = cell(ensemble_size, 1);
binder_ensemble = cell(ensemble_size, 1);

% Create training folds for ensemble diversity
cv_ensemble = cvpartition(sum(train_idx), 'KFold', ensemble_size);

for i = 1:ensemble_size
    % Get fold indices
    fold_train = cv_ensemble.training(i);
    
    % Create different polynomial degrees for diversity
    poly_degree = min(2 + i, 4); % Polynomials from degree 3 to 6
    
    % Train models on subset with different degrees
    cement_ensemble{i} = fitrensemble(train_data(fold_train, :), 'Comp_str_ln', ...
                                     'Method', 'LSBoost', 'NumLearningCycles', 100, ...
                                     'Learners', templateTree('MaxNumSplits', 2^poly_degree));
    
    binder_ensemble{i} = fitrensemble(train_data(fold_train, :), 'Comp_str_ln', ...
                                     'Method', 'LSBoost', 'NumLearningCycles', 100, ...
                                     'Learners', templateTree('MaxNumSplits', 2^poly_degree));
    
    fprintf('Trained ensemble model %d with polynomial degree %d\n', i, poly_degree);
end

%% Step 4: Model Prediction and Combination
% Function to make ensemble predictions
function preds = ensemble_predict(models, data)
    preds = zeros(height(data), length(models));
    for i = 1:length(models)
        preds(:, i) = predict(models{i}, data);
    end
    preds = mean(preds, 2); % Average predictions
end

% Make predictions using both stepwise and ensemble models
train_pred_cem_step = predict(cement_mdl, train_data);
test_pred_cem_step = predict(cement_mdl, data(test_idx, :));
train_pred_binder_step = predict(binder_mdl, train_data);
test_pred_binder_step = predict(binder_mdl, data(test_idx, :));

train_pred_cem_ens = ensemble_predict(cement_ensemble, train_data);
test_pred_cem_ens = ensemble_predict(cement_ensemble, data(test_idx, :));
train_pred_binder_ens = ensemble_predict(binder_ensemble, train_data);
test_pred_binder_ens = ensemble_predict(binder_ensemble, data(test_idx, :));

% Weighted average of stepwise and ensemble models
weight_step = 0.4;
weight_ens = 0.6;

train_pred_cem_ln = weight_step * train_pred_cem_step + weight_ens * train_pred_cem_ens;
test_pred_cem_ln = weight_step * test_pred_cem_step + weight_ens * test_pred_cem_ens;
train_pred_binder_ln = weight_step * train_pred_binder_step + weight_ens * train_pred_binder_ens;
test_pred_binder_ln = weight_step * test_pred_binder_step + weight_ens * test_pred_binder_ens;

%% Step 5: Dynamic Age-Specific Model Blending
% For each age group, calculate optimal weight between models
unique_test_ages = unique(data.Age(test_idx));

for i = 1:length(unique_test_ages)
    age = unique_test_ages(i);
    age_idx = data.Age(test_idx) == age;
    
    if sum(age_idx) >= 5 % Only analyze if enough samples
        % Calculate errors for both methods
        err_step = (test_pred_cem_step(age_idx) - data.Comp_str_ln(test_idx(age_idx))).^2;
        err_ens = (test_pred_cem_ens(age_idx) - data.Comp_str_ln(test_idx(age_idx))).^2;
        
        % Optimal weight based on inverse error proportion
        w_step = mean(err_ens) / (mean(err_step) + mean(err_ens));
        w_ens = 1 - w_step;
        
        fprintf('Age %d days: Optimal weights - Stepwise: %.2f, Ensemble: %.2f\n', ...
                age, w_step, w_ens);
    end
end

%% Step 6: Material Science-Specific Error Analysis
% Calculate transformed predictions
train_pred_cem = exp(train_pred_cem_ln);
test_pred_cem = exp(test_pred_cem_ln);
train_pred_binder = exp(train_pred_binder_ln);
test_pred_binder = exp(test_pred_binder_ln);

% Calculate relative errors (important for construction applications)
rel_err_cem = abs(test_pred_cem - data.Comp_strength(test_idx)) ./ data.Comp_strength(test_idx);
rel_err_binder = abs(test_pred_binder - data.Comp_strength(test_idx)) ./ data.Comp_strength(test_idx);

% Group by age and strength class
strength_bins = [0, 20, 40, 60, 80, 100];
strength_classes = discretize(data.Comp_strength(test_idx), strength_bins);

fprintf('\nRelative Error Analysis by Strength Class:\n');
fprintf('Strength Range (MPa) | Cement Model (%%) | Binder Model (%%)\n');
for i = 1:(length(strength_bins)-1)
    bin_idx = strength_classes == i;
    if sum(bin_idx) > 0
        fprintf('%d-%d MPa | %.2f%% | %.2f%%\n', ...
                strength_bins(i), strength_bins(i+1), ...
                mean(rel_err_cem(bin_idx))*100, ...
                mean(rel_err_binder(bin_idx))*100);
    end
end

%% Step 7: Final Performance Metrics
% Calculate final R² metrics
calc_r2 = @(y_true, y_pred) 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2);

% Calculate R² values for all model variants
r2_metrics = struct();
r2_metrics.step_train_cem_ln = calc_r2(data.Comp_str_ln(train_idx), train_pred_cem_step);
r2_metrics.step_test_cem_ln = calc_r2(data.Comp_str_ln(test_idx), test_pred_cem_step);
r2_metrics.ens_train_cem_ln = calc_r2(data.Comp_str_ln(train_idx), train_pred_cem_ens);
r2_metrics.ens_test_cem_ln = calc_r2(data.Comp_str_ln(test_idx), test_pred_cem_ens);
r2_metrics.combined_train_cem_ln = calc_r2(data.Comp_str_ln(train_idx), train_pred_cem_ln);
r2_metrics.combined_test_cem_ln = calc_r2(data.Comp_str_ln(test_idx), test_pred_cem_ln);
r2_metrics.combined_train_cem = calc_r2(data.Comp_strength(train_idx), train_pred_cem);
r2_metrics.combined_test_cem = calc_r2(data.Comp_strength(test_idx), test_pred_cem);

% Repeat for binder models
r2_metrics.step_train_binder_ln = calc_r2(data.Comp_str_ln(train_idx), train_pred_binder_step);
r2_metrics.step_test_binder_ln = calc_r2(data.Comp_str_ln(test_idx), test_pred_binder_step);
r2_metrics.ens_train_binder_ln = calc_r2(data.Comp_str_ln(train_idx), train_pred_binder_ens);
r2_metrics.ens_test_binder_ln = calc_r2(data.Comp_str_ln(test_idx), test_pred_binder_ens);
r2_metrics.combined_train_binder_ln = calc_r2(data.Comp_str_ln(train_idx), train_pred_binder_ln);
r2_metrics.combined_test_binder_ln = calc_r2(data.Comp_str_ln(test_idx), test_pred_binder_ln);
r2_metrics.combined_train_binder = calc_r2(data.Comp_strength(train_idx), train_pred_binder);
r2_metrics.combined_test_binder = calc_r2(data.Comp_strength(test_idx), test_pred_binder);

% Display final results
fprintf('\nFinal Performance Metrics:\n');
fprintf('Method                | Cement Case         | Binder Case\n');
fprintf('                      | Train R²   Test R²  | Train R²   Test R²\n');
fprintf('Stepwise (log scale)  | %.4f    %.4f  | %.4f    %.4f\n', ...
        r2_metrics.step_train_cem_ln, r2_metrics.step_test_cem_ln, ...
        r2_metrics.step_train_binder_ln, r2_metrics.step_test_binder_ln);
fprintf('Ensemble (log scale)  | %.4f    %.4f  | %.4f    %.4f\n', ...
        r2_metrics.ens_train_cem_ln, r2_metrics.ens_test_cem_ln, ...
        r2_metrics.ens_train_binder_ln, r2_metrics.ens_test_binder_ln);
fprintf('Combined (log scale)  | %.4f    %.4f  | %.4f    %.4f\n', ...
        r2_metrics.combined_train_cem_ln, r2_metrics.combined_test_cem_ln, ...
        r2_metrics.combined_train_binder_ln, r2_metrics.combined_test_binder_ln);
fprintf('Combined (raw units)  | %.4f    %.4f  | %.4f    %.4f\n', ...
        r2_metrics.combined_train_cem, r2_metrics.combined_test_cem, ...
        r2_metrics.combined_train_binder, r2_metrics.combined_test_binder);
