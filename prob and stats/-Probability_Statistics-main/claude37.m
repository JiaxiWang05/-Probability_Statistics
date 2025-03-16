%% Improved Data Splitting (Stratified by Age)
% Split data randomly within each age group (80% train, 20% test)
rng(42); % For reproducibility
train_idx = false(height(data), 1);
test_idx = false(height(data), 1);

for age = unique_ages'
    age_group = data.Age == age;
    num_samples = sum(age_group);
    
    if num_samples > 5 % Ensure minimum samples per group
        cv = cvpartition(num_samples, 'HoldOut', 0.2);
        train_idx(age_group) = cv.training;
        test_idx(age_group) = cv.test;
    else
        train_idx(age_group) = true; % Small groups go entirely to training
    end
end

%% Improved Feature Engineering
% Create interaction terms with log(Age)
data.log_age = log(data.Age);
data.wc_cem_int = data.wc_cem .* data.log_age;
data.wc_binder_int = data.wc_binder .* data.log_age;

%% Combined Model Approach (Single Stage Regression)
% Cement case model: log(Strength) ~ wc_cem + log_age + wc_cem*log_age
X_cem_train = [data.wc_cem(train_idx), data.log_age(train_idx), data.wc_cem_int(train_idx)];
X_cem_test = [data.wc_cem(test_idx), data.log_age(test_idx), data.wc_cem_int(test_idx)];

% Binder case model: log(Strength) ~ wc_binder + log_age + wc_binder*log_age
X_bind_train = [data.wc_binder(train_idx), data.log_age(train_idx), data.wc_binder_int(train_idx)];
X_bind_test = [data.wc_binder(test_idx), data.log_age(test_idx), data.wc_binder_int(test_idx)];

Y_train = data.Comp_str_ln(train_idx);
Y_test = data.Comp_str_ln(test_idx);

% Train models with regularization to prevent overfitting
cem_mdl = fitrlinear(X_cem_train, Y_train, 'Lambda', 0.1, 'Learner', 'leastsquares');
bind_mdl = fitrlinear(X_bind_train, Y_train, 'Lambda', 0.1, 'Learner', 'leastsquares');

%% Cross-Validated Performance Check
cvp = cvpartition(sum(train_idx), 'KFold', 5);
cem_cv_r2 = 1 - crossval('mse', X_cem_train, Y_train, 'Predfun', ...
    @(xtrain, ytrain, xtest) predict(fitrlinear(xtrain, ytrain, 'Lambda', 0.1), xtest), ...
    'partition', cvp) / var(Y_train);

bind_cv_r2 = 1 - crossval('mse', X_bind_train, Y_train, 'Predfun', ...
    @(xtrain, ytrain, xtest) predict(fitrlinear(xtrain, ytrain, 'Lambda', 0.1), xtest), ...
    'partition', cvp) / var(Y_train);

fprintf('Cross-validated R²:\nCement case: %.3f\nBinder case: %.3f\n', mean(cem_cv_r2), mean(bind_cv_r2));

%% Improved Prediction and Evaluation
% Make predictions
train_pred_cem_ln = predict(cem_mdl, X_cem_train);
test_pred_cem_ln = predict(cem_mdl, X_cem_test);
train_pred_bind_ln = predict(bind_mdl, X_bind_train);
test_pred_bind_ln = predict(bind_mdl, X_bind_test);

% Convert to original scale
train_pred_cem = exp(train_pred_cem_ln);
test_pred_cem = exp(test_pred_cem_ln);
train_pred_bind = exp(train_pred_bind_ln);
test_pred_bind = exp(test_pred_bind_ln);

% Calculate R² with proper train/test separation
r2_train_cem = 1 - sum((data.Comp_strength(train_idx) - train_pred_cem).^2)/...
    sum((data.Comp_strength(train_idx) - mean(data.Comp_strength(train_idx))).^2);

r2_test_cem = 1 - sum((data.Comp_strength(test_idx) - test_pred_cem).^2)/...
    sum((data.Comp_strength(test_idx) - mean(data.Comp_strength(test_idx))).^2);

r2_train_bind = 1 - sum((data.Comp_strength(train_idx) - train_pred_bind).^2)/...
    sum((data.Comp_strength(train_idx) - mean(data.Comp_strength(train_idx))).^2);

r2_test_bind = 1 - sum((data.Comp_strength(test_idx) - test_pred_bind).^2)/...
    sum((data.Comp_strength(test_idx) - mean(data.Comp_strength(test_idx))).^2);

%% Visual Diagnostics
figure;
subplot(1,2,1);
scatter(data.Comp_strength(test_idx), test_pred_cem, 'filled');
hold on; plot([0 100], [0 100], 'r--');
xlabel('Actual Strength'); ylabel('Predicted Strength');
title('Cement Case: Test Set Performance');
axis equal; grid on;

subplot(1,2,2);
scatter(data.Comp_strength(test_idx), test_pred_bind, 'filled');
hold on; plot([0 100], [0 100], 'r--');
xlabel('Actual Strength'); ylabel('Predicted Strength'); 
title('Binder Case: Test Set Performance');
axis equal; grid on;

%% Residual Analysis
figure;
subplot(1,2,1);
plotResiduals(cem_mdl, 'caseorder', 'Histogram');
title('Cement Case Residuals');

subplot(1,2,2);
plotResiduals(bind_mdl, 'caseorder', 'Histogram');
title('Binder Case Residuals');
