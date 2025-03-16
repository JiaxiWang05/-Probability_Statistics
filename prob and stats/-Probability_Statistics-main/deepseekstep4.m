% Step 4: Second regressions
% ==========================

% Design matrix for second regression (log(Age))
X_second = [ones(length(logAges),1), logAges];

% Cement case: Regress b0_cem and b1_cem against log(Age)
gamma_b0_cem = X_second \ b0_cem;
gamma_b1_cem = X_second \ b1_cem;

% Binder case: Regress b0_bind and b1_bind against log(Age)
gamma_b0_bind = X_second \ b0_bind;
gamma_b1_bind = X_second \ b1_bind;

age_14 = 14;
log_age_14 = log(age_14);

% Predicted parameters
predicted_b0_cem = gamma_b0_cem(1) + gamma_b0_cem(2) * log_age_14;
predicted_b1_cem = gamma_b1_cem(1) + gamma_b1_cem(2) * log_age_14;
predicted_b0_bind = gamma_b0_bind(1) + gamma_b0_bind(2) * log_age_14;
predicted_b1_bind = gamma_b1_bind(1) + gamma_b1_bind(2) * log_age_14;

% Get training data variables directly from trainData table
wc_cem_train = trainData.wc_cem;          % Already subsetted in Step 1
wc_binder_train = trainData.wc_binder;    % No need for additional masking
Comp_str_ln_train = trainData.Comp_str_ln;
train_Age = trainData.Age;

% Find Age=14 samples in TRAINING SET
idx_14_train = (train_Age == 14);
if ~any(idx_14_train)
    error('Age=14 not found in training set. Check Step 1 data splitting.');
end

% Cement case data
x_cem = wc_cem_train(idx_14_train);
y_ln = Comp_str_ln_train(idx_14_train);

% Original parameters (from Step 3 results)
age_idx = find(trainAges == 14);  % trainAges created in Step 3
original_b0_cem = b0_cem(age_idx);
original_b1_cem = b1_cem(age_idx);

% Plot cement case
figure;
scatter(x_cem, y_ln, 'k', 'filled'); hold on;
x_plot = linspace(min(x_cem), max(x_cem), 100);
plot(x_plot, original_b0_cem + original_b1_cem*x_plot, 'r', 'LineWidth', 2);
plot(x_plot, predicted_b0_cem + predicted_b1_cem*x_plot, 'g--', 'LineWidth', 2);
xlabel('Water:Cement Ratio');
ylabel('log(Compressive Strength)');
title('Cement Case - Age = 14 Days');
legend('Data', 'Original Model', 'Estimated Model');
hold off;

% Binder case data
x_bind = wc_binder_train(idx_14_train);
original_b0_bind = b0_bind(age_idx);
original_b1_bind = b1_bind(age_idx);

% Plot binder case
figure;
scatter(x_bind, y_ln, 'k', 'filled'); hold on;
x_plot = linspace(min(x_bind), max(x_bind), 100);
plot(x_plot, original_b0_bind + original_b1_bind*x_plot, 'r', 'LineWidth', 2);
plot(x_plot, predicted_b0_bind + predicted_b1_bind*x_plot, 'g--', 'LineWidth', 2);
xlabel('Water:Binder Ratio');
ylabel('log(Compressive Strength)');
title('Binder Case - Age = 14 Days');
legend('Data', 'Original Model', 'Estimated Model');
hold off;