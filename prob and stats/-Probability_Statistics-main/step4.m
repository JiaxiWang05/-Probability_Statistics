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

% Predicted parameters for cement case
predicted_b0_cem = gamma_b0_cem(1) + gamma_b0_cem(2) * log_age_14;
predicted_b1_cem = gamma_b1_cem(1) + gamma_b1_cem(2) * log_age_14;

% Predicted parameters for binder case
predicted_b0_bind = gamma_b0_bind(1) + gamma_b0_bind(2) * log_age_14;
predicted_b1_bind = gamma_b1_bind(1) + gamma_b1_bind(2) * log_age_14;

% Extract data for Age = 14 from trainData
idx_14_train = (trainData.Age == 14);
x_cem = trainData.wc_cem(idx_14_train);
x_bind = trainData.wc_binder(idx_14_train);
y_ln = trainData.Comp_str_ln(idx_14_train);

% Original parameters for Age = 14
age_idx = find(trainAges == 14);
original_b0_cem = b0_cem(age_idx);
original_b1_cem = b1_cem(age_idx);

% Generate regression lines (cement case)
x_plot = linspace(min(x_cem), max(x_cem), 100);
y_original = original_b0_cem + original_b1_cem * x_plot;
y_estimated = predicted_b0_cem + predicted_b1_cem * x_plot;

% Plot cement case
figure;
scatter(x_cem, y_ln, 'k', 'filled'); hold on;
plot(x_plot, y_original, 'r', 'LineWidth', 2);
plot(x_plot, y_estimated, 'g--', 'LineWidth', 2);
xlabel('Water:Cement Ratio');
ylabel('log(Compressive Strength)');
title('Cement Case - Age = 14 Days');
legend('Data', 'Original Model', 'Estimated Model');
hold off;

% Binder case
original_b0_bind = b0_bind(age_idx);
original_b1_bind = b1_bind(age_idx);

% Generate regression lines (binder case)
x_plot_bind = linspace(min(x_bind), max(x_bind), 100);
y_original_bind = original_b0_bind + original_b1_bind * x_plot_bind;
y_estimated_bind = predicted_b0_bind + predicted_b1_bind * x_plot_bind;

% Plot binder case
figure;
scatter(x_bind, y_ln, 'k', 'filled'); hold on;
plot(x_plot_bind, y_original_bind, 'r', 'LineWidth', 2);
plot(x_plot_bind, y_estimated_bind, 'g--', 'LineWidth', 2);
xlabel('Water:Binder Ratio');
ylabel('log(Compressive Strength)');
title('Binder Case - Age = 14 Days');
legend('Data', 'Original Model', 'Estimated Model');
hold off;