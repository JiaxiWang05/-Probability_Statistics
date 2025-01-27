% For the cement case
% Transformed data residuals
trainData.residual_cem_transformed = trainData.Comp_str_ln - (predicted_b0_cem + predicted_b1_cem * trainData.wc_cem);
testData.residual_cem_transformed = testData.Comp_str_ln - (predicted_b0_cem + predicted_b1_cem * testData.wc_cem);

% Raw data residuals
trainData.residual_cem_raw = trainData.Comp_strength - exp(predicted_b0_cem + predicted_b1_cem * trainData.wc_cem);
testData.residual_cem_raw = testData.Comp_strength - exp(predicted_b0_cem + predicted_b1_cem * testData.wc_cem);

% For the binder case
% Transformed data residuals
trainData.residual_binder_transformed = trainData.Comp_str_ln - (predicted_b0_bind + predicted_b1_bind * trainData.wc_binder);
testData.residual_binder_transformed = testData.Comp_str_ln - (predicted_b0_bind + predicted_b1_bind * testData.wc_binder);

% Raw data residuals
trainData.residual_binder_raw = trainData.Comp_strength - exp(predicted_b0_bind + predicted_b1_bind * trainData.wc_binder);
testData.residual_binder_raw = testData.Comp_strength - exp(predicted_b0_bind + predicted_b1_bind * testData.wc_binder);

% Cement case
figure;
subplot(2, 1, 1);
histogram(trainData.residual_cem_raw, 'Normalization', 'pdf', 'FaceColor', 'b');
hold on;
histogram(testData.residual_cem_raw, 'Normalization', 'pdf', 'FaceColor', 'r');
xlabel('Residuals (Raw Data)');
ylabel('Density');
title('Cement Case: Residual Density (Raw Data)');
legend('Training Data', 'Testing Data');
hold off;

subplot(2, 1, 2);
histogram(trainData.residual_cem_transformed, 'Normalization', 'pdf', 'FaceColor', 'b');
hold on;
histogram(testData.residual_cem_transformed, 'Normalization', 'pdf', 'FaceColor', 'r');
xlabel('Residuals (Transformed Data)');
ylabel('Density');
title('Cement Case: Residual Density (Transformed Data)');
legend('Training Data', 'Testing Data');
hold off;

% Binder case
figure;
subplot(2, 1, 1);
histogram(trainData.residual_binder_raw, 'Normalization', 'pdf', 'FaceColor', 'b');
hold on;
histogram(testData.residual_binder_raw, 'Normalization', 'pdf', 'FaceColor', 'r');
xlabel('Residuals (Raw Data)');
ylabel('Density');
title('Binder Case: Residual Density (Raw Data)');
legend('Training Data', 'Testing Data');
hold off;

subplot(2, 1, 2);
histogram(trainData.residual_binder_transformed, 'Normalization', 'pdf', 'FaceColor', 'b');
hold on;
histogram(testData.residual_binder_transformed, 'Normalization', 'pdf', 'FaceColor', 'r');
xlabel('Residuals (Transformed Data)');
ylabel('Density');
title('Binder Case: Residual Density (Transformed Data)');
legend('Training Data', 'Testing Data');
hold off;