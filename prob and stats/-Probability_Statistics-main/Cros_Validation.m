% Cross-validation for Cement case
cv = cvpartition(size(trainData, 1), 'KFold', 5);
mse_cem = zeros(cv.NumTestSets, 1);
for i = 1:cv.NumTestSets
    trainIdx = cv.training(i);
    testIdx = cv.test(i);
    
    % Train model on training fold
    mdl_cem = fitlm(wc_cem(trainIdx), Comp_str_ln(trainIdx));
    
    % Predict on test fold
    pred_ln_str_cem = predict(mdl_cem, wc_cem(testIdx));
    
    % Calculate MSE
    mse_cem(i) = mean((Comp_str_ln(testIdx) - pred_ln_str_cem).^2);
end
avg_mse_cem = mean(mse_cem);

% Cross-validation for Binder case
mse_binder = zeros(cv.NumTestSets, 1);
for i = 1:cv.NumTestSets
    trainIdx = cv.training(i);
    testIdx = cv.test(i);
    
    % Train model on training fold
    mdl_binder = fitlm(wc_binder(trainIdx), Comp_str_ln(trainIdx));
    
    % Predict on test fold
    pred_ln_str_binder = predict(mdl_binder, wc_binder(testIdx));
    
    % Calculate MSE
    mse_binder(i) = mean((Comp_str_ln(testIdx) - pred_ln_str_binder).^2);
end
avg_mse_binder = mean(mse_binder);

disp(['Average MSE for Cement case: ', num2str(avg_mse_cem)]);
disp(['Average MSE for Binder case: ', num2str(avg_mse_binder)]);