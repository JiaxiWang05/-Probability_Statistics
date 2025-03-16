% --- Step 3: Perform First Regressions ---

% Use the training data from step1 and transformed data from step2
trainAges = unique(trainData.Age);

% Initialize arrays to store regression parameters
b0_cem = []; b1_cem = []; % For cement-only case
b0_bind = []; b1_bind = []; % For binder case
logAges = []; % To store log(Age) for plotting

% Loop over each unique age in the training set
for age = trainAges'
    % Filter data for the current age
    idx = (trainData.Age == age);
    
    % Use the transformed variables from step2
    X_cem = trainData.wc_cem(idx); % Water-to-cement ratio
    X_bind = trainData.wc_binder(idx); % Water-to-binder ratio
    y = trainData.Comp_str_ln(idx); % Log-transformed compressive strength
    
    % --- Case 1: Cement-only regression ---
    X = [ones(sum(idx), 1), X_cem]; % Design matrix with intercept
    b = X \ y; % Solve for coefficients using least squares
    b0_cem = [b0_cem; b(1)]; % Store intercept (b0)
    b1_cem = [b1_cem; b(2)]; % Store slope (b1)
    
    % --- Case 2: Binder regression ---
    X = [ones(sum(idx), 1), X_bind]; % Design matrix with intercept
    b = X \ y; % Solve for coefficients using least squares
    b0_bind = [b0_bind; b(1)]; % Store intercept (b0)
    b1_bind = [b1_bind; b(2)]; % Store slope (b1)
    
    % Store log(Age) for plotting
    logAges = [logAges; log(age)];
end

% --- Plot b0 and b1 vs log(Age) ---
figure;
subplot(2, 2, 1);
plot(logAges, b0_cem, 'bo-');
xlabel('log(Age)');
ylabel('b0');
title('Cement Case: b0 vs log(Age)');
grid on;

subplot(2, 2, 2);
plot(logAges, b1_cem, 'ro-');
xlabel('log(Age)');
ylabel('b1');
title('Cement Case: b1 vs log(Age)');
grid on;

subplot(2, 2, 3);
plot(logAges, b0_bind, 'bo-');
xlabel('log(Age)');
ylabel('b0');
title('Binder Case: b0 vs log(Age)');
grid on;

subplot(2, 2, 4);
plot(logAges, b1_bind, 'ro-');
xlabel('log(Age)');
ylabel('b1');
title('Binder Case: b1 vs log(Age)');
grid on;