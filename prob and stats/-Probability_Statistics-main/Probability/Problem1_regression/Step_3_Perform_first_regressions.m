% Load data
data = readtable('Concrete_Data.csv');

% Step 2: Transform the data
Comp_str_ln = log(data.Comp_strength);
wc_cem = data.Water ./ data.Cement;
wc_binder = data.Water ./ sum([data.Cement data.Slag data.Ash],2);

% Step 3: First regressions
% Find unique ages and initialize parameter arrays
ages = unique(data.Age);
b0_cem = zeros(length(ages), 1);
b1_cem = zeros(length(ages), 1);
b0_binder = zeros(length(ages), 1);
b1_binder = zeros(length(ages), 1);

% Perform regression for each age
for i = 1:length(ages)
    % Create index for current age
    idx = (data.Age == ages(i));
    
    % Regression for cement ratio
    X_cem = [ones(sum(idx), 1), wc_cem(idx)];
    y = Comp_str_ln(idx);
    b_cem = X_cem \ y;
    b0_cem(i) = b_cem(1);
    b1_cem(i) = b_cem(2);
    
    % Regression for binder ratio
    X_binder = [ones(sum(idx), 1), wc_binder(idx)];
    b_binder = X_binder \ y;
    b0_binder(i) = b_binder(1);
    b1_binder(i) = b_binder(2);
end

% Plot results
figure(1)
subplot(2,2,1)
plot(log(ages), b0_cem, 'o-')
xlabel('log(Age)')
ylabel('b0 (cement)')
title('b0 vs log(Age) - Cement')
grid on

subplot(2,2,2)
plot(log(ages), b1_cem, 'o-')
xlabel('log(Age)')
ylabel('b1 (cement)')
title('b1 vs log(Age) - Cement')
grid on

subplot(2,2,3)
plot(log(ages), b0_binder, 'o-')
xlabel('log(Age)')
ylabel('b0 (binder)')
title('b0 vs log(Age) - Binder')
grid on

subplot(2,2,4)
plot(log(ages), b1_binder, 'o-')
xlabel('log(Age)')
ylabel('b1 (binder)')
title('b1 vs log(Age) - Binder')
grid on