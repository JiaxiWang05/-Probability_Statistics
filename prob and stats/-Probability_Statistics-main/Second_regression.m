% Step 1: Load and analyze data
data = readtable('Concrete_Data.csv');
ages = unique(data.Age);
numSamples = histc(data.Age, ages);
trainingAges = ages(numSamples >= 50);

% Step 2: Transform the data
Comp_str_ln = log(data.Comp_strength);
wc_cem = data.Water ./ data.Cement;
wc_binder = data.Water ./ sum([data.Cement data.Slag data.Ash],2);

% Step 3: First regression for each age
b0_cem = zeros(length(trainingAges), 1);
b1_cem = zeros(length(trainingAges), 1);
b0_binder = zeros(length(trainingAges), 1);
b1_binder = zeros(length(trainingAges), 1);

for i = 1:length(trainingAges)
    idx = (data.Age == trainingAges(i));
    
    % Cement regression
    X = [ones(sum(idx), 1), wc_cem(idx)];
    y = Comp_str_ln(idx);
    b = X \ y;
    b0_cem(i) = b(1);
    b1_cem(i) = b(2);
    
    % Binder regression
    X = [ones(sum(idx), 1), wc_binder(idx)];
    b = X \ y;
    b0_binder(i) = b(1);
    b1_binder(i) = b(2);
end

% Step 4: Second regression
X = [ones(length(trainingAges), 1), log(trainingAges)];

% Cement parameters regression
b0_params_cem = X \ b0_cem;
b1_params_cem = X \ b1_cem;

% Binder parameters regression
b0_params_binder = X \ b0_binder;
b1_params_binder = X \ b1_binder;

% Plot for Age = 14 days
figure(2)
idx_14 = (data.Age == 14);

% Cement plot
subplot(2,1,1)
plot(wc_cem(idx_14), Comp_str_ln(idx_14), 'b.', 'DisplayName', 'Data')
hold on
wc_range = linspace(min(wc_cem(idx_14)), max(wc_cem(idx_14)), 100)';
b0_pred = b0_params_cem(1) + b0_params_cem(2)*log(14);
b1_pred = b1_params_cem(1) + b1_params_cem(2)*log(14);
y_pred = b0_pred + b1_pred*wc_range;
plot(wc_range, y_pred, 'r-', 'DisplayName', 'Model')
xlabel('Water-Cement Ratio')
ylabel('ln(Compressive Strength)')
title('14-day Strength vs W/C Ratio')
legend('Location', 'best')
grid on
hold off

% Binder plot
subplot(2,1,2)
plot(wc_binder(idx_14), Comp_str_ln(idx_14), 'b.', 'DisplayName', 'Data')
hold on
wc_range = linspace(min(wc_binder(idx_14)), max(wc_binder(idx_14)), 100)';
b0_pred = b0_params_binder(1) + b0_params_binder(2)*log(14);
b1_pred = b1_params_binder(1) + b1_params_binder(2)*log(14);
y_pred = b0_pred + b1_pred*wc_range;
plot(wc_range, y_pred, 'r-', 'DisplayName', 'Model')
xlabel('Water-Binder Ratio')
ylabel('ln(Compressive Strength)')
title('14-day Strength vs W/B Ratio')
legend('Location', 'best')
grid on
hold off