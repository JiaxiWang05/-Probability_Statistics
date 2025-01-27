% Step 1: Load and split data
data = readtable('Concrete_Data.csv');
uniqueAges = unique(data.Age);
trainMask = data.Age > 50; % Assuming Age is a variable; correct logic:
% Correct splitting:
trainMask = false(height(data), 1);
for i = 1:length(uniqueAges)
    age = uniqueAges(i);
    count = sum(data.Age == age);
    if count > 50
        trainMask(data.Age == age) = true;
    end
end
testMask = ~trainMask;
fprintf('Unique Ages: %d\nTraining Ages: %d\nTesting Ages: %d\n', ...
    length(uniqueAges), length(unique(data.Age(trainMask))), length(unique(data.Age(testMask))));

% Step 2: Transform data
Comp_str_ln = log(data.Comp_strength);
wc_cem = data.Water ./ data.Cement;
wc_binder = data.Water ./ (data.Cement + data.Slag + data.Ash);

% Step 3: Regression per age in training
trainAges = unique(data.Age(trainMask));
b0_cem = []; b1_cem = []; b0_bind = []; b1_bind = []; logAges = [];
for age = trainAges'
    idx = data.Age == age & trainMask;
    X_cem = wc_cem(idx);
    X_bind = wc_binder(idx);
    y = Comp_str_ln(idx);
    
    % Case 1: Cement only
    X = [ones(sum(idx),1), X_cem];
    b = X \ y;
    b0_cem = [b0_cem; b(1)];
    b1_cem = [b1_cem; b(2)];
    
    % Case 2: Composite binder
    X = [ones(sum(idx),1), X_bind];
    b = X \ y;
    b0_bind = [b0_bind; b(1)];
    b1_bind = [b1_bind; b(2)];
    
    logAges = [logAges; log(age)];
end

% Plots
figure;
subplot(2,2,1);
plot(logAges, b0_cem, 'bo'); title('b0 vs log(Age) - Cement'); xlabel('log(Age)'); ylabel('b0');
subplot(2,2,2);
plot(logAges, b1_cem, 'ro'); title('b1 vs log(Age) - Cement'); xlabel('log(Age)'); ylabel('b1');
subplot(2,2,3);
plot(logAges, b0_bind, 'bo'); title('b0 vs log(Age) - Binder'); xlabel('log(Age)'); ylabel('b0');
subplot(2,2,4);
plot(logAges, b1_bind, 'ro'); title('b1 vs log(Age) - Binder'); xlabel('log(Age)'); ylabel('b1');
