 
% Step 2: Transform the data
% 1. Log-transform compressive strength
trainData.Comp_str_ln = log(trainData.Comp_strength);

% 2. Calculate water-to-cement ratio (wc_cem)
trainData.wc_cem = trainData.Water ./ trainData.Cement;

% 3. Calculate water-to-binder ratio (wc_binder)
trainData.wc_binder = trainData.Water ./ (trainData.Cement + trainData.Slag + trainData.Ash);

% Validate data to avoid division by zero or invalid log values
assert(all(trainData.Cement > 0), 'Cement has non-positive values!');
assert(all(trainData.Cement + trainData.Slag + trainData.Ash > 0), 'Binder sum has non-positive values!');
assert(all(trainData.Comp_strength > 0), 'Compressive strength has non-positive values!');

% Create two separate datasets for regression
% Dataset 1: Water-to-cement ratio (wc_cem)
dataset_cem = trainData(:, {'Age', 'Comp_str_ln', 'wc_cem'});

% Dataset 2: Water-to-binder ratio (wc_binder)
dataset_binder = trainData(:, {'Age', 'Comp_str_ln', 'wc_binder'});

% Display the first few rows of each dataset for verification
disp('Dataset 1: Water-to-Cement Ratio');
disp(head(dataset_cem));

disp('Dataset 2: Water-to-Binder Ratio');
disp(head(dataset_binder));

% Report findings
fprintf('Step 2: Data transformation complete.\n');
fprintf('Dataset 1 (Water:Cement) has %d samples.\n', height(dataset_cem));
fprintf('Dataset 2 (Water:Binder) has %d samples.\n', height(dataset_binder));