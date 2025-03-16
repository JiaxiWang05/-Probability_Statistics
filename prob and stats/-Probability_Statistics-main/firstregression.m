% Load the data
data = readtable('Concrete_Data.csv');

% Identify unique ages and their sample counts
ages = unique(data.Age);
numSamples = histcounts(data.Age, [ages; Inf]); % Count samples per age

% Split into training/testing based on sample count threshold (>=50)
trainingAges = ages(numSamples >= 50);
testingAges = ages(numSamples < 50);

% Create training and testing datasets
trainData = data(ismember(data.Age, trainingAges), :);
testData = data(ismember(data.Age, testingAges), :);

% Report findings
fprintf('Unique Ages: %d\n', length(ages));
fprintf('Training Ages: %d (with >=50 samples)\n', length(trainingAges));
fprintf('Testing Ages: %d (with <50 samples)\n', length(testingAges));