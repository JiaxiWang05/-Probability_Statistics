% Step 1: Load and analyze data
data = readtable('Concrete_Data.csv');

% Find unique ages and count samples for each age
ages = unique(data.Age);
numSamples = histc(data.Age, ages);

% Select ages with 50+ samples for training
trainingAges = ages(numSamples >= 50);

% Split data into training and testing sets
trainingIdx = ismember(data.Age, trainingAges);
trainData = data(trainingIdx, :);
testData = data(~trainingIdx, :);
