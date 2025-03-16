%% Wind Turbine SCADA Analysis with Machine Learning Integration
 

%% Configuration
TEST_SIZE = 0.3;           % Validation set size
N_TREES = 100;             % Random Forest hyperparameter
MAX_EPOCHS = 50;           % Neural Network hyperparameter
FEATURES = {'WindSpeed', 'WindSpeed^3', 'Turbulence', 'HourOfDay'};

%% Load and Preprocess Data
data = load('turbine.mat');

% Create feature-rich dataset
datasetA = create_ml_dataset(data.u_A, data.P_A, data.time_A);
datasetB = create_ml_dataset(data.u_B, data.P_B, data.time_B);

% Split datasets
[A_train, A_test] = train_test_split(datasetA, TEST_SIZE);
[B_train, B_test] = train_test_split(datasetB, TEST_SIZE);

%% Machine Learning Pipeline
% Train models on Dataset A (reference)
rf_model = train_random_forest(A_train);
nn_model = train_neural_net(A_train, MAX_EPOCHS);

% Validate models
A_test = predict_power(rf_model, A_test);
A_test = predict_power(nn_model, A_test);

% Analyze Dataset B using trained models
B_test = predict_power(rf_model, B_test);
B_test = predict_power(nn_model, B_test);

%% Enhanced Binned Analysis with ML
% Original coursework analysis
[results_A, results_B] = binned_analysis(data.u_A, data.P_A, data.u_B, data.P_B);

% ML-enhanced analysis
ml_bins = 0:1:25;
ml_results_A = ml_binned_analysis(A_test, ml_bins);
ml_results_B = ml_binned_analysis(B_test, ml_bins);

%% Visualization
% Plot comparison figures
plot_ml_comparison(results_A, ml_results_A, 'Dataset A');
plot_ml_comparison(results_B, ml_results_B, 'Dataset B');

% Display performance metrics
display_ml_metrics(A_test, B_test);

%% Helper Functions
function ds = create_ml_dataset(u, P, time)
    % Create feature-rich table with temporal features
    hours = hour(datetime(time, 'ConvertFrom', 'datenum'));
    turbulence = movstd(u, 30); % 5-hour window
    
    ds = table(u, P, turbulence, hours, ...
        'VariableNames', {'WindSpeed', 'Power', 'Turbulence', 'HourOfDay'});
    ds.WindSpeedCubed = ds.WindSpeed.^3;
end

function [train, test] = train_test_split(dataset, test_size)
    % Split dataset preserving temporal order
    split_idx = floor(height(dataset)*(1-test_size));
    train = dataset(1:split_idx,:);
    test = dataset(split_idx+1:end,:);
end

function model = train_random_forest(train)
    % Train Random Forest model
    model = TreeBagger(N_TREES, train{:,FEATURES}, train.Power, ...
        'Method', 'regression', 'OOBPrediction', 'on');
end

function net = train_neural_net(train, max_epochs)
    % Train Neural Network model
    layers = [featureInputLayer(numel(FEATURES))
              fullyConnectedLayer(50)
              reluLayer
              fullyConnectedLayer(25)
              reluLayer
              fullyConnectedLayer(1)
              regressionLayer];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', max_epochs, ...
        'ValidationData', {train{:,FEATURES}', train.Power'});
    
    net = trainNetwork(train{:,FEATURES}', train.Power', layers, options);
end

function ds = predict_power(model, ds)
    % Add predictions to dataset
    if isa(model, 'TreeBagger')
        ds.RFPred = predict(model, ds{:,FEATURES});
    else
        ds.NNPred = predict(model, ds{:,FEATURES}')';
    end
end

function results = ml_binned_analysis(ds, bins)
    % ML-enhanced binned analysis
    results = struct();
    for b = 1:length(bins)-1
        mask = (ds.WindSpeed >= bins(b)) & (ds.WindSpeed < bins(b+1));
        if sum(mask) < 5, continue; end
        
        results.mean_speed(b) = mean(ds.WindSpeed(mask));
        results.actual_power(b) = mean(ds.Power(mask));
        results.rf_power(b) = mean(ds.RFPred(mask));
        results.nn_power(b) = mean(ds.NNPred(mask));
        results.residuals(b) = results.actual_power(b) - results.rf_power(b);
    end
end

function plot_ml_comparison(orig, ml, title_str)
    % Plot ML vs traditional analysis
    figure;
    errorbar(orig.mean_speed, orig.mean_power, orig.ci, 'b', 'LineWidth', 1.5);
    hold on;
    plot(ml.mean_speed, ml.rf_power, 'g--', 'LineWidth', 2);
    plot(ml.mean_speed, ml.nn_power, 'm-.', 'LineWidth', 2);
    
    xlabel('Wind Speed (m/s)');
    ylabel('Power (kWh/10min)');
    title([title_str ' Power Curve Comparison']);
    legend('Actual', 'Random Forest', 'Neural Network');
    grid on;
end

function display_ml_metrics(A, B)
    % Calculate and display performance metrics
    metrics = {
        'RMSE (RF)'     rmse(A.RFPred, A.Power) rmse(B.RFPred, B.Power)
        'RMSE (NN)'     rmse(A.NNPred, A.Power) rmse(B.NNPred, B.Power)
        'MAE (RF)'      mae(A.RFPred, A.Power)  mae(B.RFPred, B.Power)
        'R² (RF)'       rsquared(A.RFPred, A.Power) rsquared(B.RFPred, B.Power)
        };
    
    fprintf('\n=== Machine Learning Performance Metrics ===\n');
    fprintf('%15s %12s %12s\n', 'Metric', 'Dataset A', 'Dataset B');
    for m = 1:size(metrics,1)
        fprintf('%15s %12.2f %12.2f\n', metrics{m,:});
    end
end

function val = rmse(pred, actual)
    val = sqrt(mean((pred - actual).^2));
end

function val = mae(pred, actual)
    val = mean(abs(pred - actual));
end

function val = rsquared(pred, actual)
    val = 1 - sum((actual - pred).^2)/sum((actual - mean(actual)).^2);
end
