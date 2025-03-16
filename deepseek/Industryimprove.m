%% Wind Turbine SCADA Analysis Lab Script
% Comprehensive performance monitoring with predictive maintenance features
% IEC 61400-25 compliant | Version 2.1 | March 2025

%% Configuration
BIN_WIDTH = 1;              % 1 m/s bin size
CUT_OUT_SPEED = 25;         % Turbine cut-out speed
CONFIDENCE_LEVEL = 0.95;    % Confidence level for CIs
MIN_SAMPLES_PER_BIN = 5;    % Minimum samples per bin
ALARM_THRESHOLD = 0.01;     % Downtime threshold (1%)
SYS_EFFICIENCY_TARGET = 0.42; % Betz-adjusted target

%% Main Pipeline
try
    % Initialize resource management
    cleanupObj = onCleanup(@() system_cleanup());
    
    % Load and validate data
    [scada_data, alarms] = load_scada_data('turbine.mat');
    validate_scada_structure(scada_data);
    
    % Data preprocessing
    scada_data = preprocess_data(scada_data, ALARM_THRESHOLD);
    
    % Core analysis
    [health_stats, perf_stats] = analyze_operation(scada_data, alarms, ...
                                  BIN_WIDTH, CUT_OUT_SPEED, ...
                                  CONFIDENCE_LEVEL, MIN_SAMPLES_PER_BIN);
    
    % Predictive maintenance
    fault_prob = train_fault_model(scada_data, alarms);
    
    % Visualization
    generate_analysis_figures(scada_data, perf_stats, health_stats);
    
    % Reporting
    generate_pdf_report(health_stats, perf_stats, fault_prob);
    
catch ME
    handle_scada_error(ME);
    rethrow(ME);
end

%% Data Handling Functions
function [data, alarms] = load_scada_data(filename)
    % Load SCADA data structure
    try
        raw = load(filename);
        data = struct2table(raw.scada);
        alarms = parse_alarm_logs(raw.alarms);
        
        % IEC 61400-25 compliance check
        assert(isfield(raw, 'metadata'), 'SCADA:MissingMetadata');
        validate_metadata(raw.metadata);
        
    catch loadError
        error('SCADA:DataLoad', 'Failed to load data: %s', loadError.message);
    end
end

function data = preprocess_data(data, alarm_thresh)
    % Data quality pipeline
    data = filter_operational_modes(data);
    data = handle_missing_values(data);
    data = remove_sensor_outliers(data);
    data = flag_alarm_periods(data, alarm_thresh);
    data = calculate_derived_metrics(data);
end

%% Analysis Functions
function [health, perf] = analyze_operation(data, alarms, bin_width, ...
                                           cut_out, ci_level, min_samples)
    % Performance analysis
    [perf.bins, perf.curves] = analyze_power_curves(data, bin_width, ...
                                  cut_out, ci_level, min_samples);
    
    % System health monitoring
    health.betz = calculate_betz_efficiency(data, perf.curves.theoretical);
    health.bearings = monitor_bearing_health(data.temp_main_bearing);
    health.gearbox = analyze_vibration(data.vib_gearbox_x, data.vib_gearbox_y);
    
    % Reliability metrics
    health.alarm_stats = calculate_alarm_impact(alarms, data.timestamp);
    health.availability = calculate_turbine_availability(data, alarms);
end

function fault_prob = train_fault_model(data, alarms)
    % Predictive maintenance model
    features = [data.wind_speed, data.power_output, data.temp_gearbox, ...
               data.vib_gearbox_x, data.bearing_temp];
    
    labels = create_predictive_labels(alarms, data.timestamp);
    
    model = fitcensemble(features, labels, 'Method', 'GentleBoost', ...
                        'Learners', 'tree', 'CrossVal', 'on', ...
                        'HyperparameterOptimizationOptions', struct('Verbose',0));
    
    fault_prob = kfoldPredict(model);
end

%% Visualization Functions
function generate_analysis_figures(data, perf, health)
    % Main analysis figure
    figure('Position', [100 100 1400 800], 'Name', 'SCADA Analysis Dashboard');
    
    % 3D Power Curve
    subplot(2,2,1);
    scatter3(data.wind_speed, data.wind_dir, data.power_output, 40, ...
            data.temp_ambient, 'filled');
    xlabel('Wind Speed (m/s)'); 
    ylabel('Wind Direction (°)');
    zlabel('Power (kW)');
    title('3D Power Characteristic');
    colorbar('Location', 'eastoutside');
    
    % Efficiency Analysis
    subplot(2,2,2);
    plot(perf.curves.mean_speed, health.betz, 'LineWidth', 2);
    yline(SYS_EFFICIENCY_TARGET, '--r', 'Betz Target');
    xlabel('Wind Speed (m/s)');
    ylabel('Efficiency Ratio');
    title('Betz Limit Efficiency');
    
    % Vibration Spectrum
    subplot(2,2,3);
    plot_vibration_spectrum(data.vib_gearbox_x, data.vib_gearbox_y);
    
    % Availability Timeline
    subplot(2,2,4);
    plot_availability_timeline(health.availability);
    
    % Save figure
    exportgraphics(gcf, 'scada_analysis.png', 'Resolution', 300);
end

%% Error Handling Functions
function handle_scada_error(ME)
    fprintf('\n=== SCADA ANALYSIS ERROR ===\n');
    fprintf('Timestamp: %s\n', datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    fprintf('Error ID: %s\n', ME.identifier);
    fprintf('Message: %s\n', ME.message);
    
    % Stack trace analysis
    fprintf('\nStack Trace:\n');
    for st = 1:length(ME.stack)
        fprintf('File: %s\nLine: %d\nComponent: %s\n\n', ...
               ME.stack(st).file, ...
               ME.stack(st).line, ...
               extract_subsystem(ME.stack(st).file));
    end
    
    % System cleanup
    system_cleanup();
    
    % Recovery suggestions
    suggest_recovery(ME.identifier);
end

function suggest_recovery(err_id)
    recovery_map = containers.Map(...
        {'SCADA:DataLoad', 'SCADA:MissingMetadata', 'MATLAB:UndefinedFunction'}, ...
        {'Check sensor connections', 'Validate metadata file', 'Verify function paths'}...
    );
    
    if isKey(recovery_map, err_id)
        fprintf('\nRecommended Action: %s\n', recovery_map(err_id));
    end
end

function system_cleanup()
    % Resource cleanup protocol
    fclose('all');
    delete(findall(0, 'Type', 'figure'));
    if exist('scada_data', 'var')
        clear scada_data;  % Added from search result 8
    end
end

%% Helper Functions
function subsystem = extract_subsystem(filepath)
    [~,name] = fileparts(filepath);
    component_map = containers.Map(...
        {'power_analysis', 'bearing', 'vibration'}, ...
        {'Power System', 'Main Bearing', 'Gearbox'}...
    );
    subsystem = 'General System';
    if isKey(component_map, name)
        subsystem = component_map(name);
    end
end

function plot_vibration_spectrum(vib_x, vib_y)
    % Vibration analysis plot
    Fs = 1000; % Sampling frequency
    n = length(vib_x);
    f = Fs*(0:(n/2))/n;
    
    Yx = fft(vib_x - mean(vib_x));
    Pyx = abs(Yx/n).^2;
    
    Yy = fft(vib_y - mean(vib_y));
    Pyy = abs(Yy/n).^2;
    
    plot(f(1:end-1), Pyx(1:n/2), 'b', f(1:end-1), Pyy(1:n/2), 'r');
    xlabel('Frequency (Hz)');
    ylabel('Power Spectrum');
    legend('X-axis', 'Y-axis');
    title('Gearbox Vibration Spectrum');
end
