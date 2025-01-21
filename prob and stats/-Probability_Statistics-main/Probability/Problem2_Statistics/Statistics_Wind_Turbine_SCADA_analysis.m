% Wind Turbine SCADA Data Analysis
clear all; close all;

% Load the data
load turbine.mat

%u_A and u_B: Wind speeds (in m/s) for datasets A and B.
%    P_A and P_B: Corresponding energy production (in kWh/10min).
%   These vectors are loaded into memory for analysis.

% Section 1: Data Summary and Scatter Plots
figure('Name', 'Power Curves Comparison');
%Visualize the relationship between wind speed (u) and energy production (P) for both datasets.

% Add data filtering before plotting
% Remove outliers or invalid data points
valid_A = P_A > 0 & P_A < 350;  % reasonable power range
valid_B = P_B > 0 & P_B < 350;

% Dataset A scatter plot
subplot(1,2,1);
scatter(u_A(valid_A), P_A(valid_A), 20, 'b', '.');
title('Power Curve - Dataset A');
xlabel('Wind Speed (m/s)');
ylabel('Energy Production (kWh/10min)');
grid on;
xlim([0 25]);
ylim([0 max(P_A)*1.1]);

%A scatter plot is created for Dataset A:
%    Wind speed (u_A) is plotted on the x-axis.
%    Energy production (P_A) is plotted on the y-axis.
%    scatter uses blue dots to represent data points.
%   xlim and ylim ensure the plot is limited to wind speeds between 0–25 m/s and a y-axis slightly above the maximum energy production.

% Dataset B scatter plot
subplot(1,2,2);
scatter(u_B(valid_B), P_B(valid_B), 20, 'r', '.');
title('Power Curve - Dataset B');
xlabel('Wind Speed (m/s)');
ylabel('Energy Production (kWh/10min)');
grid on;
xlim([0 25]);
ylim([0 max(P_B)*1.1]);

%    The same logic applies to Dataset B, but the data points are shown in red.

%% Section 2: Binned Analysis
%Compute and analyze statistics (mean, standard deviation, and confidence intervals) for energy production in 1 m/s wind speed bins.
%   Wind speeds are divided into 1 m/s bins from 0 to 25 m/s.
%   Arrays (mean_wind_A, mean_power_A, etc.) are initialized to store statistics for each bin.
% Define bins
bin_edges = 0:1:25;
num_bins = length(bin_edges) - 1;

% Initialize arrays for results
mean_wind_A = zeros(num_bins, 1);
mean_power_A = zeros(num_bins, 1);
std_power_A = zeros(num_bins, 1);
ci_power_A = zeros(num_bins, 1);
n_samples_A = zeros(num_bins, 1);

mean_wind_B = zeros(num_bins, 1);
mean_power_B = zeros(num_bins, 1);
std_power_B = zeros(num_bins, 1);
ci_power_B = zeros(num_bins, 1);
n_samples_B = zeros(num_bins, 1);

% For each bin calculation
for i = 1:num_bins
    u_low = bin_edges(i);
    u_upper = bin_edges(i+1);
    
    % Dataset A - using the exact method from hints
    u_list_A = (u_low <= u_A) & (u_A < u_upper);
    u_bin_A = u_A(u_list_A);
    P_bin_A = P_A(u_list_A);
    
    n_samples_A(i) = length(u_bin_A);
    if n_samples_A(i) > 0
        mean_wind_A(i) = mean(u_bin_A);
        mean_power_A(i) = mean(P_bin_A);
        std_power_A(i) = std(P_bin_A);
        ci_power_A(i) = norminv(0.975) * std_power_A(i) / sqrt(n_samples_A(i));
    else
        mean_wind_A(i) = NaN;
        mean_power_A(i) = NaN;
        std_power_A(i) = NaN;
        ci_power_A(i) = NaN;
    end
    
    % Dataset B - using the exact method from hints
    u_list_B = (u_low <= u_B) & (u_B < u_upper);
    u_bin_B = u_B(u_list_B);
    P_bin_B = P_B(u_list_B);
    
    n_samples_B(i) = length(u_bin_B);
    if n_samples_B(i) > 0
        mean_wind_B(i) = mean(u_bin_B);
        mean_power_B(i) = mean(P_bin_B);
        std_power_B(i) = std(P_bin_B);
        ci_power_B(i) = norminv(0.975) * std_power_B(i) / sqrt(n_samples_B(i));
    else
        mean_wind_B(i) = NaN;
        mean_power_B(i) = NaN;
        std_power_B(i) = NaN;
        ci_power_B(i) = NaN;
    end
end

% Plot binned analysis
figure('Name', 'Binned Analysis');
valid_bins_A = ~isnan(mean_power_A);
valid_bins_B = ~isnan(mean_power_B);

errorbar(mean_wind_A(valid_bins_A), mean_power_A(valid_bins_A), ...
    ci_power_A(valid_bins_A), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
hold on;
errorbar(mean_wind_B(valid_bins_B), mean_power_B(valid_bins_B), ...
    ci_power_B(valid_bins_B), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 6);

xlabel('Wind Speed (m/s)');
ylabel('Mean Energy Production (kWh/10min)');
title('Binned Power Curves with 95% Confidence Intervals');
legend('Dataset A', 'Dataset B', 'Location', 'northwest');
grid on;
xlim([0 25]);
ylim([0 max([max(mean_power_A); max(mean_power_B)])*1.1]);

% Display summary statistics
fprintf('Dataset A Summary:\n');
fprintf('Number of samples: %d\n', length(u_A));
fprintf('Wind speed range: %.1f to %.1f m/s\n', min(u_A), max(u_A));
fprintf('Mean energy production: %.2f kWh/10min\n\n', mean(P_A));

fprintf('Dataset B Summary:\n');
fprintf('Number of samples: %d\n', length(u_B));
fprintf('Wind speed range: %.1f to %.1f m/s\n', min(u_B), max(u_B));
fprintf('Mean energy production: %.2f kWh/10min\n', mean(P_B));
%errorbar plots mean energy production against mean wind speed for each bin, with error bars representing 95% confidence intervals.
%  Dataset A is plotted in blue, and Dataset B is plotted in red

%% Section 3: Power Curve Characteristics Analysis
% Find approximate rated power (where curve flattens)
[rated_power_A, rated_idx_A] = max(mean_power_A);
rated_speed_A = mean_wind_A(rated_idx_A);

[rated_power_B, rated_idx_B] = max(mean_power_B);
rated_speed_B = mean_wind_B(rated_idx_B);

fprintf('\nPower Curve Characteristics:\n');
fprintf('Dataset A - Approximate rated power: %.2f kWh/10min at %.1f m/s\n', ...
    rated_power_A, rated_speed_A);
fprintf('Dataset B - Approximate rated power: %.2f kWh/10min at %.1f m/s\n', ...
    rated_power_B, rated_speed_B);

% Analyze cubic region (before rated power)
% This will help identify the characteristic cubic curve mentioned in the assignment

%% Section 4: Statistical Comparison
% Calculate overall differences between datasets
mean_diff = mean(mean_power_B - mean_power_A);
max_diff = max(abs(mean_power_B - mean_power_A));

fprintf('\nDataset Comparison:\n');
fprintf('Mean difference (B-A): %.2f kWh/10min\n', mean_diff);
fprintf('Maximum absolute difference: %.2f kWh/10min\n', max_diff);

% Find regions of significant differences (where confidence intervals don't overlap)
significant_diff = abs(mean_power_B - mean_power_A) > (ci_power_A + ci_power_B);
sig_wind_speeds = mean_wind_A(significant_diff);

if ~isempty(sig_wind_speeds)
    fprintf('Significant differences found at wind speeds: ');
    fprintf('%.1f ', sig_wind_speeds);
    fprintf('m/s\n');
end
