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

% Dataset A scatter plot
subplot(1,2,1);
scatter(u_A, P_A, 20, 'b', '.');
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
scatter(u_B, P_B, 20, 'r', '.');
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
bin_edges = 0:1:25; % 1 m/s wide bins from 0 to 25 m/s
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

% Calculate statistics for each bin
for i = 1:num_bins
    % Define bin range
    u_low = bin_edges(i);
    u_upper = bin_edges(i+1);
    
%For each bin, the range is defined by u_low (lower edge) and u_upper (upper edge).
    % Dataset A
    u_list_A = (u_low <= u_A) & (u_A < u_upper);
    u_bin_A = u_A(u_list_A);
    P_bin_A = P_A(u_list_A);
    
    n_samples_A(i) = length(u_bin_A);
    if n_samples_A(i) > 0
        mean_wind_A(i) = mean(u_bin_A);
        mean_power_A(i) = mean(P_bin_A);
        std_power_A(i) = std(P_bin_A);
        % Calculate 95% confidence interval
        ci_power_A(i) = norminv(0.975) * std_power_A(i) / sqrt(n_samples_A(i));
    end
    
    % Dataset B
    u_list_B = (u_low <= u_B) & (u_B < u_upper);
    u_bin_B = u_B(u_list_B);
    P_bin_B = P_B(u_list_B);
    
    n_samples_B(i) = length(u_bin_B);
    if n_samples_B(i) > 0
        mean_wind_B(i) = mean(u_bin_B);
        mean_power_B(i) = mean(P_bin_B);
        std_power_B(i) = std(P_bin_B);
        % Calculate 95% confidence interval
        ci_power_B(i) = norminv(0.975) * std_power_B(i) / sqrt(n_samples_B(i));
    end
end

% Plot binned data with confidence intervals
figure('Name', 'Binned Analysis');
errorbar(mean_wind_A, mean_power_A, ci_power_A, 'b-o', 'LineWidth', 1.5);
hold on;
errorbar(mean_wind_B, mean_power_B, ci_power_B, 'r-o', 'LineWidth', 1.5);
xlabel('Wind Speed (m/s)');
ylabel('Mean Energy Production (kWh/10min)');
title('Binned Power Curves with 95% Confidence Intervals');
legend('Dataset A', 'Dataset B');
grid on;
xlim([0 25]);
ylim([0 max([mean_power_A; mean_power_B])*1.1]);

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
