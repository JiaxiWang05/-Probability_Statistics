%% Cement Data Analysis Script
% Revised version with modern MATLAB practices
% Performs exploratory analysis and regression modeling

%% Initialize
clear; close all; clc;

%% Load Data
try
    load('cement_data.mat'); % Original data file
catch
    % Create sample data if file not found
    cure_time = [1 1 1 2 2 2 3 3 3 3 3 7 7 7 7 7 28 28 28 28 28]';
    strength = [12.4 11.6 13.5 19.7 18.6 20.4 26.9 25.3 28.1 27.5 24.9 ...
               35.9 34.8 37.8 33.4 36.1 48.9 46.3 49.2 47.6 45.7]';
    disp('Using sample data as cement_data.mat not found');
end

%% Preprocessing
% Get unique cure times and counts
[cure_times, ~, idx] = unique(cure_time);
n_groups = length(cure_times);

% Create structure for grouped data
for i = 1:n_groups
    group_data(i).time = cure_times(i);
    group_data(i).strength = strength(idx == i);
    group_data(i).mean = mean(group_data(i).strength);
    group_data(i).std = std(group_data(i).strength);
    group_data(i).dstrength = group_data(i).strength - group_data(i).mean;
end

%% Visualization 1: Raw Data and Group Stats
figure('Color','white','Position',[100 100 1200 600])

% Raw data scatter plot
subplot(2,3,1)
scatter(cure_time, strength, 40, 'filled')
title('Raw Strength Data')
xlabel('Cure Time (days)')
ylabel('Compressive Strength (MPa)')
grid on

% Grouped means
subplot(2,3,2)
hold on
for i = 1:n_groups
    plot(group_data(i).time, group_data(i).mean, 'ro', 'MarkerSize', 8)
end
title('Group Means')
xlabel('Cure Time (days)')
ylabel('Mean Strength (MPa)')
grid on
xlim([0 max(cure_times)+1])

% Difference from group means
subplot(2,3,3)
hold on
for i = 1:n_groups
    scatter(repmat(group_data(i).time, size(group_data(i).dstrength)),...
            group_data(i).dstrength, 40, 'filled')
end
title('Deviation from Group Means')
xlabel('Cure Time (days)')
ylabel('Deviation (MPa)')
grid on

%% Transformation Analysis
% Log transform
log_strength = log(strength);
inv_ctime = 1./cure_time;

% Linear regression
X = [ones(size(inv_ctime)) inv_ctime];
b = X \ log_strength;
log_strength_hat = X*b;
residuals = log_strength - log_strength_hat;

%% Visualization 2: Transformed Data
subplot(2,3,4)
scatter(cure_time, log_strength, 40, 'filled')
hold on
[ct_sorted, sort_idx] = sort(cure_time);
plot(ct_sorted, log_strength_hat(sort_idx), 'r-', 'LineWidth', 2)
title('Log-Transformed Strength')
xlabel('Cure Time (days)')
ylabel('log(Strength)')
grid on

subplot(2,3,5)
scatter(inv_ctime, log_strength, 40, 'filled')
hold on
plot(inv_ctime(sort_idx), log_strength_hat(sort_idx), 'r-', 'LineWidth', 2)
title('Inverse Cure Time Relationship')
xlabel('1/Cure Time (days^{-1})')
ylabel('log(Strength)')
grid on

% Residual analysis
subplot(2,3,6)
scatter(inv_ctime, residuals, 40, 'filled')
hold on
yline(0, '--', 'LineWidth', 2)
title('Residual Analysis')
xlabel('1/Cure Time (days^{-1})')
ylabel('Residuals')
grid on

%% Statistical Summary
fprintf('\n=== Group Statistics ===\n')
fprintf('Cure Time |   Mean | Std Dev | Sample Size\n')
fprintf('-----------------------------------------\n')
for i = 1:n_groups
    fprintf('%6d    | %6.2f |  %5.2f  |     %d\n',...
            group_data(i).time, group_data(i).mean,...
            group_data(i).std, numel(group_data(i).strength))
end

fprintf('\n=== Regression Model ===\n')
fprintf('log(Strength) = %.2f + %.2f*(1/Cure Time)\n', b(1), b(2))
fprintf('R-squared: %.3f\n', 1 - var(residuals)/var(log_strength))

%% Additional Residual Diagnostics
figure('Color','white')
subplot(1,2,1)
histogram(residuals, 20, 'FaceColor', [0.2 0.4 0.8])
title('Residual Distribution')
xlabel('Residual Value')
ylabel('Frequency')

subplot(1,2,2)
qqplot(residuals)
title('Q-Q Plot of Residuals')
grid on
