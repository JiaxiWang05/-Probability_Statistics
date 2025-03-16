% Load data
load('turbine.mat');

% Remove invalid entries first
u_A = u_A(~isnan(u_A) & ~isinf(u_A));
P_A = P_A(~isnan(P_A) & ~isinf(P_A));

% Advanced outlier removal
[P_clean_A, TF_A] = rmoutliers(P_A, 'movmedian', 30); % 30-sample moving median
u_clean_A = u_A(~TF_A);
fprintf('Removed %d outliers from Dataset A\n', sum(TF_A));

% Define bin edges and confidence interval width
bin_edges = 0:1:25;
ci_width = norminv(0.975);

% Calculate mean power for Dataset A directly from cleaned data
mean_power_A_value = mean(P_clean_A, 'omitnan'); % Direct mean from cleaned data
fprintf('Mean power for Dataset A: %.2f kWh\n', mean_power_A_value);

% Wind speed distribution analysis
ws_bins = 0:3:25;
[N,~] = histcounts(u_clean_A, ws_bins);
fprintf('Data distribution in rated region (12-25m/s): %.1f%%\n',...
        100*sum(N(5:end))/sum(N));

% Tolerance adjustment based on IEC 61400-12
rated_hours = sum((u_clean_A >= 12) & (u_clean_A <= 25)) / 6; % Calculate rated hours
tolerance = 0.05 * 275 * rated_hours; % 5% of expected energy

% Update assertion
assert(abs(sum(P_clean_A) - 275 * rated_hours) < tolerance,...
       'AEP deviation exceeds 5% tolerance');

% Plot results
figure;
errorbar(mean_u_A, mean_power_A, ci_power_A, 'bo', 'MarkerFaceColor', 'b');
hold on;
errorbar(mean_u_B, mean_power_B, ci_power_B, 'rs', 'MarkerFaceColor', 'r');
xlabel('Wind Speed (m/s)');
ylabel('Mean Energy (kWh/10min)');
legend('Sample A', 'Sample B');
title('Binned Power Curves with 95% CI');

% Additional diagnostics
energy_weighted = sum(P_clean_A) / sum(u_clean_A); % Energy-weighted calculation
time_weighted = mean(P_clean_A); % Time-weighted calculation
fprintf('Energy-weighted: %.1f kWh\nTime-weighted: %.1f kWh\n',...
        energy_weighted, time_weighted);

% Test the process_sample function with random data
test_u = 10 * rand(1000, 1); % Random wind speeds 0-10 m/s
test_P = 200 * rand(1000, 1); % Random power values
[mu, mp, ci] = process_sample(test_u, test_P, 0:1:10, 1.96);

% Quality check for bin reliability
n_samples_A = sum(~isnan(mean_power_A));  % Count valid samples
assert(n_samples_A > 0, 'No valid samples in Dataset A');

% Example for checking significant differences
threshold = 10;  % Define threshold as needed
sig_wind_speeds = mean_u_A(mean_power_A > threshold);
assert(any(sig_wind_speeds >= 10 & sig_wind_speeds <= 20), 'Significant wind speeds not found');

% Sensitivity analysis
bin_widths = [0.5, 1, 2]; % Test different bin sizes
for bw = bin_widths
    edges = 0:bw:25;
    [~, P_mean] = process_sample(u_A, P_A, edges, norminv(0.975));
    fprintf('Bin %.1fm: Mean power=%.2f\n', bw, nanmean(P_mean)); 
end

% Outlier visualization
figure;
subplot(2,1,1);
histogram(P_A, 'BinWidth', 10);
hold on;
xline(max_power, 'r--', 'Threshold');
title('Dataset A Power Distribution');

subplot(2,1,2);
histogram(u_A, 'BinEdges', 0:1:30);
xline(25 + cutout_buffer, 'r--', 'Threshold');
title('Dataset A Wind Speed Distribution');

% Quantify extreme values
high_power = sum(P_A > max_power);
high_wind = sum(u_A > 25);
fprintf('Dataset A extremes: %d (%.1f%%) > %.1f kWh, %d (%.1f%%) > 25m/s\n', ...
    high_power, 100*high_power/numel(P_A), max_power, ...
    high_wind, 100*high_wind/numel(u_A));

% Function definitions should be at the end of the file
function [mean_u, mean_P, ci] = process_sample(u, P, bins, z_score)
    % Validate inputs
    if isempty(u) || isempty(P) || length(u) ~= length(P)
        error('Input vectors u and P must be non-empty and of the same length.');
    end
    
    % Initialize output variables with NaN
    num_bins = length(bins) - 1;
    mean_u = nan(num_bins, 1);
    mean_P = nan(num_bins, 1);
    ci = nan(num_bins, 1);
    
    % Calculate mean and confidence intervals for each bin
    for i = 1:num_bins
        mask = (u >= bins(i)) & (u < bins(i+1));
        if sum(mask) < 3  % Minimum data threshold
            continue;  % Skip this bin if not enough data
        end
        mean_u(i) = mean(u(mask));
        mean_P(i) = mean(P(mask));
        ci(i) = z_score * std(P(mask)) / sqrt(sum(mask)); % Confidence interval
    end
end

function checkDataQuality(u, P)
    % Check data quality
    valid_samples = sum(~isnan(P) & ~isinf(P));
    fprintf('Valid samples: %d/%d (%.1f%%)\n', valid_samples, length(P), 100 * valid_samples / length(P));
end

% Create a test script for process_sample
function testProcessSample
    % Define test cases and expected results
    % Implement assertions to validate functionality
end

% Create a benchmarking function
function benchmarkAgainstSpec(spec_file)
    % Load specifications and compare results
end
