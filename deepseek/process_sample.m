% BinnedAnalysiswithConfidenceIntervals.m

% ------ MAIN SCRIPT ------
% 1. Data loading
load('turbine.mat');

% 2. Preprocessing
[u_clean_A, P_clean_A] = preprocess_data(u_A, P_A);
[u_clean_B, P_clean_B] = preprocess_data(u_B, P_B);

% 3. Analysis
bin_edges = 0:1:25;
ci_width = norminv(0.975);
[mean_u_A, mean_power_A, ci_power_A] = process_sample(u_clean_A, P_clean_A, bin_edges, ci_width);
[mean_u_B, mean_power_B, ci_power_B] = process_sample(u_clean_B, P_clean_B, bin_edges, ci_width);

% 4. Visualization
plot_power_curves(mean_u_A, mean_power_A, ci_power_A, mean_u_B, mean_power_B, ci_power_B);

% 5. Validation
run_data_quality_checks(u_clean_A, P_clean_A);
run_data_quality_checks(u_clean_B, P_clean_B);

% ------ FUNCTIONS ------
function [mean_u, mean_P, ci] = process_sample(u, P, bins, z_score)
    % Validate inputs
    if isempty(u) || isempty(P) || length(u) ~= length(P)
        error('Input vectors u and P must be non-empty and of the same length.');
    end
    
    % Initialize outputs with NaN
    mean_u = nan(length(bins)-1, 1);
    mean_P = nan(length(bins)-1, 1);
    ci = nan(length(bins)-1, 1);
    
    for i = 1:length(bins)-1
        mask = (u >= bins(i)) & (u < bins(i+1));
        if sum(mask) < 3  % Minimum data threshold
            continue;  % Skip this bin if not enough data
        end
        mean_u(i) = mean(u(mask));
        mean_P(i) = mean(P(mask));
        ci(i) = z_score * std(P(mask)) / sqrt(sum(mask)); % Confidence interval
    end
end

function [u_clean, P_clean] = preprocess_data(u, P)
    % Remove invalid entries
    u_clean = u(~isnan(u) & ~isinf(u));
    P_clean = P(~isnan(P) & ~isinf(P));
    
    % Advanced outlier removal
    [P_clean, TF] = rmoutliers(P_clean, 'movmedian', 30); % 30-sample moving median
    u_clean = u_clean(~TF);
    fprintf('Removed %d outliers from dataset\n', sum(TF));
end

function plot_power_curves(mean_u_A, mean_power_A, ci_power_A, mean_u_B, mean_power_B, ci_power_B)
    % Plot results
    figure;
    errorbar(mean_u_A, mean_power_A, ci_power_A, 'bo', 'MarkerFaceColor', 'b');
    hold on;
    errorbar(mean_u_B, mean_power_B, ci_power_B, 'rs', 'MarkerFaceColor', 'r');
    xlabel('Wind Speed (m/s)');
    ylabel('Mean Energy (kWh/10min)');
    legend('Sample A', 'Sample B');
    title('Binned Power Curves with 95% CI');
end

function run_data_quality_checks(u, P)
    % Check data quality
    valid_samples = sum(~isnan(P) & ~isinf(P));
    fprintf('Valid samples: %d/%d (%.1f%%)\n', valid_samples, length(P), 100 * valid_samples / length(P));
    assert(valid_samples > 0, 'No valid samples found in the dataset.');
end
