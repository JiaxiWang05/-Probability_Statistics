%% Wind Turbine SCADA Analysis
% Comprehensive analysis of turbine power curves with statistical validation

%% Configuration Parameters
BIN_WIDTH = 1;              % 1 m/s bin size
CUT_OUT_SPEED = 25;         % Turbine cut-out speed
CONFIDENCE_LEVEL = 0.95;    % For confidence intervals
MIN_SAMPLES_PER_BIN = 5;    % Minimum samples for valid bin

% Define consistent color scheme
colors = struct(...
    'reference', [0 0.4470 0.7410],... % Blue
    'evaluation', [0.8500 0.3250 0.0980],... % Orange
    'theoretical', [0.5 0.5 0.5],... % Gray
    'annotation', [0.2 0.2 0.2]); % Dark gray

%% Main Analysis Pipeline
% Load and validate data
try
    load('turbine.mat');
    validate_data(u_A, P_A, u_B, P_B);
catch ME
    error('Data loading failed: %s', ME.message);
end

% Generate enhanced figures
plot_power_curves(u_A, P_A, u_B, P_B, colors);
[results_A, results_B] = binned_analysis(u_A, P_A, u_B, P_B, ...
                            BIN_WIDTH, CUT_OUT_SPEED, CONFIDENCE_LEVEL, ...
                            MIN_SAMPLES_PER_BIN, colors);
                        
% Display statistical summary
display_results_summary(results_A, results_B);

%% Helper Functions
function validate_data(u_A, P_A, u_B, P_B)
    % Data integrity checks
    assert(all(size(u_A) == size(P_A)), 'Dataset A size mismatch');
    assert(all(size(u_B) == size(P_B)), 'Dataset B size mismatch');
    assert(numel(u_A) == 5000, 'Dataset A incomplete');
    assert(numel(u_B) == 5000, 'Dataset B incomplete');
end

function plot_power_curves(u_A, P_A, u_B, P_B, colors)
    % Create professional-quality figure
    fig = figure('Position', [100 100 1200 600], 'Color', 'white', 'Renderer', 'painters');
    
    % Dataset A
    subplot(1,2,1);
    density_scatter(u_A, P_A, colors.reference);
    hold on;
    
    % Add Betz curve consistently
    theoretical_speed = linspace(0, 25, 100);
    theoretical_power = min(0.5*1.225*pi*(82^2)/4*theoretical_speed.^3*0.593/6, 300);
    plot(theoretical_speed, theoretical_power, '--', 'Color', colors.theoretical, ...
         'LineWidth', 2);
    
    xlabel('Wind Speed (m/s)', 'FontWeight', 'bold', 'FontSize', 12);
    ylabel('Energy Production (kWh/10min)', 'FontWeight', 'bold', 'FontSize', 12);
    title('Dataset A: Reference Period', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on;
    set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'GridAlpha', 0.3);
    axis([0 25 0 350]);
    text(2, 320, 'Optimal Operation', 'FontSize', 12, 'Color', colors.reference);
    legend({'Measured Data', 'Theoretical Limit (Betz)'}, 'Location', 'northwest', ...
           'FontSize', 11);
    
    % Dataset B with enhanced visualization
    subplot(1,2,2);
    density_scatter(u_B, P_B, colors.evaluation);
    hold on;
    plot(theoretical_speed, theoretical_power, '--', 'Color', colors.theoretical, ...
         'LineWidth', 2);
    
    xlabel('Wind Speed (m/s)', 'FontWeight', 'bold', 'FontSize', 12);
    ylabel('Energy Production (kWh/10min)', 'FontWeight', 'bold', 'FontSize', 12);
    title('Dataset B: Evaluation Period', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; box on;
    set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'GridAlpha', 0.3);
    axis([0 25 0 350]);
    text(2, 320, 'Degraded Performance', 'FontSize', 12, 'Color', colors.evaluation);
    legend({'Measured Data', 'Theoretical Limit (Betz)'}, 'Location', 'northwest', ...
           'FontSize', 11);
    
    % Highlight anomaly cluster with improved styling
    rectangle('Position', [10, 80, 10, 40], 'Curvature', [0.2, 0.2], ...
              'EdgeColor', colors.evaluation, 'LineStyle', ':', 'LineWidth', 2);
    text(21, 100, 'Anomaly Cluster', 'Color', colors.evaluation, 'FontSize', 11);
    
    % Add professional annotation
    annotation('textbox', [0.35, 0.02, 0.3, 0.05], 'String', ...
               'Figure 1: Comparative Power Curves with Betz Limit Reference', ...
               'HorizontalAlignment', 'center', 'EdgeColor', 'none', ...
               'FontSize', 12, 'FontWeight', 'bold');
    
    % Export as high-resolution figure
    exportgraphics(gcf, 'Figure1_PowerCurvesComparison.png', 'Resolution', 600);
end

function density_scatter(x, y, color)
    % Create enhanced density-based scatter plot
    [density, ~, ~] = histcounts2(x, y, 50);
    
    % Add bounds checking before using sub2ind
    % Find indices where points fall within grid bounds
    valid_idx = row_idx >= 1 & row_idx <= size(density, 1) & ...
                col_idx >= 1 & col_idx <= size(density, 2);
    
    % Only use valid indices for sub2ind
    if any(valid_idx)
        density(sub2ind(size(density), row_idx(valid_idx), col_idx(valid_idx))) = ...
            density(sub2ind(size(density), row_idx(valid_idx), col_idx(valid_idx))) + 1;
    end
    
    % Alternative approach if above doesn't work:
    % for i = 1:numel(row_idx)
    %     if row_idx(i) >= 1 && row_idx(i) <= size(density,1) && ...
    %        col_idx(i) >= 1 && col_idx(i) <= size(density,2)
    %         density(row_idx(i), col_idx(i)) = density(row_idx(i), col_idx(i)) + 1;
    %     end
    % end
    
    % Scale point sizes and transparency
    size_factor = 20 * (1 + density/max(density));
    alpha_factor = 0.2 + 0.6 * density/max(density);
    
    % Draw points with enhanced styling
    scatter(x, y, size_factor, 'filled', ...
            'MarkerFaceColor', color, ...
            'MarkerFaceAlpha', 'flat', ...
            'AlphaData', alpha_factor);
end

function [results_A, results_B] = binned_analysis(u_A, P_A, u_B, P_B, ...
                                     bin_width, cut_out, ci_level, min_samples, colors)
    % Perform binned statistical analysis
    bin_edges = 0:bin_width:cut_out;
    z_score = norminv(1 - (1 - ci_level)/2);
    
    % Process both datasets
    results_A = process_dataset(u_A, P_A, bin_edges, z_score, min_samples);
    results_B = process_dataset(u_B, P_B, bin_edges, z_score, min_samples);
    
    % Create enhanced visualization
    figure('Position', [100, 100, 1000, 400]);
    
    % First plot - notice the very minimal gap with the second plot
    subplot('Position', [0.05, 0.12, 0.44, 0.78]);  
    % First residual plot code here
    title('Cement Model Residuals', 'FontSize', 12);
    % Make sure x-axis label is concise or removed if possible
    
    % Second plot - positioned immediately after the first
    subplot('Position', [0.51, 0.12, 0.44, 0.78]);  % Almost touching first plot
    % Second residual plot code here
    title('Binder Model Residuals', 'FontSize', 12);
    
    % Add region annotations with enhanced styling
    add_region_annotations(cut_out);
    
    % Export as high-resolution figure
    exportgraphics(gcf, 'Figure2_BinnedAnalysis.png', 'Resolution', 600);
end

function plot_confidence_intervals(results_A, results_B, colors)
    % Plot confidence intervals with enhanced styling
    area_alpha = 0.2;
    
    % Dataset A confidence intervals
    fill_confidence_interval(results_A, colors.reference, area_alpha);
    
    % Dataset B confidence intervals
    fill_confidence_interval(results_B, colors.evaluation, area_alpha);
    
    % Add main lines and error bars
    plot_dataset_lines(results_A, results_B, colors);
    
    % Enhance visualization
    enhance_plot_styling();
end

function fill_confidence_interval(results, color, alpha)
    x = results.mean_speed;
    y = results.mean_power;
    ci = results.ci;
    
    fill_x = [x; flipud(x)];
    fill_y = [y+ci; flipud(y-ci)];
    valid_idx = ~isnan(fill_y);
    fill(fill_x(valid_idx), fill_y(valid_idx), color, ...
         'FaceAlpha', alpha, 'EdgeColor', 'none');
    hold on;
end

function plot_dataset_lines(results_A, results_B, colors)
    % Plot main lines with enhanced markers
    plot(results_A.mean_speed, results_A.mean_power, 'o-', ...
         'Color', colors.reference, 'MarkerFaceColor', colors.reference, ...
         'LineWidth', 2.5, 'MarkerSize', 8);
    plot(results_B.mean_speed, results_B.mean_power, 's-', ...
         'Color', colors.evaluation, 'MarkerFaceColor', colors.evaluation, ...
         'LineWidth', 2.5, 'MarkerSize', 8);
    
    % Add error bars
    errorbar(results_A.mean_speed, results_A.mean_power, results_A.ci, '.', ...
            'Color', colors.reference, 'LineWidth', 1.5);
    errorbar(results_B.mean_speed, results_B.mean_power, results_B.ci, '.', ...
            'Color', colors.evaluation, 'LineWidth', 1.5);
end

function enhance_plot_styling()
    grid on; box on;
    set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'GridAlpha', 0.3);
    xlabel('Wind Speed (m/s)', 'FontWeight', 'bold', 'FontSize', 14);
    ylabel('Mean Energy Production (kWh/10min)', 'FontWeight', 'bold', 'FontSize', 14);
    title('Binned Power Curve Analysis with 95% Confidence Intervals', ...
          'FontWeight', 'bold', 'FontSize', 16);
    legend({'CI Region A', 'CI Region B', 'Dataset A', 'Dataset B'}, ...
           'Location', 'northwest', 'FontSize', 12);
end

function add_region_annotations(cut_out)
    % Add region annotations with enhanced styling
    text(5, 50, 'Cubic Region', 'FontSize', 11, 'FontWeight', 'bold');
    text(15, 290, 'Rated Power Region', 'FontSize', 11, 'FontWeight', 'bold');
    text(22, 150, 'Cut-out Region', 'FontSize', 11, 'FontWeight', 'bold');
    xlim([0 cut_out]);
end

function results = process_dataset(u, P, bin_edges, z_score, min_samples)
    % Process individual dataset
    n_bins = length(bin_edges) - 1;
    results = struct(...
        'mean_speed', nan(n_bins,1), ...
        'mean_power', nan(n_bins,1), ...
        'std_power', nan(n_bins,1), ...
        'ci', nan(n_bins,1), ...
        'count', zeros(n_bins,1));
    
    for b = 1:n_bins
        mask = (u >= bin_edges(b)) & (u < bin_edges(b+1));
        if sum(mask) < min_samples, continue; end
        
        bin_u = u(mask);
        bin_P = P(mask);
        
        results.mean_speed(b) = mean(bin_u);
        results.mean_power(b) = mean(bin_P);
        results.std_power(b) = std(bin_P);
        results.ci(b) = z_score * results.std_power(b)/sqrt(sum(mask));
        results.count(b) = sum(mask);
    end
end

function display_results_summary(results_A, results_B)
    % Display enhanced statistical summary
    fprintf('\n=== Dataset Comparison Summary ===\n');
    fprintf('%20s %12s %12s %12s\n', 'Metric', 'Dataset A', 'Dataset B', 'Difference');
    
    metrics = {
        'Mean Power (all)' mean(results_A.mean_power, 'omitnan') mean(results_B.mean_power, 'omitnan')
        'Max Power' max(results_A.mean_power) max(results_B.mean_power)
        'CI Width @15m/s' results_A.ci(15) results_B.ci(15)
        'Valid Bins' sum(~isnan(results_A.mean_power)) sum(~isnan(results_B.mean_power))
        };
    
    for m = 1:size(metrics,1)
        diff = metrics{m,3} - metrics{m,2};
        fprintf('%20s %12.2f %12.2f %12.2f\n', metrics{m,1}, metrics{m,2}, metrics{m,3}, diff);
    end
end
