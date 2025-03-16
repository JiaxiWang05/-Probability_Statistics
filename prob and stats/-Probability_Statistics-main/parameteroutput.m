%% Wind Turbine SCADA Analysis
% Comprehensive analysis of turbine power curves with statistical validation

%% Configuration Parameters
BIN_WIDTH = 1;              % 1 m/s bin size
CUT_OUT_SPEED = 25;         % Turbine cut-out speed
CONFIDENCE_LEVEL = 0.95;    % For confidence intervals
MIN_SAMPLES_PER_BIN = 5;    % Minimum samples for valid bin

%% Main Analysis Pipeline
% Load and validate data
try
    load('turbine.mat');
    validate_data(u_A, P_A, u_B, P_B);
catch ME
    error('Data loading failed: %s', ME.message);
end

% Generate required figures
plot_power_curves(u_A, P_A, u_B, P_B);
[results_A, results_B] = binned_analysis(u_A, P_A, u_B, P_B, ...
                            BIN_WIDTH, CUT_OUT_SPEED, CONFIDENCE_LEVEL, ...
                            MIN_SAMPLES_PER_BIN);
                        
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

function plot_power_curves(u_A, P_A, u_B, P_B)
    % Create professional-quality figure
    fig = figure('Position', [100 100 1200 600], 'Color', 'white');
    
    % Custom color scheme (colorblind-friendly)
    colors = {[0.0 0.447 0.741], [0.85 0.325 0.098]};
    
    % Dataset A
    subplot(1,2,1);
    scatter(u_A, P_A, 10, 'filled', 'MarkerFaceColor', colors{1}, 'MarkerFaceAlpha', 0.3);
    hold on;
    
    % Add Betz curve to both plots consistently
    theoretical_speed = linspace(0, 25, 100);
    theoretical_power = min(0.5*1.225*pi*(82^2)/4*theoretical_speed.^3*0.593/6, 300);
    plot(theoretical_speed, theoretical_power, 'k--', 'LineWidth', 1.2);
    
    xlabel('Wind Speed (m/s)', 'FontWeight', 'bold');
    ylabel('Energy Production (kWh/10min)', 'FontWeight', 'bold');
    title('Dataset A: Reference Period', 'FontSize', 12, 'FontWeight', 'bold');
    grid on; box on;
    axis([0 25 0 350]);
    text(2, 320, 'Optimal Operation', 'FontSize', 10, 'Color', colors{1});
    legend('Measured Data', 'Theoretical Limit (Betz)', 'Location', 'northwest');
    
    % Dataset B
    subplot(1,2,2);
    scatter(u_B, P_B, 10, 'filled', 'MarkerFaceColor', colors{2}, 'MarkerFaceAlpha', 0.3);
    hold on;
    plot(theoretical_speed, theoretical_power, 'k--', 'LineWidth', 1.2);
    
    xlabel('Wind Speed (m/s)', 'FontWeight', 'bold');
    ylabel('Energy Production (kWh/10min)', 'FontWeight', 'bold');
    title('Dataset B: Evaluation Period', 'FontSize', 12, 'FontWeight', 'bold');
    grid on; box on;
    axis([0 25 0 350]);
    text(2, 320, 'Degraded Performance', 'FontSize', 10, 'Color', colors{2});
    legend('Measured Data', 'Theoretical Limit (Betz)', 'Location', 'northwest');
    
    % Highlight key features
    subplot(1,2,2);
    % Highlight cluster at ~100 kWh
    rectangle('Position', [10, 80, 10, 40], 'Curvature', [0.2, 0.2], ...
              'EdgeColor', colors{2}, 'LineStyle', ':', 'LineWidth', 1.5);
    text(21, 100, 'Anomaly Cluster', 'Color', colors{2});
    
    % Add annotation to compare graphics
    annotation('textbox', [0.35, 0.02, 0.3, 0.05], 'String', ...
               '', ...
               'HorizontalAlignment', 'center', 'EdgeColor', 'none', ...
               'FontSize', 10, 'FontWeight', 'bold');
    
    % Export as high-resolution figure
    set(gcf, 'PaperPositionMode', 'auto');
    print('-dpng', '-r300', 'Figure1_PowerCurvesComparison.png');
end

function [results_A, results_B] = binned_analysis(u_A, P_A, u_B, P_B, ...
                                     bin_width, cut_out, ci_level, min_samples)
    % Perform binned statistical analysis
    bin_edges = 0:bin_width:cut_out;
    z_score = norminv(1 - (1 - ci_level)/2);
    
    % Process both datasets
    results_A = process_dataset(u_A, P_A, bin_edges, z_score, min_samples);
    results_B = process_dataset(u_B, P_B, bin_edges, z_score, min_samples);
    
    % Visualize results with enhanced formatting
    figure('Position', [100 100 900 600], 'Color', 'white');

    % Color scheme
    colors = {[0.0 0.447 0.741], [0.85 0.325 0.098]};
    area_alpha = 0.15;

    % Create shaded error regions first (behind the lines)
    x_A = results_A.mean_speed;
    y_A = results_A.mean_power;
    ci_A = results_A.ci;

    x_B = results_B.mean_speed;
    y_B = results_B.mean_power;
    ci_B = results_B.ci;

    % Fill confidence intervals
    fill_x = [x_A; flipud(x_A)];
    fill_y = [y_A+ci_A; flipud(y_A-ci_A)];
    fill_x = fill_x(~isnan(fill_y));
    fill_y = fill_y(~isnan(fill_y));
    fill(fill_x, fill_y, colors{1}, 'FaceAlpha', area_alpha, 'EdgeColor', 'none');
    hold on;

    fill_x = [x_B; flipud(x_B)];
    fill_y = [y_B+ci_B; flipud(y_B-ci_B)];
    fill_x = fill_x(~isnan(fill_y));
    fill_y = fill_y(~isnan(fill_y));
    fill(fill_x, fill_y, colors{2}, 'FaceAlpha', area_alpha, 'EdgeColor', 'none');

    % Draw main lines with markers
    plot(x_A, y_A, 'o-', 'Color', colors{1}, 'MarkerFaceColor', colors{1}, ...
         'LineWidth', 2, 'MarkerSize', 6);
    plot(x_B, y_B, 's-', 'Color', colors{2}, 'MarkerFaceColor', colors{2}, ...
         'LineWidth', 2, 'MarkerSize', 6);

    % Add standard errorbar plots for better visibility of confidence intervals
    errorbar(x_A, y_A, ci_A, '.', 'Color', colors{1}, 'LineWidth', 1.2);
    errorbar(x_B, y_B, ci_B, '.', 'Color', colors{2}, 'LineWidth', 1.2);

    % Draw critical annotation: significant CI difference at 15 m/s
    idx_A = find(round(x_A) == 15);
    idx_B = find(round(x_B) == 15);
    if ~isempty(idx_A) && ~isempty(idx_B)
        plot([15, 15], [y_A(idx_A)-ci_A(idx_A
