function differencecode()
    % Load turbine data
    try
        load('turbine.mat');
        fprintf('Data loaded successfully\n');
    catch ME
        error('Failed to load turbine.mat: %s', ME.message);
    end
    
    % Process both datasets with binning
    [results_A, results_B] = process_datasets(u_A, P_A, u_B, P_B);
    
    % Create the difference plot
    plot_dataset_differences(results_A, results_B);
end

function [results_A, results_B] = process_datasets(u_A, P_A, u_B, P_B)
    % Process datasets with 1 m/s bins
    bin_edges = 0:1:25;
    z_score = norminv(0.975);  % 95% confidence interval
    min_samples = 5;
    
    % Process both datasets
    results_A = process_single_dataset(u_A, P_A, bin_edges, z_score, min_samples);
    results_B = process_single_dataset(u_B, P_B, bin_edges, z_score, min_samples);
end

function results = process_single_dataset(u, P, bin_edges, z_score, min_samples)
    % Process individual dataset
    n_bins = length(bin_edges) - 1;
    results = struct(...
        'mean_speed', nan(n_bins, 1), ...
        'mean_power', nan(n_bins, 1), ...
        'std_power', nan(n_bins, 1), ...
        'ci', nan(n_bins, 1), ...
        'count', zeros(n_bins, 1));
    
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

function plot_dataset_differences(results_A, results_B)
    % Create a plot showing differences between datasets A and B
    % highlighting where degradation is most significant
    
    % Create figure with appropriate size
    figure('Position', [100 100 900 500], 'Color', 'white');
    
    % Get wind speeds and calculate power differences
    wind_speeds = results_A.mean_speed;
    power_diff = results_A.mean_power - results_B.mean_power;
    
    % Calculate combined confidence intervals for statistical significance
    combined_ci = sqrt(results_A.ci.^2 + results_B.ci.^2);
    
    % Create bar chart with color coding
    bar_h = bar(wind_speeds, power_diff, 0.7);
    
    % Set all bars to blue for consistent appearance
    set(bar_h, 'FaceColor', [0.0 0.447 0.741]);
    
    % Add error bars for combined confidence intervals
    hold on;
    errorbar(wind_speeds, power_diff, combined_ci, '.k', 'LineWidth', 1);
    
    % Add reference line at zero
    plot([0 25], [0 0], 'k--', 'LineWidth', 1.5);
    
    % Highlight regions of most significant degradation
    [max_diff, max_idx] = max(abs(power_diff));
    if ~isnan(max_diff)
        highlight_x = wind_speeds(max_idx);
        highlight_y = power_diff(max_idx);
        
        plot(highlight_x, highlight_y, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    end
    
    % Add annotations for key operational regions
    x_regions = [5, 12, 20];
    region_names = {'Cubic Region', 'Rated Power Transition', 'High Wind Region'};
    y_pos = min(power_diff) - 10;
    
    for i = 1:length(x_regions)
        text(x_regions(i), y_pos, region_names{i}, 'FontSize', 11, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontWeight', 'bold');
    end
    
 
 
     % Enhance visualization
    grid on; box on;
    set(gca, 'FontSize', 11, 'LineWidth', 1);
    xlabel('Wind Speed (m/s)', 'FontWeight', 'bold', 'FontSize', 12);
    ylabel('Power Difference: A - B (kWh/10min)', 'FontWeight', 'bold', 'FontSize', 12);
    title('Performance Degradation Analysis: Dataset A vs Dataset B', ...
          'FontWeight', 'bold', 'FontSize', 14);
    
    % Add explanatory legend
    legend('Power Difference (A-B)', '95% Confidence Interval', 'Zero Reference', ...
           'Maximum Degradation Point', 'Location', 'southeast', 'FontSize', 10);
    
    % Set axis limits
    xlim([0 25]);
    ylim([-100 200]);  % Based on the image
    
    % Export as high-resolution figure
    set(gcf, 'PaperPositionMode', 'auto');
    print('-dpng', '-r300', 'Figure3_PerformanceDegradation.png');
    
    fprintf('Plot created and saved as Figure3_PerformanceDegradation.png\n');
end
