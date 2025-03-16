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
    
    % Determine statistical significance
    is_significant = abs(power_diff) > combined_ci;
    
    % Create bar chart with color coding
    bar_h = bar(wind_speeds, power_diff, 0.7);
    
    % Color bars based on significance and magnitude
    % Red for significant negative differences, blue for positive
    cmap = jet(64);
    diff_normalized = (power_diff - min(power_diff)) / (max(power_diff) - min(power_diff));
    
    % Set bar colors based on significance and magnitude
    for i = 1:length(wind_speeds)
        if ~isnan(power_diff(i))
            if is_significant(i)
                if power_diff(i) < 0
                    set(bar_h, 'FaceColor', 'r');  % Significant negative difference
                else
                    set(bar_h, 'FaceColor', 'b');  % Significant positive difference
                end
            else
                % Use gradient for non-significant differences
                color_idx = max(1, round(diff_normalized(i) * 64));
                set(bar_h, 'FaceColor', cmap(color_idx, :));
            end
        end
    end
    
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
        
        plot(highlight_x, highlight_y, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
        text(highlight_x + 0.5, highlight_y, sprintf('Maximum Degradation\n%.1f kWh/10min', abs(highlight_y)), ...
            'FontWeight', 'bold', 'VerticalAlignment', 'middle');
    end
    
    % Add annotations for key operational regions
    x_regions = [5, 12, 20];
    region_names = {'Cubic Region', 'Rated Power Transition', 'High Wind Region'};
    y_pos = min(power_diff) - 10;
    
    for i = 1:length(x_regions)
        text(x_regions(i), y_pos, region_names{i}, 'FontSize', 10, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
        line([x_regions(i) x_regions(i)], [y_pos y_pos+5], 'Color', 'k', 'LineStyle', ':');
    end
    
    % Enhance visualization
    grid on; box on;
    set(gca, 'FontSize', 10, 'LineWidth', 1);
    xlabel('Wind Speed (m/s)', 'FontWeight', 'bold', 'FontSize', 12);
    ylabel('Power Difference: A - B (kWh/10min)', 'FontWeight', 'bold', 'FontSize', 12);
    title('Performance Degradation Analysis: Dataset A vs Dataset B', ...
          'FontWeight', 'bold', 'FontSize', 14);
    
    % Add explanatory legend
    legend('Power Difference (A-B)', '95% Confidence Interval', 'Zero Reference', ...
           'Maximum Degradation Point', 'Location', 'southeast', 'FontSize', 10);
    
    % Set axis limits
    xlim([0 25]);
    max_range = max(abs([min(power_diff), max(power_diff)])) * 1.2;
    ylim([-max_range max_range]);
    
    % Add summary statistics annotation
    mean_diff = mean(power_diff, 'omitnan');
    pct_diff = 100 * mean_diff / mean(results_A.mean_power, 'omitnan');
    
    annotation('textbox', [0.7, 0.8, 0.25, 0.15], ...
               'String', {sprintf('Mean Difference: %.1f kWh/10min', mean_diff), ...
                         sprintf('Percent Reduction: %.1f%%', -pct_diff), ...
                         sprintf('Most Affected: %.1f-%.1f m/s', ...
                         wind_speeds(max_idx)-0.5, wind_speeds(max_idx)+0.5)}, ...
               'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'k', ...
               'FontSize', 10, 'FontWeight', 'bold');
    
    % Export as high-resolution figure
    set(gcf, 'PaperPositionMode', 'auto');
    print('-dpng', '-r300', 'Figure3_DegradationAnalysis.png');
end
