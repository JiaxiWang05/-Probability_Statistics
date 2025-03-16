check% Binary Symmetric Channel Analysis with Bayesian Theory Implementation
% Corrected formulas and enhanced visualization for top-band coursework

 

%% Probability parameter initialization
p_range = 0:0.01:1;        % Prior probability range
p_values = [0.3, 0.5, 0.7]; % Key prior probabilities
q_values = [0.3, 0.5, 0.7]; % Channel quality parameters
q_range = 0:0.01:1;        % Channel error rate range

%% Create professional figure layout
figure('Position', [100, 100, 1400, 900], 'Color', 'w')
colors = lines(3);          % MATLAB's professional color palette
line_styles = {'-', '--', ':'}; % Line style variations

%% Case 1: Receiver Probability Analysis (Corrected Formulas)
% --- Figure 1: P[A1] vs q with fixed p ---
subplot(2,3,1)
for i = 1:length(p_values)
    % CORRECTED FORMULA: P[A1] = q*p + (1-q)*(1-p)
    P_A1 = q_range*p_values(i) + (1-q_range)*(1-p_values(i));
    plot(q_range, P_A1, 'LineWidth', 2, 'Color', colors(i,:), ...
        'LineStyle', line_styles{1})
    hold on
end
format_subplot('Channel Error Rate (q)', 'P[A1]', ...
    'Probability of Receiving 0 vs Channel Quality', ...
    {'p = 0.3', 'p = 0.5', 'p = 0.7'})

% --- Figure 2: P[A1] vs p with fixed q ---
subplot(2,3,2)
for i = 1:length(q_values)
    % CORRECTED FORMULA: P[A1] = q*p + (1-q)*(1-p)
    P_A1 = q_values(i)*p_range + (1-q_values(i))*(1-p_range);
    plot(p_range, P_A1, 'LineWidth', 2, 'Color', colors(i,:), ...
        'LineStyle', line_styles{1})
    hold on
end
format_subplot('Prior Probability (p)', 'P[A1]', ...
    'Receiver Probability 0 vs Prior Distribution', ...
    {'q = 0.3', 'q = 0.5', 'q = 0.7'})

% --- Figure 3: P[A2] vs q with fixed p ---
subplot(2,3,3)
for i = 1:length(p_values)
    % CORRECTED FORMULA: P[A2] = (1-q)*p + q*(1-p)
    P_A2 = (1-q_range)*p_values(i) + q_range*(1-p_values(i));
    plot(q_range, P_A2, 'LineWidth', 2, 'Color', colors(i,:), ...
        'LineStyle', line_styles{1})
    hold on
end
format_subplot('Channel Error Rate (q)', 'P[A2]', ...
    'Probability of Receiving 1 vs Channel Quality', ...
    {'p = 0.3', 'p = 0.5', 'p = 0.7'})

% --- Figure 4: P[A2] vs p with fixed q ---
subplot(2,3,4)
for i = 1:length(q_values)
    % CORRECTED FORMULA: P[A2] = (1-q)*p + q*(1-p)
    P_A2 = (1-q_values(i))*p_range + q_values(i)*(1-p_range);
    plot(p_range, P_A2, 'LineWidth', 2, 'Color', colors(i,:), ...
        'LineStyle', line_styles{1})
    hold on
end
format_subplot('Prior Probability (p)', 'P[A2]', ...
    'Receiver Probability 1 vs Prior Distribution', ...
    {'q = 0.3', 'q = 0.5', 'q = 0.7'})

%% Case 2: Posterior Probabilities & MAP Analysis (Corrected)
% --- Figure 5: P[B1|A1] and P[B2|A1] ---
subplot(2,3,5)
for i = 1:length(q_values)
    % CORRECTED BAYESIAN FORMULA
    numerator = q_values(i) * p_range;
    denominator = q_values(i) * p_range + (1-q_values(i)) * (1-p_range);
    P_B1_A1 = numerator ./ denominator;
    P_B2_A1 = 1 - P_B1_A1;
    
    plot(p_range, P_B1_A1, 'LineWidth', 2, 'Color', colors(i,:))
    hold on
    plot(p_range, P_B2_A1, '--', 'LineWidth', 2, 'Color', colors(i,:))
end
format_subplot('Prior Probability (p)', 'Posterior Probability', ...
    'Bayesian Inference Given Received 0', ...
    {'B1|A1, q=0.3', 'B2|A1, q=0.3', 'B1|A1, q=0.5', ...
    'B2|A1, q=0.5', 'B1|A1, q=0.7', 'B2|A1, q=0.7'})

% --- Figure 6: P[B1|A2] and P[B2|A2] ---
subplot(2,3,6)
for i = 1:length(q_values)
    % CORRECTED BAYESIAN FORMULA
    numerator = (1-q_values(i)) * p_range;
    denominator = (1-q_values(i)) * p_range + q_values(i) * (1-p_range);
    P_B1_A2 = numerator ./ denominator;
    P_B2_A2 = 1 - P_B1_A2;
    
    plot(p_range, P_B1_A2, 'LineWidth', 2, 'Color', colors(i,:))
    hold on
    plot(p_range, P_B2_A2, '--', 'LineWidth', 2, 'Color', colors(i,:))
end
format_subplot('Prior Probability (p)', 'Posterior Probability', ...
    'Bayesian Inference Given Received 1', ...
    {'B1|A2, q=0.3', 'B2|A2, q=0.3', 'B1|A2, q=0.5', ...
    'B2|A2, q=0.5', 'B1|A2, q=0.7', 'B2|A2, q=0.7'})

%% Final formatting and MAP threshold calculation
sgtitle('Comprehensive BSC Analysis with Bayesian Decision Theory', ...
    'FontSize', 16, 'FontWeight', 'bold')
exportgraphics(gcf, 'BSC_Analysis_Professional.png', 'Resolution', 300)

% MAP decision threshold calculation (Corrected)
threshold_p = 1 - q_values;  % Correct formula: p_threshold = 1-q
disp('MAP Decision Thresholds:')
disp(table(q_values', threshold_p', 'VariableNames', {'q', 'p_threshold'}))

%% Professional Formatting Function
function format_subplot(xlab, ylab, title_text, legend_items)
    set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on')
    xlabel(xlab, 'FontSize', 12, 'FontWeight', 'bold')
    ylabel(ylab, 'FontSize', 12, 'FontWeight', 'bold')
    title(title_text, 'FontSize', 13, 'FontWeight', 'bold')
    legend(legend_items, 'Location', 'best', 'FontSize', 9)
    grid on
    ylim([0 1])
    xlim([0 1])
end
