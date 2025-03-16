% Parameters setup
p_values = [0.3 0.5 0.7]; % Prior probability values
q_values = [0.3 0.5 0.7]; % Channel quality values
p_range = 0:0.01:1;       % Range for prior probability
q_range = 0:0.01:1;       % Range for channel quality

% Create figure with 6 subplots
figure('Position', [100 100 1200 800]);

% Figure 1: P[A1] vs q for different p values
subplot(2,3,1)
for i = 1:length(p_values)
    p = p_values(i);
    % P[A1] = P[A1|B1]P[B1] + P[A1|B2]P[B2] = q*p + (1-q)*(1-p)
    PA1 = q_range.*p + (1-q_range).*(1-p);
    plot(q_range, PA1, 'LineWidth', 2); hold on;
end
xlabel('Channel Error Rate (q)');
ylabel('P[A1]');
title('Probability of Receiving 0 vs Channel Quality');
legend('p = 0.3', 'p = 0.5', 'p = 0.7', 'Location', 'best');
grid on; ylim([0 1]);

% Figure 2: P[A1] vs p for different q values
subplot(2,3,2)
for i = 1:length(q_values)
    q = q_values(i);
    % P[A1] = q*p + (1-q)*(1-p)
    PA1 = q.*p_range + (1-q).*(1-p_range);
    plot(p_range, PA1, 'LineWidth', 2); hold on;
end
xlabel('Prior Probability (p)');
ylabel('P[A1]');
title('Receiver Probability 0 vs Prior Distribution');
legend('q = 0.3', 'q = 0.5', 'q = 0.7', 'Location', 'best');
grid on; ylim([0 1]);

% Figure 3: P[A2] vs q for different p values
subplot(2,3,3)
for i = 1:length(p_values)
    p = p_values(i);
    % P[A2] = P[A2|B1]P[B1] + P[A2|B2]P[B2] = (1-q)*p + q*(1-p)
    PA2 = (1-q_range).*p + q_range.*(1-p);
    plot(q_range, PA2, 'LineWidth', 2); hold on;
end
xlabel('Channel Error Rate (q)');
ylabel('P[A2]');
title('Probability of Receiving 1 vs Channel Quality');
legend('p = 0.3', 'p = 0.5', 'p = 0.7', 'Location', 'best');
grid on; ylim([0 1]);

% Figure 4: P[A2] vs p for different q values
subplot(2,3,4)
for i = 1:length(q_values)
    q = q_values(i);
    % P[A2] = (1-q)*p + q*(1-p)
    PA2 = (1-q).*p_range + q.*(1-p_range);
    plot(p_range, PA2, 'LineWidth', 2); hold on;
end
xlabel('Prior Probability (p)');
ylabel('P[A2]');
title('Receiver Probability 1 vs Prior Distribution');
legend('q = 0.3', 'q = 0.5', 'q = 0.7', 'Location', 'best');
grid on; ylim([0 1]);

% Figure 5: P[B1|A1] and P[B2|A1] vs p for different q values
subplot(2,3,5)
colors = {'blue', 'red', 'yellow'};
for i = 1:length(q_values)
    q = q_values(i);
    % Calculate P[A1] first
    PA1 = q.*p_range + (1-q).*(1-p_range);
    
    % P[B1|A1] = P[A1|B1]P[B1]/P[A1] = (q*p)/(q*p + (1-q)*(1-p))
    PB1A1 = (q.*p_range) ./ PA1;
    
    % P[B2|A1] = P[A1|B2]P[B2]/P[A1] = ((1-q)*(1-p))/(q*p + (1-q)*(1-p))
    PB2A1 = ((1-q).*(1-p_range)) ./ PA1;
    
    % Plot both posterior probabilities
    plot(p_range, PB1A1, 'Color', colors{i}, 'LineWidth', 2); hold on;
    plot(p_range, PB2A1, '--', 'Color', colors{i}, 'LineWidth', 2);
end
xlabel('Prior Probability (p)');
ylabel('Posterior Probability');
title('Bayesian Inference Given Received 0');
legend('P[B1|A1], q=0.3', 'P[B2|A1], q=0.3', 'P[B1|A1], q=0.5', 'P[B2|A1], q=0.5', 'P[B1|A1], q=0.7', 'P[B2|A1], q=0.7', 'Location', 'best');
grid on; ylim([0 1]);

% Figure 6: P[B1|A2] and P[B2|A2] vs p for different q values
subplot(2,3,6)
for i = 1:length(q_values)
    q = q_values(i);
    % Calculate P[A2] first
    PA2 = (1-q).*p_range + q.*(1-p_range);
    
    % P[B1|A2] = P[A2|B1]P[B1]/P[A2] = ((1-q)*p)/((1-q)*p + q*(1-p))
    PB1A2 = ((1-q).*p_range) ./ PA2;
    
    % P[B2|A2] = P[A2|B2]P[B2]/P[A2] = (q*(1-p))/((1-q)*p + q*(1-p))
    PB2A2 = (q.*(1-p_range)) ./ PA2;
    
    % Plot both posterior probabilities
    plot(p_range, PB1A2, 'Color', colors{i}, 'LineWidth', 2); hold on;
    plot(p_range, PB2A2, '--', 'Color', colors{i}, 'LineWidth', 2);
end
xlabel('Prior Probability (p)');
ylabel('Posterior Probability');
title('Bayesian Inference Given Received 1');
legend('P[B1|A2], q=0.3', 'P[B2|A2], q=0.3', 'P[B1|A2], q=0.5', 'P[B2|A2], q=0.5', 'P[B1|A2], q=0.7', 'P[B2|A2], q=0.7', 'Location', 'best');
grid on; ylim([0 1]);

% Format figure
set(gcf, 'color', 'w');
sgtitle('Comprehensive BSC Analysis with Bayesian Decision Theory', 'FontSize', 14);

% Save figure with high resolution
print('BSC_Analysis_Professional', '-dpng', '-r300');
