% Binary Symmetric Channel Analysis
%Analyze Receiver Probabilities: How prior probabilities (p) and error rates (q) affect the probabilities of correctly or incorrectly receiving transmitted bits.
%Design and Interpret MAP Decision Rules: How to maximize a posteriori probabilities to make optimal decisions about transmitted bits based on received data.
clear all; close all;

% Define probability ranges
p_range = 0:0.01:1;  
p_values = [0.3 0.5 0.7];  
q_values = [0.3 0.5 0.7];  
q_range = 0:0.01:1;

% Create one figure with 6 subplots
figure('Position', [100, 100, 1200, 800])

% Figure 1: P[A1] with fixed p values, varying q (now correct)
subplot(2,3,1)
for i = 1:length(p_values)
    P_A1 = (1-q_range).*p_values(i) + q_range.*(1-p_values(i));
    plot(q_range, P_A1, 'LineWidth', 1.5)
    hold on
end
title('P[A1] vs q for different p values')
xlabel('Channel Error Rate (q)')
ylabel('P[A1]')
legend('p = 0.3', 'p = 0.5', 'p = 0.7')
grid on
ylim([0 1])

% Figure 2: P[A1] with fixed q values, varying p
subplot(2,3,2)
for i = 1:length(q_values)
    P_A1 = (1-q_values(i)).*p_range + q_values(i).*(1-p_range);
    plot(p_range, P_A1, 'LineWidth', 1.5)
    hold on
end
title('P[A1] vs p for different q values')
xlabel('Prior Probability (p)')
ylabel('P[A1]')
legend('q = 0.3', 'q = 0.5', 'q = 0.7')
grid on
ylim([0.3 0.7])

% Figure 3: P[A2] with fixed p values, varying q (now correct)
subplot(2,3,3)
for i = 1:length(p_values)
    P_A2 = q_range.*p_values(i) + (1-q_range).*(1-p_values(i));
    plot(q_range, P_A2, 'LineWidth', 1.5)
    hold on
end
title('P[A2] vs q for different p values')
xlabel('Channel Error Rate (q)')
ylabel('P[A2]')
legend('p = 0.3', 'p = 0.5', 'p = 0.7')
grid on
ylim([0 1])

% Figure 4: P[A2] with fixed q values, varying p
subplot(2,3,4)
for i = 1:length(q_values)
    P_A2 = q_values(i).*p_range + (1-q_values(i)).*(1-p_range);
    plot(p_range, P_A2, 'LineWidth', 1.5)
    hold on
end
title('P[A2] vs p for different q values')
xlabel('Prior Probability (p)')
ylabel('P[A2]')
legend('q = 0.3', 'q = 0.5', 'q = 0.7')
grid on
ylim([0.3 0.7])

% Figure 5 (bottom left) - Posterior probabilities remain the same
subplot(2,3,5)
for i = 1:length(q_values)
    P_B1_A1 = (1-q_values(i)).*p_range./(p_range.*(1-q_values(i)) + (1-p_range).*q_values(i));
    P_B2_A1 = 1 - P_B1_A1;
    plot(p_range, P_B1_A1, 'LineWidth', 1.5)
    hold on
    plot(p_range, P_B2_A1, '--', 'LineWidth', 1.5)
end
title('P[B1|A1] and P[B2|A1] vs p')
xlabel('Prior Probability (p)')
ylabel('Posterior Probability')
legend('P[B1|A1], q=0.3', 'P[B2|A1], q=0.3', 'P[B1|A1], q=0.5', 'P[B2|A1], q=0.5', 'P[B1|A1], q=0.7', 'P[B2|A1], q=0.7')
grid on

% Figure 6 (bottom right) - Posterior probabilities remain the same
subplot(2,3,6)
for i = 1:length(q_values)
    P_B1_A2 = (q_values(i).*p_range)./(q_values(i).*p_range + (1-q_values(i)).*(1-p_range));
    P_B2_A2 = 1 - P_B1_A2;
    plot(p_range, P_B1_A2, 'LineWidth', 1.5)
    hold on
    plot(p_range, P_B2_A2, '--', 'LineWidth', 1.5)
end
title('P[B1|A2] and P[B2|A2] vs p')
xlabel('Prior Probability (p)')
ylabel('Posterior Probability')
legend('P[B1|A2], q=0.3', 'P[B2|A2], q=0.3', 'P[B1|A2], q=0.5', 'P[B2|A2], q=0.5', 'P[B1|A2], q=0.7', 'P[B2|A2], q=0.7')
grid on

% Adjust spacing between subplots
sgtitle('Binary Symmetric Channel Analysis', 'FontSize', 14)
set(gcf, 'Color', 'white')
