% Binary Symmetric Channel Analysis
%Analyze Receiver Probabilities: How prior probabilities (p) and error rates (q) affect the probabilities of correctly or incorrectly receiving transmitted bits.
%Design and Interpret MAP Decision Rules: How to maximize a posteriori probabilities to make optimal decisions about transmitted bits based on received data.
clear all; close all;

% Define probability ranges
p_range = 0:0.01:1;  % Range for prior probability
q_values = [0.3 0.5 0.7];  % Different error rates
%    p_range: Defines a range of prior probabilities (P[B1] = p), which varies from 0 to 1.
%    q_values: Specifies different error rates (q) of the channel, where q represents the probability that a transmitted bit is flipped.
% Case 1: Receiver probability analysis

figure(1)
for i = 1:length(q_values)
    % P[A1] - probability of receiving bit "0"
    P_A1 = (1-q_values(i)).*p_range + q_values(i).*(1-p_range);
    plot(p_range, P_A1, 'LineWidth', 1.5)
    hold on
end

%The aim here is to calculate and visualize:
%1.    P[A1]: Probability of receiving “0”.
%2.    P[A2]: Probability of receiving “1”.
%This is analyzed by varying the prior probabilities (p) and error rates (q).
title('Figure 1: P[A1] vs p for different q values')
xlabel('Prior Probability (p)')
ylabel('P[A1]')
legend('q = 0.3', 'q = 0.5', 'q = 0.7')
grid on
%(a) Calculating and Plotting P[A1] vs p
%    Formula:  P[A1] = (1-q)p + q(1-p)
%   (1-q)p: Probability of correctly receiving “0”.
%   q(1-p): Probability of incorrectly receiving “0” (when “1” was transmitted).
%   Plot: For each q value, this is plotted against p.


figure(2)
for i = 1:length(q_values)
    % P[A1] with varying q
    q_range = 0:0.01:1;
    P_A1_q = (1-q_range).*q_values(i) + q_range.*(1-q_values(i));
    plot(q_range, P_A1_q, 'LineWidth', 1.5)
    hold on
end
title('Figure 2: P[A1] vs q for different p values')
xlabel('Channel Error Rate (q)')
ylabel('P[A1]')
legend('p = 0.3', 'p = 0.5', 'p = 0.7')
grid on
%Calculating and Plotting P[A1] vs q
%Formula: Similar to above, but now q varies.
%  Plot: For fixed p values, this explores how the channel error rate (q) impacts the probability of receiving “0

figure(3)
for i = 1:length(q_values)
    % P[A2] - probability of receiving bit "1"
    P_A2 = q_values(i).*p_range + (1-q_values(i)).*(1-p_range);
    plot(p_range, P_A2, 'LineWidth', 1.5)
    hold on
end
title('Figure 3: P[A2] vs p for different q values')
xlabel('Prior Probability (p)')
ylabel('P[A2]')
legend('q = 0.3', 'q = 0.5', 'q = 0.7')
grid on
%(c) Calculating and Plotting P[A2] vs p
%    Formula:  P[A2] = qp + (1-q)(1-p)
%  qp: Probability of incorrectly receiving “1”.
%  (1-q)(1-p): Probability of correctly receiving “1”.
%  Plot: Similar to P[A1] vs p, but for P[A2].

figure(4)
for i = 1:length(q_values)
    % P[A2] with varying q
    q_range = 0:0.01:1;
    P_A2_q = q_range.*q_values(i) + (1-q_range).*(1-q_values(i));
    plot(q_range, P_A2_q, 'LineWidth', 1.5)
    hold on
end
title('Figure 4: P[A2] vs q for different p values')
xlabel('Channel Error Rate (q)')
ylabel('P[A2]')
legend('p = 0.3', 'p = 0.5', 'p = 0.7')
grid on
%Calculating and Plotting P[A2] vs q
%  Formula: Similar to above, but q varies.
%  Plot: Examines how P[A2] depends on q.
% Case 2: MAP Decision Rule Analysis
q_values_case2 = [0.3 0.5];  % q values for case 2

%This case focuses on the Maximum A Posteriori (MAP) decision rule:
%   Decide whether B1 or B2 was transmitted based on posterior probabilities:
%    P[B1|A1] ,  P[B2|A1]  when “0” is received.
%     P[B1|A2] ,  P[B2|A2]  when “1” is received.

figure(5)
for i = 1:length(q_values_case2)
    % Calculate P[B1|A1] and P[B2|A1]
    P_B1_A1 = (1-q_values_case2(i)).*p_range./(p_range.*(1-q_values_case2(i)) + (1-p_range).*q_values_case2(i));
    P_B2_A1 = 1 - P_B1_A1;
    plot(p_range, P_B1_A1, 'LineWidth', 1.5)
    hold on
    plot(p_range, P_B2_A1, '--', 'LineWidth', 1.5)
end
title('Figure 5: P[B1|A1] and P[B2|A1] vs p')
xlabel('Prior Probability (p)')
ylabel('Posterior Probability')
legend('P[B1|A1], q=0.3', 'P[B2|A1], q=0.3', 'P[B1|A1], q=0.5', 'P[B2|A1], q=0.5')
grid on

%Posterior Formula:
%    P[B1|A1] = \frac{(1-q)p}{p(1-q) + (1-p)q}
%   P[B2|A1] = 1 - P[B1|A1]
%   Plot: Compares P[B1|A1] and P[B2|A1] for different q values.


figure(6)
for i = 1:length(q_values_case2)
    % Calculate P[B1|A2] and P[B2|A2]
    P_B1_A2 = q_values_case2(i).*p_range./(p_range.*q_values_case2(i) + (1-p_range).*(1-q_values_case2(i)));
    P_B2_A2 = 1 - P_B1_A2;
    plot(p_range, P_B1_A2, 'LineWidth', 1.5)
    hold on
    plot(p_range, P_B2_A2, '--', 'LineWidth', 1.5)
end
title('Figure 6: P[B1|A2] and P[B2|A2] vs p')
xlabel('Prior Probability (p)')
ylabel('Posterior Probability')
legend('P[B1|A2], q=0.3', 'P[B2|A2], q=0.3', 'P[B1|A2], q=0.5', 'P[B2|A2], q=0.5')
grid on
%
%  Posterior Formula:
%    P[B1|A2] = \frac{qp}{pq + (1-p)(1-q)}
% P[B2|A2] = 1 - P[B1|A2]
%s   Plot: Compares P[B1|A2] and P[B2|A2].
