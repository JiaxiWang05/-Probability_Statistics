% Define bin edges
bin_edges = 0:1:25;

% Initialize arrays to hold results
mean_wind_speed = zeros(length(bin_edges)-1, 1);
mean_energy = zeros(length(bin_edges)-1, 1);
std_energy = zeros(length(bin_edges)-1, 1);
ci_lower = zeros(length(bin_edges)-1, 1);
ci_upper = zeros(length(bin_edges)-1, 1);

% Loop through each bin
for i = 1:length(bin_edges)-1
    % Find indices for the current bin
    bin_indices_A = (data.u_A >= bin_edges(i)) & (data.u_A < bin_edges(i+1));
    bin_indices_B = (data.u_B >= bin_edges(i)) & (data.u_B < bin_edges(i+1));
    
    % Calculate statistics for Dataset A
    if any(bin_indices_A)
        mean_wind_speed(i) = mean(data.u_A(bin_indices_A));
        mean_energy(i) = mean(data.P_A(bin_indices_A));
        std_energy(i) = std(data.P_A(bin_indices_A));
        n = sum(bin_indices_A);
        ci = 1.96 * (std_energy(i) / sqrt(n)); % 95% CI
        ci_lower(i) = mean_energy(i) - ci;
        ci_upper(i) = mean_energy(i) + ci;
    end
    
    % Calculate statistics for Dataset B
    if any(bin_indices_B)
        mean_wind_speed(i) = mean(data.u_B(bin_indices_B));
        mean_energy(i) = mean(data.P_B(bin_indices_B));
        std_energy(i) = std(data.P_B(bin_indices_B));
        n = sum(bin_indices_B);
        ci = 1.96 * (std_energy(i) / sqrt(n)); % 95% CI
        ci_lower(i) = mean_energy(i) - ci;
        ci_upper(i) = mean_energy(i) + ci;
    end
end

% Plotting the results
figure;
hold on;
errorbar(mean_wind_speed, mean_energy, mean_energy - ci_lower, ci_upper - mean_energy, 'o', 'DisplayName', 'Dataset A', 'Color', 'b');
errorbar(mean_wind_speed, mean_energy, mean_energy - ci_lower, ci_upper - mean_energy, 'o', 'DisplayName', 'Dataset B', 'Color', 'r');
xlabel('Wind Speed (m/s)');
ylabel('Mean Energy Production (kWh/10min)');
title('Binned Power Curve with 95% Confidence Intervals');
legend('Dataset A', 'Dataset B');
grid on;
hold off;