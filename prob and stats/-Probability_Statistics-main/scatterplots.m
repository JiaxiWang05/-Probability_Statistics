load('turbine.mat');

figure;
subplot(1,2,1);
scatter(u_A, P_A, 3, 'filled');
xlabel('Wind Speed (m/s)');
ylabel('Energy (kWh/10min)');
title('Sample A Power Curve');
xlim([0 25]);

subplot(1,2,2);
scatter(u_B, P_B, 3, 'filled');
xlabel('Wind Speed (m/s)');
ylabel('Energy (kWh/10min)');
title('Sample B Power Curve');
xlim([0 25]);
