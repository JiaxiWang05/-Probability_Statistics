% Step 2: Transform the data
Comp_str_ln = log(data.Comp_strength);

% Calculate water-to-cement and water-to-binder ratios
wc_cem = data.Water ./ data.Cement; % Water content : Cement
wc_binder = data.Water ./ sum([data.Cement data.Slag data.Ash],2); % Water : Binder (sum) 