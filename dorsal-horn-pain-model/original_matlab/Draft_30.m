% --- Define Parameters ---
N_neurons = 200; % Set total number of neurons in the model
N_NS = 40; % Set number of nociceptor-specific neurons
N_WDR = N_neurons - N_NS; % Set number of wide dynamic range neurons
N_INH = 40; % Set number of inhibitory interneurons
dt = 0.1; % Set the simulation time step in milliseconds
T = 2000; % Set total simulation time in milliseconds
noise_amplitude = 0.5; % Set amplitude of background noise
base_inhibition = 0.2; % Set base scaling factor for top-down inhibition
delay_range = [1, 5]; % Define range of synaptic delays in milliseconds

% --- Synaptic Weight Variations ---
synaptic_weight_factors = [0.5, 1, 1.5, 2]; % Define scaling factors for synaptic weights - change to alter weights in multiplication incriments
num_variations = length(synaptic_weight_factors); % Get number of variations

% Prepare a single figure for subplots
figure; % Create a new figure window
set(gcf, 'Position', [100, 100, 1400, 900]); % Set the figure size and position on screen

% Create an array to store synchrony results for each variation
synchrony_all_iterations = zeros(num_variations, T/dt); % Initialize array to store synchrony data for synchriny testing

% Loop through each synaptic weight factor
for variation_idx = 1:num_variations
    weight_factor = synaptic_weight_factors(variation_idx); % Get the current weight factor
    
    % --- Heterogeneous Izhikevich Model Parameters ---
    a = [0.02 + 0.005*randn(N_NS,1); 0.02 + 0.005*randn(N_WDR,1); 0.02 + 0.005*randn(N_INH,1)]; 
    % Generate random "a" parameter values for each neuron group
    b = 0.2 + 0.01*randn(N_neurons+N_INH,1); 
    % Generate random "b" parameter values for each neuron
    c = -65 + 5*randn(N_neurons+N_INH,1); 
    % Generate random "c" parameter values for each neuron
    d = [8 + 1*randn(N_NS,1); 8 + 1*randn(N_WDR,1); 2 + 0.5*randn(N_INH,1)]; 
    % Generate random "d" parameter values for each neuron group
    
    % --- Synaptic Parameters ---
    g_AMPA = weight_factor * (0.01 + 0.005*randn(N_neurons+N_INH, N_neurons+N_INH)); 
    % Generate synaptic weights for AMPA receptors, scaled by the weight factor
    g_NMDA = weight_factor * (0.05 + 0.01*randn(N_neurons+N_INH, N_neurons+N_INH)); 
    % Generate synaptic weights for NMDA receptors, scaled by the weight factor
    g_GABA = weight_factor * (0.02 + 0.005*randn(N_neurons+N_INH, N_neurons+N_INH)); 
    % Generate synaptic weights for GABA receptors, scaled by the weight factor

    % --- Connectivity ---
    positions = rand(N_neurons+N_INH, 2); 
    % Generate random 2D positions for each neuron and interneuron
    distances = squareform(pdist(positions)); 
    % Calculate the pairwise Euclidean distances between neurons
    connectivity_matrix = zeros(N_neurons+N_INH); 
    % Initialize the connectivity matrix with zeros
    connectivity_matrix(1:N_NS, N_NS+1:N_NS+N_WDR) = distances(1:N_NS, N_NS+1:N_NS+N_WDR) < 0.3; 
    % Connect nociceptor-specific neurons to wide dynamic range neurons if they are close enough
    connectivity_matrix(N_NS+1:N_NS+N_WDR, end-N_INH+1:end) = distances(N_NS+1:N_NS+N_WDR, end-N_INH+1:end) < 0.2; 
    % Connect wide dynamic range neurons to inhibitory interneurons if they are close enough
    connectivity_matrix(end-N_INH+1:end, 1:N_neurons) = distances(end-N_INH+1:end, 1:N_neurons) < 0.4; 
    % Connect inhibitory interneurons to both nociceptor-specific and wide dynamic range neurons
    connectivity_matrix(1:N_neurons+1:end) = 0; 
    % Ensure no self-connections (diagonal elements set to 0)

    weights_exc = rand(N_neurons+N_INH) .* connectivity_matrix * 2; 
    % Generate random excitatory synaptic weights, scaled by connectivity matrix
    weights_inh = -rand(N_neurons+N_INH) .* connectivity_matrix; 
    % Generate random inhibitory synaptic weights (negative values for inhibition)
    weights_exc(end-N_INH+1:end, :) = weights_exc(end-N_INH+1:end, :) * 2; 
    % Double the excitatory weights for inhibitory interneurons
    delays = delay_range(1) + diff(delay_range) * rand(N_neurons+N_INH) .* connectivity_matrix; 
    % Randomly assign synaptic delays based on the given range and connectivity

    % --- States and Inputs ---
    V = c; 
    % Initialize the membrane potential for each neuron with the c parameter
    u = zeros(N_neurons+N_INH, 1); 
    % Initialize the recovery variable u for each neuron
    spike_times = cell(N_neurons+N_INH, 1); 
    % Initialize a cell array to store spike times for each neuron
    last_spike = -inf * ones(N_neurons+N_INH, 1); 
    % Initialize last spike times for each neuron to a very negative value
    nociceptor_input = zeros(T/dt, N_NS); 
    % Initialize input to nociceptor neurons
    nociceptor_input(200:400, :) = 100 + 20 * randn(201, N_NS); 
    % Apply random stimulus input to nociceptor neurons at specific time intervals

    % --- Simulation Loop ---
    for t = 1:T/dt 
        currents = zeros(N_neurons+N_INH, 1) + noise_amplitude * randn(N_neurons+N_INH, 1); 
        % Add noise to the current for each neuron
        for i = 1:N_neurons+N_INH 
            for j = 1:N_neurons+N_INH 
                if connectivity_matrix(i, j) == 1 
                    if ~isempty(spike_times{j}) 
                        delayed_spike_time = spike_times{j}(end) + delays(i, j); 
                        % Calculate the delayed spike time for neuron i based on neuron j's spike
                        if t*dt > delayed_spike_time 
                            currents(i) = currents(i) + ...
                                weights_exc(i, j) * exp(-(t*dt - delayed_spike_time) / 2) * (V(j) - 0) + ...
                                weights_inh(i, j) * exp(-(t*dt - delayed_spike_time) / 5) * (V(j) + 75); 
                            % Add the contribution of synaptic currents (excitatory and inhibitory) to neuron i
                        end
                    end
                end
            end
        end
        currents(1:N_NS) = currents(1:N_NS) + nociceptor_input(t, :)'; 
        % Add nociceptor input to the currents of nociceptor-specific neurons
        currents(N_NS+1:N_NS+N_WDR) = currents(N_NS+1:N_NS+N_WDR) - ...
            (base_inhibition + 0.1 * mean(currents(N_NS+1:N_NS+N_WDR))) * mean(currents(N_NS+1:N_NS+N_WDR)); 
        % Apply inhibition to the wide dynamic range neurons

        dVdt = 0.04 * V.^2 + 5 * V + 140 - u + currents; 
        % Update the membrane potential based on the Izhikevich model equations
        dudt = a .* (b .* V - u); 
        % Update the recovery variable based on the Izhikevich model equations
        V = V + dt * dVdt; 
        % Update membrane potential for each neuron
        u = u + dt * dudt; 
        % Update the recovery variable for each neuron

        spikes = (V >= 30); 
        % Detect spikes when membrane potential exceeds threshold (30 mV)
        V(spikes) = c(spikes); 
        % Reset the membrane potential for spiking neurons to their "c" value
        u(spikes) = u(spikes) + d(spikes); 
        % Update the recovery variable for spiking neurons
        for i = 1:N_neurons+N_INH
            if spikes(i) 
                spike_times{i} = [spike_times{i}, t*dt]; 
                % Record spike time for each neuron
                last_spike(i) = t*dt; 
                % Update the last spike time
            end
        end
    end

    % --- Use plotSynchronyFromSpikeTimes for Synchrony Calculation ---
    subplot(num_variations, 2, (variation_idx-1)*2 + 1); 
    % Create a subplot for raster plot
    plotRasterFromSpikeTimes(spike_times); 
    % Plot raster for the current weight variation
    title(sprintf('Raster Plot (Weight Factor = %.1f)', weight_factor));
    xlabel('Time (ms)');
    ylabel('Neuron Index');
    
    % Store synchrony for this variation using plotSynchronyFromSpikeTimes
    subplot(num_variations, 2, (variation_idx-1)*2 + 2); 
    % Create a subplot for synchrony plot
    plotSynchronyFromSpikeTimes(spike_times, T/dt, dt); 
    % Plot synchrony for the current weight variation
    title(sprintf('Synchrony (Weight Factor = %.1f)', weight_factor));
    xlabel('Time (ms)');
    ylabel('Synchrony');
end

% --- Display Synchrony Across All Variations ---
figure;
hold on;
for variation_idx = 1:num_variations
    % Synchrony is already calculated and plotted in individual subplots
    % For global synchrony plot, you can either extract it from the subplots or use a global variable
end
title('Synchrony Across Different Weight Variations');
xlabel('Time (ms)');
ylabel('Synchrony');
legend(arrayfun(@(x) sprintf('Weight Factor = %.1f', x), synaptic_weight_factors, 'UniformOutput', false));
hold off;
