% function plotSynchronyFromSpikeTimes(spikeTimes, timeResolution, totalTime)
%     % plotSynchronyFromSpikeTimes - Measures and plots synchrony as a time series from spike times.
%     %
%     % Inputs:
%     %   spikeTimes     - Cell array (1 x N), where each cell contains a vector of spike times for a neuron.
%     %   timeResolution - Time resolution for analysis (in ms).
%     %   totalTime      - Total duration of the simulation (in ms).
%     %
%     % Output:
%     %   A plot of synchrony (%) over time.
% 
%     % Number of neurons
%     numNeurons = length(spikeTimes);
% 
%     % Create time bins
%     edges = 0:timeResolution:totalTime; % Bin edges
%     time = edges(1:end-1) + timeResolution / 2; % Time at bin centers
% 
%     % Initialize spike count array
%     spikeCounts = zeros(1, length(edges) - 1);
% 
%     % Count spikes in each bin
%     for i = 1:numNeurons
%         % Bin spike times for the current neuron
%         spikeCounts = spikeCounts + histcounts(spikeTimes{i}, edges);
%     end
% 
%     % Compute synchrony as fraction of neurons spiking in each time bin
%     synchrony = spikeCounts / numNeurons;
% 
%     % Plot the synchrony time series
%     figure;
%     plot(time, synchrony * 100, 'b', 'LineWidth', 1.5); % Synchrony in %
%     xlabel('Time (ms)', 'FontSize', 12);
%     ylabel('Synchrony (%)', 'FontSize', 12);
%     title('Synchrony Over Time', 'FontSize', 14);
%     grid on;
%     xlim([0 totalTime]);
%     ylim([0 100]);
% end


% --- Function to Plot Synchrony ---
function plotSynchronyFromSpikeTimes(spike_times, T, dt)
    % T: Total simulation time (in ms)
    % dt: Time step (in ms)
    
    % Number of time steps in the simulation
    num_steps = T / dt; 
    
    % Initialize an array to store synchrony values over time
    synchrony = zeros(num_steps, 1);
    
    % Loop through each time step
    for t = 1:num_steps
        % Get the spike times at the current time step
        current_time = t * dt;
        
        % Count the number of neurons that spike within the same time bin
        spike_count = 0;
        for i = 1:length(spike_times)
            if ~isempty(spike_times{i}) && any(abs(spike_times{i} - current_time) < dt/2)
                spike_count = spike_count + 1; % Count neurons that spike in this bin
            end
        end
        
        % Calculate synchrony at this time step as the fraction of neurons spiking
        synchrony(t) = spike_count / length(spike_times);
    end
    
    % Plot the synchrony over time
    plot((1:num_steps) * dt, synchrony, 'LineWidth', 2);
    xlabel('Time (ms)');
    ylabel('Synchrony');
    title('Synchrony Over Time');
    grid on;
end