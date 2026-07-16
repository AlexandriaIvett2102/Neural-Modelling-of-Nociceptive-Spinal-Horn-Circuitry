% --- Function to Plot Raster ---
function plotRasterFromSpikeTimes(spike_times)
    % spike_times: cell array where each cell contains the spike times for a neuron
    hold on;
    for i = 1:length(spike_times)
        spikes = spike_times{i};
        plot(spikes, repmat(i, size(spikes)), 'k.', 'MarkerSize', 5);  % Plot spikes as dots
    end
    hold off;
    xlabel('Time (ms)');
    ylabel('Neuron Index');
end