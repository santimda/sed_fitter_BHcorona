import matplotlib.pyplot as plt
import numpy as np

from smbh_corona.plot_utils import (
    plot_bands,
    plot_components,
    plot_components_confidence_intervals,
)


def test_sed_plot_helpers_render_components_confidence_and_bands():
    frequencies = np.geomspace(1e9, 1e11, 30)
    components = [np.full(frequencies.shape, i + 1.0) for i in range(5)]
    component_samples = [
        np.tile(component, (10, 1)) for component in components
    ]
    figure, axis = plt.subplots()

    plot_components(axis, frequencies, components, label_prefix="")
    plot_components_confidence_intervals(
        axis, frequencies, component_samples, sigmas=1
    )
    plot_bands(axis, ymin=0.1, ymax=10.0, bands_to_plot=["B3"])
    figure.canvas.draw()

    assert axis.get_legend_handles_labels()[0]
    plt.close(figure)
