from odesolver_module import ODESolver, Regressor, Visualizer
import numpy as np
import sys
from clusters_module import ClustersVisualizer, SpikingModelComparison

si = 0.3

tau = {'se': 3, 'si': 5, 'me': 1, 'mi': 1, 'z': 120}
g = {'ee': 0, 'ii': 0, 'ei': 1, 'ie': 1}
mu = {'e': 1, 'i': -0.01}

z_range = [0.1, 1]

gz_target = [0.2, 0.24, 0.26, 0.28]
num_clusters = 2

# To view the bifurcation from 2 to 3 clusters, comment out the above two lines and uncomment the following two lines.
# gz_target = [0.35, 0.4, 0.45, 0.5]
# num_clusters = 3

t_max = 200

solver = ODESolver(tau, g, mu, si, gz_target, z_range, t_max)
z_array, g_z_array, tts_array, theta_df_dict = solver.compute_tts()
weight = 1/(np.array(z_array)*g_z_array**5)

func_type = 'exponential'
coefficients = Regressor().function_fit(z_array, g_z_array, tts_array, weight, func_type)
predicted_values = [Regressor().evaluate_fit(coefficients, z, g_z, func_type) for z, g_z in zip(z_array, g_z_array)]

Visualizer().plot_original_and_interpolated(coefficients, z_array, g_z_array, tts_array, func_type, font_size=20)
Visualizer().plot_subplots_for_gz(z_array, g_z_array, tts_array, predicted_values, font_size=35)

T = 400


clusters_visualizer = ClustersVisualizer(num_clusters, tau, g, mu, si, coefficients, func_type)
clusters_visualizer.steady_state_vs_gz(gz_target, theta_df_dict)
two_clust_map = SpikingModelComparison(num_clusters, T, tau, g, mu, si, coefficients, func_type)
two_clust_map.comparison_plots(gz_target)


    


