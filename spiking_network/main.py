from facade import theta_ei,clusters_saving,reordering,plot_e_i_raster
import time 
import numpy as np
import matplotlib.pyplot as plt
from community_detection import Louvain

plt.close('all')

T = 2000

num_cells = {'e': 200, 'i': 40}
tau = {'se': 3, 'si': 5, 'me': 1, 'mi': 1, 'z': 120}
g = {'ee': 0, 'ii': 0, 'ei': 1, 'ie': 1, 'z': 0.7}
delta = {'e': 0.01, 'i': 0.01}
mu = {'e': 1, 'i': -0.01}

# If quenched is True, the Lorenzian heterogeneity is fixed for the whole simulation. Otherwise, Gaussian noise is used
# for each neuron.
quenched = False

z_init =  'random'
same_as_z = False
seed = int(np.random.rand(1)[0]*1000)

start = time.time()
tout, data_e_dict, data_i_dict, con_mat = theta_ei(num_cells,T,tau,g,delta,mu,quenched,seed,z_init,same_as_z)
end = time.time()
print("Elapsed time:", end - start)

setot = np.sum(data_e_dict['s'],axis=0)/num_cells['e']
sitot = np.sum(data_i_dict['s'],axis=0)/num_cells['i']

t_stable = 3000
labels_e,num_of_clusters_e,Q = Louvain(data_e_dict['theta'][:,t_stable:].T, global_mode=True)
e_clusters_dict,e_clusters_info_df = clusters_saving(data_e_dict,num_cells['e'],num_of_clusters_e, labels_e)
e_clusters_info_df = e_clusters_info_df.sort_values(by='relative size',ascending=False).reset_index()
reordered_e_dict = reordering(e_clusters_dict,labels_e)
print(e_clusters_info_df.head())

plot_e_i_raster(tout, reordered_e_dict['firing'], data_i_dict['firing'])

