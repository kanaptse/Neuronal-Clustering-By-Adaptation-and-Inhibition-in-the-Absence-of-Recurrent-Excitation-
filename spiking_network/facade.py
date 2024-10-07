import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.csgraph import connected_components
import pandas as pd
import json


def theta_ei(num_cells,T,tau,g,delta,mu,quenched,seed,z_init=None,same_as_z=False,all2all = True):
    # print("setting up parameters")    
    rng = np.random.RandomState(seed)
    
    in_degree = 50
    
    stim_start = 0
    stim_end = 10000
    
    # connectivity matrix
    if all2all:
        ee = np.ones((num_cells['e'], num_cells['e'])) / num_cells['e']
        ii = np.ones((num_cells['i'], num_cells['i'])) / num_cells['i']
        ei = np.ones((num_cells['i'], num_cells['e'])) / num_cells['e']
        ie = np.ones((num_cells['e'], num_cells['i'])) / num_cells['i']
    else:
        ee = np.zeros((num_cells['e'], num_cells['e']))
        ie = np.ones((num_cells['e'], num_cells['i']))
        for i in range(num_cells['e']):
            ee[i, :] = rng.rand(1, num_cells['e']) * (in_degree / num_cells['e'])
            ie[i, :] = rng.rand(1, num_cells['i']) * (in_degree / num_cells['i'])
        ee[ee != 0] = 1 / in_degree
        ie[ie != 0] = 1 / in_degree
    
        ii = np.zeros((num_cells['i'], num_cells['i']))
        ei = np.zeros((num_cells['i'], num_cells['e']))
        for i in range(num_cells['i']):
            ii[i, :] = rng.rand(1, num_cells['i']) * (in_degree / num_cells['i'])
            ei[i, :] = rng.rand(1, num_cells['e']) * (in_degree / num_cells['e'])
        ii[ii != 0] = 1 / in_degree
        ei[ei != 0] = 1 / in_degree
    
    if quenched:
        ne = np.tan(np.pi * (rng.rand(num_cells['e']) - 1/2))
        ni = np.tan(np.pi * (rng.rand(num_cells['i']) - 1/2))
    else:
        ne = np.zeros(num_cells['e']) 
        ni = np.zeros(num_cells['i'])
    # set up numerics
    dt = 0.05
    
    # set up dataframes for the variables
    maxtimes = int(T/ dt)
    tout = np.array(range(maxtimes))*dt
    
    theta_e = np.zeros((num_cells['e'],maxtimes))
    theta_i = np.zeros((num_cells['i'],maxtimes))
    firing_e = np.zeros((num_cells['e'],maxtimes))
    firing_i = np.zeros((num_cells['i'],maxtimes))
    
    se = np.zeros((num_cells['e'],maxtimes))
    si = np.zeros((num_cells['i'],maxtimes))
    
    z = np.zeros((num_cells['e'],maxtimes))
    
       
    # initialize
    
    theta_i[:,0] = 2*np.pi*np.random.rand(num_cells['i'])-np.pi
    si[:,0] = np.random.rand(num_cells['i'])
    
    def null_surface(THETA,SI):
        return ((1-np.cos(THETA))/(1+np.cos(THETA))+mu['e']-g['ie']*SI)/g['z']

    def null_surface_inverse(Z,SI):
        return  -np.arccos((1+mu['e']-g['ie']*SI-g['z']*Z)/(1-mu['e']+g['ie']*SI+g['z']*Z))
    
    if isinstance(z_init,np.ndarray):
        si[:,0] = 0.3
        z[:,0] = z_init
        theta_e[:,0] = null_surface_inverse(z[:,0],si[:,0])
        print(theta_e[:,0])
        theta_i[:,0] = -np.arccos((1+mu['i'])/(1-mu['i']))
    
    elif z_init=='ordered':
        si[:,0] = 0.3
        z_0 = null_surface(0,0.3)
        z[:,0] = np.arange(num_cells['e'])/num_cells['e']*2 + z_0
        theta_e[:,0] = null_surface_inverse(z[:,0],si[:,0])
        theta_i[:,0] = -np.arccos((1+mu['i'])/(1-mu['i']))
    elif z_init=='random':
        z[:,0] = np.random.rand(num_cells['e'])
        theta_e[:,0] = np.random.rand(num_cells['e'])*2*np.pi - np.pi
        if num_cells['i'] == 1:
            si[:,0] = 0.3
    else:
        z[:,0] = np.zeros(num_cells['e'])
    if same_as_z:
        theta_e[:,0] = 2*np.pi*z[:,0]-np.pi
        se[:,0] = z[:,0]

    # set up the output arrays
    
    # print("starting simulation")
    
    if quenched:
        noise_e = delta['e'] * ne
        noise_i = delta['i'] * ni
    
    for i in range(1,maxtimes):
        if stim_start < tout[i] and tout[i] < stim_end:
            mu_on = 1
        else:
            mu_on = 0
        
        
        if not quenched:
            noise_e = np.random.normal(scale=np.sqrt(delta['e']),size=num_cells['e'])
            noise_i = np.random.normal(scale=np.sqrt(delta['i']),size=num_cells['i'])
        
        if all2all:
            syn_input_e =  g['ee'] * np.sum(se[:,i-1])/num_cells['e'] - g['ie'] * np.sum(si[:,i-1])/num_cells['i']
            syn_input_i =  g['ei'] * np.sum(se[:,i-1])/num_cells['e'] - g['ii'] * np.sum(si[:,i-1])/num_cells['i']
        else:
            syn_input_e =  g['ee'] * (ee @ se[:,i-1]) - g['ie']* (ie @ si[:,i-1])
            syn_input_i =  g['ei'] * (ei @ se[:,i-1]) - g['ii'] * (ii @ si[:,i-1])
            
        integrant_e = 1 - np.cos(theta_e[:,i-1]) + (noise_e + mu_on*mu['e'] + syn_input_e - g['z']* z[:,i-1]) *(1 + np.cos(theta_e[:,i-1]))
        integrant_i = 1 - np.cos(theta_i[:,i-1]) + (noise_i + mu_on*mu['i'] + syn_input_i) * (1 + np.cos(theta_i[:,i-1]))

        theta_e[:,i] = theta_e[:,i-1] + dt/tau['me'] * integrant_e
        theta_i[:,i] = theta_i[:,i-1] + dt/tau['mi'] * integrant_i
        
        se[:,i] = se[:,i-1] + (-1)* dt * se[:,i-1] / tau['se'] + (theta_e[:,i]>np.pi)
        si[:,i] = si[:,i-1] - dt * si[:,i-1] / tau['si'] + (theta_i[:,i]>np.pi)
        z[:,i] = z[:,i-1] - dt * z[:,i-1] / tau['z'] + (theta_e[:,i]>np.pi)
        firing_e[:,i] = theta_e[:,i]>np.pi
        firing_i[:,i] = theta_i[:,i]>np.pi
  
        
        theta_e[:,i] = -2*np.pi*(theta_e[:,i]>np.pi) + theta_e[:,i]
        theta_i[:,i] = -2*np.pi*(theta_i[:,i]>np.pi) + theta_i[:,i]
            
    firing_e_df = pd.DataFrame(firing_e)
    firing_i_df = pd.DataFrame(firing_i)
    data_e_dict = {'s':se,'theta':theta_e,'firing':np.array(firing_e_df),'noise':ne,'z':z}

    data_i_dict = {'s':si,'theta':theta_i,'firing':np.array(firing_i_df),'noise':ni}
    con_mat = {'ee':ee,'ii':ii,'ei':ei,'ie':ie}
    return tout, data_e_dict, data_i_dict, con_mat

def clusters_saving(data_dict,num_of_cells,num_of_clusters, labels):
    cluster_dict = {}
    total_info_list = []
    for i in np.unique(labels):
        cluster_dict[i]={}
        single_cluster_info = []
        for item in data_dict:
            cluster_dict[i][item] = data_dict[item][np.where(labels==i)[0]] 
            if item == 's':
                single_cluster_info.append(len(cluster_dict[i][item]))
                single_cluster_info.append(len(cluster_dict[i][item])/num_of_cells)
            if item == 'noise':
                single_cluster_info.append(max(abs(cluster_dict[i][item])))
                single_cluster_info.append(np.mean(abs(cluster_dict[i][item])))
        total_info_list.append(single_cluster_info)
    total_info_df = pd.DataFrame(total_info_list,columns=['cluster size','relative size','max noise','mean noise'])
    return cluster_dict, total_info_df

def reordering(clusters_dict, labels):
    reordered_dict = {}
    
    for i in np.unique(labels):
        if i == 0:
            for item in clusters_dict[0]:
                reordered_dict[item] = clusters_dict[0][item]
        else:
            for item in clusters_dict[i]:
                reordered_dict[item] = np.append(reordered_dict[item],clusters_dict[i][item],axis=0)
    return reordered_dict

def activities_plot(cell_type,tout,stot,reordered_dict,g,num_cells,variable,plot_jump):  
    title1 = ""
    title2 = ""
    if cell_type == 'e':
        title1 = f"setot\n gz:{round(g['z'],3)}, gei:{round(g['ei'],3)}, gie:{round(g['ie'],3)}, gee:{round(g['ee'],3)}\n Ecells:{num_cells['e']}, Icells:{num_cells['i']}"
        title2 = "se"
    elif cell_type == 'i':
        title1 = f"sitot\n gz:{round(g['z'],3)}, gei:{round(g['ei'],3)}, gie:{round(g['ie'],3)}, gee:{round(g['ee'],3)}\n Ecells:{num_cells['e']}, Icells:{num_cells['i']}"
        title2 = "si"

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(tout, stot)
    plt.title(title1)

    id_ = np.arange(len(reordered_dict[variable]))
    
    if variable == 'firing':
        # Scatter plot for 'firing'
        plt.subplot(2, 1, 2)
        y, x = np.where(reordered_dict[variable] == 1)
        plt.scatter(tout[x], y, s=1)  # Adjust the size of points with 's' if needed
        plt.title(title2 + " (Firing Pattern)")
    else:
        # Original pcolor plot for other variables
        plt.subplot(2, 1, 2)
        h = plt.pcolor(tout[::plot_jump], id_, np.clip(reordered_dict[variable][:, ::plot_jump], 0, 1), shading='auto')
        plt.title(title2)
        plt.setp(h, edgecolor='none')

    plt.xlabel('time (ms)')
    plt.ylabel('identity of neuron')
    plt.show()

    
def clusters_activity_plot(clusters_dict,tout,variable,number):
    plt.figure()
    id_ = []
    for i in range(number):
        plt.plot(tout,np.sum(clusters_dict[i][variable],axis=0)/len(clusters_dict[i][variable]))
        id_.append(i)
    plt.legend(id_)
    plt.title(variable)
    
def save_to_dict(file_name,num_cells,T,tau,g,delta,mu,quenched,seed,z_init=None,same_as_z=False):
    tout, data_e_dict, data_i_dict, con_mat = theta_ei(num_cells,T,tau,g,delta,mu,quenched,seed,z_init,same_as_z)
    
    data_dict = {}
    data_dict['input'] = {}
    data_dict['output'] = {}
    
    data_dict['input']['num_cells'] = num_cells
    data_dict['input']['tau'] = tau
    data_dict['input']['g'] = g
    data_dict['input']['delta'] = delta
    data_dict['input']['mu'] = mu
    data_dict['output']['tout'] = tout
    data_dict['output']['tout'] = tout.tolist()
    for key in data_e_dict:
        data_e_dict[key] = data_e_dict[key].tolist()
    for key in data_i_dict:
        data_i_dict[key] = data_i_dict[key].tolist()
    data_dict['output']['data_e_dict'] = data_e_dict
    data_dict['output']['data_i_dict'] = data_i_dict
    
    with open('data/'+file_name, 'w') as f:
        json.dump(data_dict, f)
        
def read_dict(file_name):
    with open('data/'+file_name, 'r') as f:
        data_dict = json.load(f)
    params = data_dict['input']
    tout = np.array(data_dict['output']['tout'])
    data_e_dict = data_dict['output']['data_e_dict']
    for key in data_e_dict:
        data_e_dict[key] = np.array(data_e_dict[key])
    data_i_dict = data_dict['output']['data_i_dict']
    for key in data_i_dict:
        data_i_dict[key] = np.array(data_i_dict[key])
    return params, tout, data_e_dict, data_i_dict

def plot_e_i_raster(tout, reordered_dict_e, data_i_dict, font_size=30):
    dot_size = 10
    plt.figure(figsize=(20, 12))

    # Extract the indices of firing events for E and I cells
    y_e, x_e = np.where(reordered_dict_e == 1)
    y_i, x_i = np.where(data_i_dict == 1)

    # Offset for I cells to plot them at the bottom
    num_i_cells = data_i_dict.shape[0]
    y_e_offset = y_e + num_i_cells

    # Plot E cells in blue
    plt.scatter(tout[x_e], y_e_offset, color='blue', s=dot_size, label='E cells')

    # Plot I cells in red
    plt.scatter(tout[x_i], y_i, color='red', s=dot_size, label='I cells')

    # Set title, labels, and legend with the specified font size
    plt.title("Raster Plot of E and I Cells", fontsize=font_size)
    plt.xlabel('Time (ms)', fontsize=font_size)
    plt.ylabel('Neuron Index', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # plt.legend(fontsize=font_size)

    plt.show()


    
    





