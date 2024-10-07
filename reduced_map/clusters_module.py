import numpy as np
import matplotlib.pyplot as plt
from spiking_network.facade import theta_ei
import pandas as pd
from spiking_network.community_detection import Louvain
import sys
from sympy import symbols, Eq, solve
import os

current_dir = os.path.dirname(__file__) 
up_one_level = os.path.dirname(current_dir)

class ClusterBase:
    def __init__(self, num_clusters, tau, g, mu, si0, coefficients):
        self.num_clusters = num_clusters
        self.tau = tau
        self.g = g
        self.mu = mu
        self.si0 = si0
        self.a0 = coefficients[0]
        self.a1 = coefficients[1]
        self.tts_i()

    def tts_i(self):
        dt = 0.05
        t = 0
        theta = -np.arccos((1+self.mu['i'])/(1-self.mu['i']))

        while theta < np.pi:
            t += dt
            t = round(t, 2)

            if t > 50:
                raise Exception("Time to spike of I cell exceeded a threshold of 50ms.")
            
            setot = np.exp(-t/self.tau['se'])/self.num_clusters
            integrant = 1 - np.cos(theta) + (self.mu['i'] + self.g['ei'] * setot) * (1 + np.cos(theta))
            theta = theta + dt/self.tau['mi'] * integrant

        self.ti_f = t
        tr = -self.tau['si']*np.log(self.si0)
        self.t_next = self.ti_f + tr

class ClustersCalculator(ClusterBase):
    def __init__(self, num_clusters, tau, g, mu, si0, coefficients, func_type):
        super().__init__(num_clusters, tau, g, mu, si0, coefficients)
        self.tts_i()
        self.func_type = func_type
         
    def tts_e(self, z, gz):
        if self.func_type == 'linear':
            return self.a0 + self.a1 * gz * z
        elif self.func_type =='exponential':
            return self.a0 * np.exp(self.a1 * gz * z)
        
    def tts_e_inv(self, t, gz):
        if self.func_type == 'linear':
            return (t - self.a0)/(self.a1 * gz)
        elif self.func_type =='exponential':
            return 1/(self.a1*gz) * np.log(t/self.a0)
    
    def update(self, gz, t, *ds):
        # New logic to update based on number of clusters
        if self.num_clusters == 2:
            d = ds[0]
            t_new = self.tts_e(self.tts_e_inv(t+d, gz)*np.exp(-(t+self.t_next)/self.tau['z']), gz)
            d_new = self.tts_e((self.tts_e_inv(t, gz)*np.exp(-t/self.tau['z'])+1)*np.exp(-self.t_next/self.tau['z']), gz) - t_new
            return t_new, d_new
        elif self.num_clusters == 3:
            d1, d2 = ds
            t_new = self.tts_e(self.tts_e_inv(t+d1, gz)*np.exp(-(t+self.t_next)/self.tau['z']), gz)
            d1_new = self.tts_e(self.tts_e_inv(t+d1+d2, gz)*np.exp(-(t+self.t_next)/self.tau['z']), gz) - t_new
            d2_new = self.tts_e((self.tts_e_inv(t, gz)*np.exp(-t/self.tau['z'])+1)*np.exp(-self.t_next/self.tau['z']), gz) - (t_new + d1_new)            
            return t_new, d1_new, d2_new

    def get_steady_state(self, gz, max_iterations=1000, tolerance=1e-3):
        # Initial values based on number of clusters
        if self.num_clusters == 2:
            states = [5, 8]
        elif self.num_clusters == 3:
            states = [10, 8, 8]
    
        for _ in range(max_iterations):
            old_states = states.copy()
            states = list(self.update(gz, *states))
    
            # Check convergence
            if all(abs(new - old) < tolerance for new, old in zip(states, old_states)):
                break
    
        return states
    
    def theta_intersect(self, theta_df, t_ss, gz):
       ti_f_total = t_ss + self.ti_f
       closest_index_series = pd.DataFrame((theta_df.index - ti_f_total), index=theta_df.index).abs().idxmin()
       closest_index = float(closest_index_series.iloc[0])
       new_theta_df = pd.DataFrame(theta_df.loc[closest_index, :]).T
       new_theta_df.reset_index(inplace=True, drop=True)
       new_z_df = pd.DataFrame(np.array(theta_df.columns) * np.exp(-ti_f_total / self.tau['z'])).T
       new_z_df.columns = theta_df.columns
       thres_df = np.arccos((1 + self.mu['e'] - self.g['ie'] - gz * new_z_df) / (1 - self.mu['e'] + self.g['ie'] + gz * new_z_df))
       dist_from_thres_df = new_theta_df - thres_df
       new_df = pd.concat([new_theta_df, new_z_df, thres_df, dist_from_thres_df], axis=0)
       
       new_df.index = [r'theta value when i cell fires', 'z value when i cell fires', 'threshold theta value', 'distance from threshold']
       diff_df_positive = new_df.loc['distance from threshold'].where(new_df.loc['distance from threshold'] > 0, np.inf) 
   
       z = diff_df_positive.idxmin()
       
       t_cut = self.tts_e(z, gz)
       return t_cut
    
    def find_cutoff(self, gz_target, theta_df_dict):
        old_d = 0
        new_d = 0
        
        old_d_cut = 0
        new_d_cut = 0
        
        old_gz = 0
        
        for gz in gz_target:
            states = self.get_steady_state(gz)
            new_d = states[1]
            new_d_cut = self.theta_intersect(theta_df_dict[gz], states[0], gz) - states[0]
            if old_d_cut>old_d and new_d_cut<new_d:
                gz_pair = [old_gz, gz]
                d_pair = [old_d, new_d]
                d_cut_pair = [old_d_cut, new_d_cut]
                return float(self.calculate_cutoff_intersetion(gz_pair, d_pair, d_cut_pair))
            old_gz = gz
            old_d = new_d
            old_d_cut = new_d_cut
        raise Exception('Transition does not present in this gz range.')
        

    
    @staticmethod
    def calculate_cutoff_intersetion(gz_pair, d_pair, d_cut_pair):
        gz, d = symbols('gz d')
        
        m1 = (d_pair[1] - d_pair[0]) / (gz_pair[1] - gz_pair[0])
        line1 = Eq(d - d_pair[0], m1 * (gz - gz_pair[0]))
        
        m2 = (d_cut_pair[1] - d_cut_pair[0]) / (gz_pair[1] - gz_pair[0])
        line2 = Eq(d - d_cut_pair[0], m2 * (gz - gz_pair[0]))
        
        intersection = solve((line1, line2), (gz, d))
        gz_transition = intersection[gz]
        
        return gz_transition
        
        
        
        
class ClustersVisualizer(ClustersCalculator):
    def __init__(self, num_clusters, tau, g, mu, si0, coefficients, func_type):
        super().__init__(num_clusters, tau, g, mu, si0, coefficients, func_type)
    
    def plot_evolution(self, gz, max_iterations=1000, tolerance=1e-3, one_graph=False):
        # Initial setup based on number of clusters
        initial_values = {
            2: [10, 8],
            3: [10, 8, 8]
        }
        
        labels_dict = {
            2: [r'$t_n$', r'$d_n$'],
            3: [r'$t_n$', r'$d^1_n$', r'$d^2_n$']
        }
        
        colors = ['blue', 'red', 'green']
    
        states = initial_values[self.num_clusters]
        states_histories = [[val] for val in states]
        labels = labels_dict[self.num_clusters]
        
        for n in range(max_iterations):
            old_states = states.copy()
            states = list(self.update(gz, *states))
            
            for idx, state in enumerate(states):
                states_histories[idx].append(state)
            
            # Check convergence
            if all(abs(new - old) < tolerance for new, old in zip(states, old_states)):
                break
    
        # Plotting
        
        font_size = 20  # You can set this to the desired value
        dot_size = 200
        
        if one_graph:
            if self.num_clusters == 2:
                plt.figure(figsize=(15, 10))
                plt.plot(states_histories[0], states_histories[1], zorder=1)
                plt.xlabel(f'{labels[0].split()[0]}', fontsize=font_size)
                plt.ylabel(f'{labels[1].split()[0]}', fontsize=font_size)
                plt.title(f"Evolution of a {self.num_clusters}-cluster map", fontsize=font_size)
                plt.tick_params(axis='both', labelsize=font_size)  # Set tick font size
                plt.scatter(states_histories[0][0], states_histories[1][0], marker='o', color='red', label='initial state', zorder=2, s=dot_size)
                plt.scatter(states_histories[0][-1], states_histories[1][-1], marker='o',color='black', label='steady state', zorder=2, s=dot_size)
                plt.legend(fontsize=font_size)
                target_path = os.path.join(up_one_level, 'figures', '2lust_evol.png')
                plt.savefig(target_path)
            elif self.num_clusters == 3:
                fig = plt.figure(figsize=(15, 10))
                ax = fig.add_subplot(111, projection='3d')
            
        
                # Plotting data
                ax.plot3D(states_histories[0], states_histories[1], states_histories[2], label='trajectory')
                ax.plot3D(states_histories[0], states_histories[1], np.zeros(len(states_histories[2])), label='2D projection')

                ax.scatter(states_histories[0][0], states_histories[1][0], 
                           states_histories[2][0], marker='o', color='red', label='initial state',
                           s=dot_size, zorder=2)
                ax.scatter(states_histories[0][-1], states_histories[1][-1], states_histories[2][-1],
                           marker='o', color='black', label='steady state', 
                           s=dot_size, zorder=2)
                
                
                # Labels, titles, and tick font size
                ax.set_xlabel(f'{labels[0].split()[0]}', fontsize=font_size)
                ax.set_ylabel(f'{labels[1].split()[0]}', fontsize=font_size)
                ax.set_zlabel(f'{labels[2].split()[0]}', fontsize=font_size)
                ax.legend(fontsize=font_size)
                ax.set_title(f"Evolution of a {self.num_clusters}-cluster map", fontsize=font_size)
                for t in [ax.xaxis, ax.yaxis, ax.zaxis]:  # Set tick font size for 3D plot
                    t.label.set_fontsize(font_size)
                target_path = os.path.join(up_one_level, 'figures', '3lust_evol.png')
                
                plt.savefig(target_path)
        
        else:       
            plt.figure(figsize=(10, 5))
            for idx, (history, label, color) in enumerate(zip(states_histories, labels, colors)):
                plt.subplot(1, self.num_clusters, idx + 1)
                plt.plot(history, color=color, label=label)
                plt.title(f"Evolution of {label.split()[0]}", fontsize=font_size)
                plt.xlabel('Iteration', fontsize=font_size)
                plt.ylabel(f'{label.split()[0]} value', fontsize=font_size)
                plt.tick_params(axis='both', labelsize=font_size)  # Set tick font size
        
            plt.tight_layout()
            plt.show()
    
    def steady_state_vs_gz(self, gz_target, theta_df_dict, cutoff_only=True, font_size=30):
        # Initial setup based on number of clusters
        states_labels = {
            2: ['t steady state', r'$d_{ss}$'],
            3: ['t steady state', r'$d^1_{ss}$', r'$d^2_{ss}$']
        }
        
        colors = ['blue', 'red', 'orange', 'green']
    
        states_histories = {label: [] for label in states_labels[self.num_clusters]}
        t_cut_list = []
    
        for gz in gz_target:
            states = self.get_steady_state(gz)
            t_cut = self.theta_intersect(theta_df_dict[gz], states[0], gz) - states[0]
            
            for state, label in zip(states, states_labels[self.num_clusters]):
                states_histories[label].append(state)
    
            t_cut_list.append(t_cut)
    
        # Plotting
        plt.figure(figsize=(20, 12))
        linewidth=10
        if cutoff_only:
            idx = 1
            label = states_labels[self.num_clusters][1]
            plt.plot(gz_target, states_histories[label], label=label, color=colors[idx], linewidth=linewidth)
            title = r"$d_{ss}$ and $d_{ss}^{cut}$" if self.num_clusters == 2 else r"$d^1_{ss}$ and $d^{cut}_{ss}$"
        else:
            for idx, label in enumerate(states_labels[self.num_clusters]):
                plt.plot(gz_target, states_histories[label], label=label, color=colors[idx])
            title = "Steady State values vs gz" if self.num_clusters == 2 else "Steady State values of t, d1 and d2 vs gz"
            
        plt.plot(gz_target, t_cut_list, label=r"$d^{cut}_{ss}$", color='green', linestyle='--', linewidth=linewidth)
    
        # Set font sizes for title, x and y labels, and tick labels
        plt.title(title, fontsize=font_size)
        plt.xlabel(r"$g_z$", fontsize=font_size)
        plt.ylabel("t (ms)", fontsize=font_size)
        plt.legend(fontsize=font_size)
        plt.grid(True)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.show()

    
class SpikingModelComparison(ClustersCalculator):
    def __init__(self, num_clusters, T, tau, g, mu, si0, coefficients, func_type):
        super().__init__(num_clusters, tau, g, mu, si0, coefficients, func_type)
        self.T = T

    def run_and_plot(self, ax, gz, *args, show_xlabel=True, show_ylabel=True):
        self.g['z'] = gz
        num_cells = {'e':self.num_clusters, 'i':1}
        if self.num_clusters == 2:

            t, d = args
            tts = np.array([t, t + d])
        elif self.num_clusters == 3:
            t, d1, d2 = args
            tts = np.array([t, t + d1, t + d1 + d2])

        z_init = self.tts_e_inv(tts, gz)

        delta = {'e': 0, 'i': 0}
        tout, data_e_dict, _, _ = theta_ei(num_cells, self.T, self.tau, self.g, delta, self.mu, False, 1, z_init)

        ax.set_title(r"$g_z$ = {:.2f}".format(gz))
        for i in range(self.num_clusters):
            # ax.plot(tout, data_e_dict['z'][i, :], label=f'Cell e{i+1}')
            ax.plot(tout, data_e_dict['z'][i, :])
        if show_xlabel:
            ax.set_xlabel('time (ms)')
        if show_ylabel:
            ax.set_ylabel('z')

        # Place legend to the left of the subplot
        # ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    
    def comparison_plots(self, gz_target, font_size=30):
        if len(gz_target) == 2:
            num_rows = 2
            fig, axs = plt.subplots(num_rows, 1, figsize=(15, 5 * num_rows))
        else:
            num_rows = (len(gz_target) + 1) // 2  # Compute number of rows, rounding up
            fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))  # Create a grid of subplots
    
        # Flatten the axes array for easier indexing
        axs = axs.ravel()
    
        for idx, gz in enumerate(gz_target):
            args = self.get_steady_state(gz)
            self.run_and_plot(axs[idx], gz, *args)
            
            # Set font sizes for titles, x and y labels, and tick labels
            axs[idx].title.set_fontsize(font_size)
            axs[idx].xaxis.label.set_fontsize(font_size)
            axs[idx].yaxis.label.set_fontsize(font_size)
            axs[idx].tick_params(axis='both', which='major', labelsize=font_size)
            axs[idx].tick_params(axis='both', which='minor', labelsize=font_size - 2)
    
        # Remove any unused subplot axes
        try:
            for idx in range(len(gz_target), num_rows * 2):
                fig.delaxes(axs[idx])
        except:
            pass
        plt.tight_layout()
        plt.show()


    def cal_num_of_clust(self, gz, *args):
        self.g['z'] = gz
        num_cells = {'e':self.num_clusters, 'i':1}
        if self.num_clusters == 2:
            t, d = args
            tts = np.array([t, t + d])
        elif self.num_clusters == 3:
            t, d1, d2 = args
            tts = np.array([t, t + d1, t + d1 + d2])

        z_init = self.tts_e_inv(tts, gz)

        delta = {'e': 0, 'i': 0}
        tout, data_e_dict, _, _ = theta_ei(num_cells, self.T, self.tau, self.g, delta, self.mu, False, 1, z_init)
        # setot = np.sum(data_e_dict['s'],axis=0)/num_cells['e']
        # plt.figure()
        # plt.plot(tout, setot)
        
        dt = 0.05
        t_stable = int((self.T - 300)/dt)

        _, num_of_clusters_e, _ = Louvain(data_e_dict['z'][:, t_stable:].T)
    
        # is_periodic, _ = check_periodicity(np.array(setot[t_stable:].T))
        # return int(num_of_clusters_e), is_periodic
        out_of_phase = are_rows_out_of_phase(data_e_dict['z'][:, t_stable:])
        # print('out of phase:', out_of_phase)
        return int(num_of_clusters_e), out_of_phase
        

    def cal_num_of_clust_list(self, gz_target):
        num_of_clust_list = []
        out_of_phase_list  = []
        for gz in gz_target:
            args = self.get_steady_state(gz)
            num_of_clusters_e, out_of_phase = self.cal_num_of_clust(gz, *args)
            num_of_clust_list.append(num_of_clusters_e)
            out_of_phase_list.append(out_of_phase)
        return np.array(num_of_clust_list), np.array(out_of_phase_list)
      
        # num_of_clust_list = []
        # is_periodic_list  = []
        # for gz in gz_target:
        #     args = self.get_steady_state(gz)
        #     num_of_clusters_e, is_periodic = self.cal_num_of_clust(gz, *args)
        #     num_of_clust_list.append(num_of_clusters_e)
        #     is_periodic_list.append(is_periodic)
        # return np.array(num_of_clust_list), np.array(is_periodic_list)
      
    
    def find_transition_gz(self, gz_target):
        """
        Find the tranisition gz value having number of clusters from self.num_clusters-1 to self.num_clusters. 
        Right after transition, the solution should be having self.num_clusters clusters and the activities of 
        neurons are equal but just out of phase.
        """
        num_of_clust_array, out_of_phase_array = self.cal_num_of_clust_list(gz_target)
        desired_clust_num_array = num_of_clust_array == self.num_clusters
        
        find_transition_list = (desired_clust_num_array * out_of_phase_array).tolist()
        result, transition_index = analyze_bool_list(find_transition_list)
        print(find_transition_list, result, transition_index)
        return result, transition_index
    
def are_rows_out_of_phase(arr, average_difference_threshold=1e-3, max_shift_fraction=1/3):
    """
    Check if the rows of a 2D numpy array are equal but out of phase with non-circular shifts,
    using the average difference.

    Parameters:
    arr (numpy array): 2D numpy array.
    average_difference_threshold (float): Maximum allowed average difference to consider rows equal.
    max_shift_fraction (float): Maximum fraction of the row length to shift.

    Returns:
    bool: True if all rows are out of phase, False otherwise.
    """
    num_rows = arr.shape[0]
    row_length = arr.shape[1]
    max_shift = int(row_length * max_shift_fraction)

    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            row1 = arr[i]
            row2 = arr[j]

            for shift in range(-max_shift, max_shift + 1):
                if shift > 0:
                    shifted_row1 = row1[shift:]
                    compared_row2 = row2[:len(shifted_row1)]
                elif shift < 0:
                    shifted_row1 = row1[:shift]
                    compared_row2 = row2[-len(shifted_row1):]
                else:
                    shifted_row1 = row1
                    compared_row2 = row2

                average_difference = np.mean(np.abs(shifted_row1 - compared_row2))
                if average_difference <= average_difference_threshold:
                    break
            else:
                # If no shift was found that makes the rows equal within the average difference threshold, return False
                return False
    return True

def check_periodicity(arr, tol=1e-3):
    """
    Check if a 1D numpy array is almost fully periodic within a tolerance.
    
    :param arr: 1D numpy array.
    :param tol: Tolerance for floating point comparisons.
    :return: Tuple (is_periodic, chunk_size).
    """
    n = len(arr)

    for chunk_size in range(1, n // 2 + 1):
        chunk = arr[:chunk_size]
        times_to_repeat = n // chunk_size
        repeated_chunk = np.tile(chunk, times_to_repeat)
        if np.allclose(repeated_chunk, arr[:times_to_repeat * chunk_size], atol=tol):
            return True, chunk_size

    return False, None


def analyze_bool_list(bool_list):
    last_false = -1
    transition_found = False

    for i, value in enumerate(bool_list):
        if value is True:
            if last_false != -1:
                transition_found = True
        elif value is False:
            if transition_found:
                # A False is found after a True, which is not allowed

                return "Invalid sequence", None
            last_false = i

    if transition_found:
        return 'Found transition', last_false
    elif last_false == len(bool_list) - 1:
        return 'All false', None
    else:
        return 'All true', None

def calculate_max_errors(arr, tol=1e-3):
    n = len(arr)
    max_errors = []
    for chunk_size in range(1, n // 2 + 1):
        if n % chunk_size == 0:
            chunk = arr[:chunk_size]
            repeated_chunk = np.tile(chunk, n // chunk_size)
            max_error = np.max(np.abs(repeated_chunk - arr))
            max_errors.append(max_error)
    return max_errors

def plot_max_errors(arr, tol=1e-3):
    max_errors = calculate_max_errors(arr, tol)
    chunk_sizes = range(1, len(arr) // 2 + 1)
    plt.plot(chunk_sizes, max_errors, marker='o')
    plt.xlabel('Chunk Size')
    plt.ylabel('Maximum Error')
    plt.title('Maximum Error vs Chunk Size')
    plt.show()