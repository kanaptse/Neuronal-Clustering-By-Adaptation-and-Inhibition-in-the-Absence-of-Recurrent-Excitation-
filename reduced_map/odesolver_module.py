import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

np.set_printoptions(threshold=np.inf)

warnings.filterwarnings("ignore", message="invalid value encountered in arccos")


class ODESolver:
    
    def __init__(self, tau, g, mu, si, gz_target, z_range, t_max):
            self.tau = tau
            self.g = g
            self.mu = mu
            self.si = si
            self.gz_target = gz_target
            self.z_range = z_range
            self.t_max = t_max
        
    def passing_times_e(self, z_list):
        dt = 0.001
        t_out = np.round(np.arange(0, self.t_max, dt), 3)
        theta = np.zeros((len(t_out), len(z_list)))
        theta[0, :] = -np.arccos((1 + self.mu['e'] - self.g['ie'] * self.si - self.g['z'] * z_list) / (1 - self.mu['e'] + self.g['ie'] * self.si + self.g['z'] * z_list)) 
        
        for i in range(1, len(t_out)):
            t = t_out[i-1]
            integrant = 1 - np.cos(theta[i-1, :]) + (self.mu['e'] - self.g['ie'] * self.si * np.exp(-t/self.tau['si']) - self.g['z'] * z_list * np.exp(-t/self.tau['z'])) * (1 + np.cos(theta[i-1, :]))
            theta[i, :] = theta[i-1, :] + dt / self.tau['me'] * integrant

        theta_df = pd.DataFrame(theta, index=t_out, columns=z_list)
        
        def get_index(col, target):
            return theta_df[col].gt(target).idxmax()

        tts = {col: get_index(col, np.pi) for col in theta_df.columns}
        tts_df = pd.DataFrame(tts, index=[0]).T
        return tts_df, theta_df
    
    
    def null_surface(self,THETA):
        return ((1 - np.cos(THETA)) / (1 + np.cos(THETA)) + self.mu['e'] - self.g['ie'] * self.si) / self.g['z']
    
    def null_surface_theta_init(self,z):
        return -np.arccos((1+self.mu['e']-self.g['ie']*self.si-self.g['z']*z)/(1-self.mu['e']+self.g['ie']*self.si+self.g['z']*z))
    
    def null_surface_threshold(self,z):
        return np.arccos((1+self.mu['e']-self.g['ie']-self.g['z']*z)/(1-self.mu['e']+self.g['ie']+self.g['z']*z))

    @staticmethod
    def tts_inv(tts_df, t):
        diff_df = t - tts_df  
        diff_df_positive = diff_df.where(diff_df > 0, np.inf)  
        z = diff_df_positive.idxmin().values
        return z[0] 

    def tts_i(self):
        dt = 0.05
        t = 0
        theta = -0.1
        while theta < np.pi:
            t += dt
            t = round(t, 2)
            setot = np.exp(-t / self.tau['z'])
            integrant = 1 - np.cos(theta) + (self.mu['i'] + self.g['ei'] * setot) * (1 + np.cos(theta))   
            theta = theta + dt / self.tau['mi'] * integrant
        return t, setot
    
    def compute_tts(self):
        z_array = []
        g_z_array = []
        tts_array = []
        theta_df_dict = {}

        for g_val in self.gz_target:
            print(round(g_val, 4))
            self.g['z'] = g_val
            z_start = max(round(self.null_surface(0), 3), 0)
            z_list = np.arange(z_start + self.z_range[0], z_start+ self.z_range[1], 0.01).round(5)
            tts_df, theta_df = self.passing_times_e(z_list)
            z_array.extend(z_list)
            g_z_array.extend([g_val] * len(z_list))
            tts_array.extend(tts_df[0].values)
            theta_df_dict[g_val] = theta_df
            
        return np.array(z_array), np.array(g_z_array), np.array(tts_array), theta_df_dict

class Regressor:
    def __init__(self):
        pass

    @staticmethod
    def function_fit(z_array, g_z_array, tts_array, weight, func_type):
        if func_type == 'linear':
            X = np.column_stack([
                np.ones(z_array.shape),
                z_array * g_z_array
            ])  
            model = LinearRegression(fit_intercept=False).fit(X, tts_array, sample_weight=weight)
            coeff = model.coef_
        elif func_type == 'exponential':
            def exp_func(z_gz, a0, a1):
                z, gz = z_gz
                return a0 * np.exp(a1 * gz * z)
            
            z_gz = np.array([z_array, g_z_array])

            # Perform the exponential fit
            popt, pcov = curve_fit(exp_func, z_gz, tts_array, p0=[1, 1], sigma=1/weight)
            coeff = popt
        else:
            raise ValueError('func_type should be linear or exponential only!')
            
        return coeff

    @staticmethod
    def evaluate_fit(coeff, z, gz, func_type):
        if func_type == 'linear':
            return coeff[0] + coeff[1] * z * gz
        elif func_type == 'exponential':
            return coeff[0] * np.exp(coeff[1] * z * gz)


class Visualizer(Regressor):
    
    @staticmethod
    def plot_curve_fixed_z(z_value, z_array, g_z_array, tts_array):
        indices = np.where(np.isclose(z_array, z_value))[0]
        
        # Use these indices to filter g_z_array and tts_array
        filtered_gz = np.array(g_z_array)[indices]
        filtered_tts = np.array(tts_array)[indices]
        
        # Plot
        plt.figure()
        plt.plot(filtered_gz, filtered_tts, label=f"z = {z_value}")
        plt.xlabel("g['z']")
        plt.ylabel('Time to Spike (tts)')
        plt.legend()
        plt.title(f"Curve for fixed z = {z_value}")
        plt.show()

    @staticmethod
    def plot_curve_fixed_gz(gz_value, z_array, g_z_array, tts_array):
        # Get indices of g_z_array where the value matches gz_value
        indices = np.where(np.isclose(g_z_array, gz_value))[0]
        
        # Use these indices to filter z_array and tts_array
        filtered_z = np.array(z_array)[indices]
        filtered_tts = np.array(tts_array)[indices]
        
        # Plot
        plt.figure()
        plt.plot(filtered_z, filtered_tts, label=f"g['z'] = {gz_value}")
        plt.xlabel('z_list')
        plt.ylabel('Time to Spike (tts)')
        plt.legend()
        plt.title(f"Curve for fixed g['z'] = {gz_value}")
        plt.show()

    def plot_original_and_interpolated(self, coeff, z_array, g_z_array, tts_array, func_type, font_size=20):
        # Convert arrays to numpy arrays for manipulation
        z_grid, g_z_grid = np.meshgrid(np.linspace(min(z_array), max(z_array), 100), 
                                       np.linspace(min(g_z_array), max(g_z_array), 100))
        
        tts_grid = self.evaluate_fit(coeff, z_grid, g_z_grid, func_type)
        
        # Plot the surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        z_grid[tts_grid>40] = None
        g_z_grid[tts_grid>40] = None
        tts_grid[tts_grid>40] = None
    
        # Plot the interpolated surface
        surf = ax.plot_surface(z_grid, g_z_grid, tts_grid, cmap=plt.cm.viridis, edgecolor='none', alpha=0.7)
        
        
        
        # Scatter plot of original data
        ax.scatter(z_array, g_z_array, tts_array, c='red', marker='o')
    
        # Set labels with font size
        ax.set_xlabel('z', fontsize=font_size)
        ax.set_ylabel(r"$g_z$", fontsize=font_size)
        ax.set_zlabel('TTS (ms)', fontsize=font_size)
        
        # ax.set_zlim(bottom=0, top=40)
        

        # Colorbar with font size
        cbar = fig.colorbar(surf)
        cbar.ax.tick_params(labelsize=font_size) 
    
        plt.show()

    @staticmethod
    def plot_for_gz(gz_value, z_array, g_z_array, tts_array, predicted_values):
        z_array_np = np.array(z_array)
        g_z_array_np = np.round(np.array(g_z_array),4)
        tts_array_np = np.array(tts_array)
        predicted_values_np = np.array(predicted_values)

        # Extract data for the specific gz_value
        mask = g_z_array_np == gz_value
        z_filtered = z_array_np[mask]
        tts_filtered = tts_array_np[mask]
        predicted_filtered = predicted_values_np[mask]

        # Plot
        plt.figure(figsize=(10,6))
        plt.plot(z_filtered, tts_filtered, 'o', label='Actual Data', color='blue')
        plt.plot(z_filtered, predicted_filtered, 'x', label='Predicted Values', color='red')
        plt.title(f'Plot for gz = {gz_value}')
        plt.xlabel('z')
        plt.ylabel('TTS')
        plt.legend()
        plt.grid(True)
        plt.show()


    @staticmethod
    def plot_subplots_for_gz(z_array, g_z_array, tts_array, predicted_values, font_size=10):
        g_vals = np.unique(g_z_array)[::]
        g_vals = np.round(g_vals, 4)
        num_g_vals = len(g_vals)
        
        rows = int(np.ceil(num_g_vals / 2))
        fig1, axs1 = plt.subplots(rows, 2, figsize=(15, 6*rows))
        
        if rows == 1:
            axs1 = np.array([axs1])
        
        for i in range(rows):
            for j in range(2):
                idx = 2*i + j
                if idx < num_g_vals:
                    g_val = g_vals[idx]
    
                    z_array_np = np.array(z_array)
                    g_z_array_np = np.round(np.array(g_z_array), 4)
                    tts_array_np = np.array(tts_array)
                    predicted_values_np = np.array(predicted_values)
    
                    mask = g_z_array_np == g_val
                    z_filtered = z_array_np[mask]
                    tts_filtered = tts_array_np[mask]
                    predicted_filtered = predicted_values_np[mask]
    
                    axs1[i, j].plot(z_filtered, tts_filtered, 'o', label='From ODE', color='red')
                    axs1[i, j].plot(z_filtered, predicted_filtered, 'x', label='Fit', color='blue')
                    axs1[i, j].set_title(r'$g_z$ = {:.2f}'.format(g_val), fontsize=font_size)
                    axs1[i, j].tick_params(axis='both', labelsize=font_size)
                    if j == 0:
                        axs1[i, j].set_ylabel('TTS (ms)', fontsize=font_size)
                    if i == rows - 1:                      
                        axs1[i, j].set_xlabel('z', fontsize=font_size)
                    # axs1[i, j].legend(fontsize=font_size)
                    axs1[i, j].grid(True)
    

        
        if num_g_vals % 2 != 0:
            fig1.delaxes(axs1[rows-1, 1])
    
        fig1.tight_layout()
        plt.show()

    




