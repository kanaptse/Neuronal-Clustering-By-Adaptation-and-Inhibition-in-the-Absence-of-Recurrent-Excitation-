import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def passing_times_e(z_list,si,tau,g,delta,mu):
   dt = 0.001
   t_out = np.round(np.arange(0,200,dt),4)
   theta = np.zeros((len(t_out),len(z_list)))
   theta[0,:] = -np.arccos((1+mu['e']-g['ie']*si-g['z']*z_list)/(1-mu['e']+g['ie']*si+g['z']*z_list)) 
   for i in range(1,len(t_out)):
       t = t_out[i-1]
       integrant = 1-np.cos(theta[i-1,:])+(mu['e']-g['ie']*si*np.exp(-t/tau['si'])-g['z']*z_list*np.exp(-t/tau['z']))*(1+np.cos(theta[i-1,:]))
       theta[i,:] = theta[i-1,:] + dt/tau['me'] * integrant
   theta_df = pd.DataFrame(theta,index=t_out,columns=z_list)

   def get_index(col,target):
       return theta_df[col].gt(target).idxmax()

   tts = {col:get_index(col, np.pi) for col in theta_df.columns}
   tts_df = pd.DataFrame(tts,index=[0]).T

   return tts_df, theta_df

def tts_inv(tts_df,t):
    diff_df = t - tts_df  
    diff_df_positive = diff_df.where(diff_df > 0, np.inf)  
    z = diff_df_positive.idxmin().values
    return z[0]

def theta_intersect(theta_df,ti_f,si_f):
    closest_index_series = pd.DataFrame((theta_df.index - ti_f),index=theta_df.index).abs().idxmin()
    closest_index = float(closest_index_series.iloc[0])
    new_theta_df = pd.DataFrame(theta_df.loc[closest_index,:]).T
    new_theta_df.reset_index(inplace=True,drop=True)
    new_z_df = pd.DataFrame(np.array(theta_df.columns)*np.exp(-ti_f/tau['z'])).T
    new_z_df.columns = theta_df.columns
    thres_df = np.arccos((1+mu['e']-g['ie']*si_f-g['z']*new_z_df)/(1-mu['e']+g['ie']*si_f+g['z']*new_z_df)) 
    dist_from_thres_df = new_theta_df-thres_df
    new_df = pd.concat([new_theta_df,new_z_df,thres_df,dist_from_thres_df],axis=0)
    
    new_df.index = ['theta value when i cell fires', 'z value when i cell fires','threshold theta value','distance from threshold']
    diff_df_positive = new_df.loc['distance from threshold'].where(new_df.loc['distance from threshold'] > 0, np.inf) 

    z = diff_df_positive.idxmin()
    return z

def tts_i(P,tts_df,setot_old,tr):
    setot_old = setot_old * np.exp(-tr/tau['se'])
    dt = 0.05
    t = 0
    if tr == 0:
        theta = -np.arccos((1+mu['i'])/(1-mu['i']))
    else:    
        tr = round(tr/dt)*dt
        theta = -np.pi
    tmax = round(tts_df['tts'].max(),1)
    t_int_list = np.arange(0.1,tmax,dt)
    P_int_list = []
    for t_int in t_int_list:
        z = tts_inv(tts_df,t_int)
        P_int_list.append(P(z)*tts_df.loc[z,'dz/dt'])  
    t_int_list = t_int_list + tr
    int_df = pd.DataFrame(P_int_list)
    int_df.index = t_int_list

    while theta<np.pi:
        t += dt
        t = round(t,2)
        

        setot = setot_old * np.exp(-t/tau['se'])

        if t>tr:
            setot += (int_df.loc[:t,0]*dt*np.exp(-(t-int_df.loc[:t,0].index)/tau['se'])).rolling(2).mean().sum()

        integrant = 1-np.cos(theta)+(mu['i']+g['ei']*setot)*(1+np.cos(theta))   
        theta = theta + dt/tau['mi'] * integrant
    tf = t-tr
    return tf,setot

def integ(t_int,P,t):
    z = tts_inv(tts_df,t_int)
    return P(z)*tts_df.loc[z,'dz/dt']*np.exp(-(t-t_int)/tau['se']) 


def create_P_new(z_cut,ti_f,tr,P):
    def P_nf(z):
        if z>z_cut:
            return P(z)
        else:
            return 0
    def P_f(z):
        if z<=z_cut:
            return P(z)
        else:
            return 0
    def P_new(z):      
        return (P_nf(z*np.exp((ti_f+tr)/tau['z']))+P_f((z*np.exp(tr/tau['z'])-1)*np.exp(ti_f/tau['z'])))*np.exp((ti_f+tr)/tau['z'])
    return P_nf,P_f,P_new

# def cluster_sizes
def null_surface(THETA,SI,mu,g):
    return ((1-np.cos(THETA))/(1+np.cos(THETA))+mu['e']-g['ie']*SI)/g['z']

init_cond = 'uniform'

if init_cond == 'uniform':
    def P0(z):
        start = null_surface(0,si0,mu,g)
        end = null_surface(0,si0,mu,g)+2
        return float(start<z and z<end) /2
elif init_cond == 'triangle':
    def P0(z):
        start = null_surface(0,si0,mu,g)
        end = null_surface(0,si0,mu,g)+2
        return float(start<z and z<end) * (z-start) /(end-start)
elif init_cond == 'parabola':
    def P0(z):
        start = null_surface(0,si0,mu,g)
        end = null_surface(0,si0,mu,g)+2
        if start<z and z<end:
            return -3/4*(z-start)*(z-end)
        else:
            return 0
elif init_cond == 'custom':
    def P0(z):
        start = null_surface(0,si0,mu,g)
        first_cluster = start+1<z and z<start+1.2
        second_cluster = start+1.7<z and z<start+1.9
        return float(first_cluster or second_cluster)/0.4

if __name__ == "__main__":
    tau = {}
    tau['se'] = 3
    tau['si'] = 5
    tau['me'] = 1
    tau['mi'] = 1
    tau['z'] =120
    g = {}
    g['ei'] = 1
    g['ie'] = 1
    g['z'] = 0.6
    delta = {}
    delta['e'] = 0.01
    delta['i'] = 0.01
    mu = {}
    mu['e'] = 1
    mu['i'] =-0.01
    num_of_iter = 10

    start = time.time()
    
    si0 = 0.3
    z_start = round(null_surface(0,si0,mu,g)-0.1,1)
    z_list = np.arange(z_start,z_start+3,0.01).round(5)
    z_diff = np.diff(z_list)[0]
    
    tts_df, theta_df = passing_times_e(z_list,si0,tau,g,delta,mu)
    tts_df.columns = ['tts']
    tts_df['dz/dt'] = (z_diff/tts_df.diff())
    
    tts_df.dropna(inplace=True)
    tts_df = tts_df.replace(np.inf, 0)

    P_list = []
    P_pair_list = []
    P_list.append(P0)
    
    z_cut_list = []
    ti_f_list = []
    tr_list = []
    setot_list = []
       
    setot = 0
    tr = 0
    for i in range(1,num_of_iter):  
        print('Iteration', i)
        ti_f,setot = tts_i(P_list[i-1],tts_df,setot,tr)
        setot_list.append(setot)
        si_f = si0*np.exp(-ti_f/tau['si'])+1
        z_cut = theta_intersect(theta_df,ti_f,si_f)
        tr = tau['si']*np.log(np.exp(-ti_f/tau['si'])+1/si0)
        
        P_nf,P_f,P_new = create_P_new(z_cut,ti_f,tr,P_list[i-1])
        P_pair = [P_nf,P_f]
        P_pair_list.append(P_pair)
        
        z_cut_list.append(z_cut)
        ti_f_list.append(ti_f)
        tr_list.append(tr)
        P_list.append(P_new)
    
    map_time =  (pd.DataFrame(ti_f_list) + pd.DataFrame(tr_list)).cumsum()
    map_time = pd.concat([pd.DataFrame([0]), map_time]).reset_index(drop=True)
       
    end = time.time()
    print("Elapsed time:", end - start)    
    
    ti_f_list.insert(0,0)
    tr_list.insert(0,0)
    firing = map_time - pd.DataFrame(tr_list)
    
    
    plt.rcParams['font.size'] = 10

    fig, axs = plt.subplots(4,2)
    plt.subplots_adjust(wspace=0.1)
    i = 0

    num_rows = axs.shape[0]
    num_cols = axs.shape[1]
    
    for i, ax in enumerate(axs.flat):
        density = [P_list[i](z) for z in z_list]
        
        # Determine if the subplot is in the last row
        if i // num_cols == num_rows - 1:
            ax.set_xlabel('z')
        else:
            ax.set_xlabel('')
        
        # Determine if the subplot is in the first column
        if i % num_cols == 0:
            ax.set_ylabel('density')
        else:
            ax.set_ylabel('')
        
        if i > 0:
            ti_f = ti_f_list[i]
            tr = tr_list[i]
            
            P_nf_values = [P_pair_list[i-1][0](z*np.exp((ti_f+tr)/tau['z']))*np.exp((ti_f+tr)/tau['z']) for z in z_list]
            P_f_values = [P_pair_list[i-1][1]((z*np.exp(tr/tau['z'])-1)*np.exp(ti_f/tau['z']))*np.exp((ti_f+tr)/tau['z']) for z in z_list]
            
            ax.plot(z_list, P_nf_values, label=f'$P_{{{i-1}}}^{{nf}}$', linewidth=6, color="C0")
            ax.plot(z_list, P_f_values, label=f'$P_{{{i-1}}}^f$', linewidth=6, color="C2")
        
        ax.plot(z_list, density, label=rf'$P_{{{i}}}$', linewidth=2, color="C1")
        
        if i < len(z_cut_list):
            ax.axvline(x=z_cut_list[i], color='r', linestyle='--')
            
        ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
        
        # Place "Iteration i" in the top left corner of the plot
        ax.text(0.8, 0.95, f'Iteration {i}', transform=ax.transAxes, verticalalignment='top')

    fig1, axs1 = plt.subplots(4,2)
    i = 0
    for ax in axs1.flat:
        density = [P_list[i](z)*tts_df.loc[z,'dz/dt'] for z in z_list[1:]]
        
        ax.set_xlabel('time-to-spike (ms)') 
        ax.set_xlim([0,65])
        ax.set_ylabel(f'Iteration {i}')
        
        if i>0:
            ti_f = ti_f_list[i]
            tr = tr_list[i]
            
            P_nf_values = [P_pair_list[i-1][0](z*np.exp((ti_f+tr)/tau['z']))*np.exp((ti_f+tr)/tau['z'])*tts_df.loc[z,'dz/dt'] for z in z_list[1:]]
            P_f_values = [P_pair_list[i-1][1]((z*np.exp(tr/tau['z'])-1)*np.exp(ti_f/tau['z']))*np.exp((ti_f+tr)/tau['z'])*tts_df.loc[z,'dz/dt'] for z in z_list[1:]]
            
            
            ax.plot(tts_df['tts'].values,P_nf_values,label=rf'$P_{{{i-1}}}^{{nf}}$', linewidth=6, color="C0")
            ax.plot(tts_df['tts'].values,P_f_values,label=rf'$P_{{{i-1}}}^f$', linewidth=6, color="C2")
        ax.plot(tts_df['tts'].values,density,label=rf'$P_{i}$', linewidth=2, color="C1")
        if i<len(z_cut_list):
            if z_cut_list[i]<tts_df.index[0]:
                ax.axvline(x=0, color='r', linestyle='--')
            else:
                ax.axvline(x=tts_df.loc[z_cut_list[i],'tts'], color='r', linestyle='--')
        ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
    
        i += 1
    
    plt.show()


    

