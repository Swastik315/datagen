
#------------------Package for CIA - sigma_cia --------------------------
import h5py
import numpy as np
from closest_index import closest_index, prior_index


# Reading both H2 and He
def read_CIA(wl_model, T_fine, input_path, cia_species = ['H2-H2','H2-He']):
    
    cia_file = h5py.File(input_path + 'Opacity_database_cia.hdf5','r')

    interpolated_cia = np.zeros((len(cia_species), len(T_fine), len(wl_model)))
    
    for q, spec_q in enumerate(cia_species):
        
        T_q_cia = np.array(cia_file[spec_q + '/T'])
        nu_q_cia = np.array(cia_file[spec_q + '/nu'])
        log_sigma_q_cia = np.array(cia_file[spec_q + '/log(cia)'])
        
        pre_cia_interp = sigma_subset_wl(wl_model, log_sigma_q_cia, nu_q_cia)
        
        interpolated_cia [q,:,:] = T_interp_cia(pre_cia_interp, T_q_cia, T_fine)
        
    cia_file.close()
    
    return interpolated_cia
        
     
        
'''
#TEST

input_path = "/home/seps05/Desktop/POSEIDON/inputs/opacity/"

from tau_vert_update import wl_grid, generate_fine_pressure_grid, generate_fine_temperature_grid, \
                            generate_pressure_grid
                            
wl_min = 0.5
wl_max = 5.0
R =1000
wl_model = wl_grid(wl_min, wl_max, R)

T_fine_min = 400
T_fine_max = 2000
T_fine_step = 10
T_fine = generate_fine_temperature_grid(T_fine_min, T_fine_max, T_fine_step)    
 
sigma = read_CIA(wl_model, T_fine, input_path)   
'''

# only one species reading
def read_cia_alt(cia_pair='H2-He'):
    
    '''
    Function to read the collision-induced-absorption opacity file.

    Parameters:
        cia_pair (str): Name of the CIA pair to read from the file.

    Returns:
        tuple: Tuple containing Temperature_cia, log_sigma_cia, and nu_cia arrays.
    '''

    with h5py.File("/home/seps05/Desktop/POSEIDON/inputs/opacity/Opacity_database_cia.hdf5", 'r') as f:

        # chm_spec = list(f.keys())
    
        dset_select = f[cia_pair]
    
    
        Temp = dset_select['T']
        Temperature_cia = Temp[:]
    
        z = dset_select['log(cia)']
        log_sigma_cia = z[:]
    
        s = dset_select['nu']
        nu_cia = s[:]     
    
    return Temperature_cia, log_sigma_cia, nu_cia
 


# April 5
# Interpolating for CIA

from interpolation_package import nearest_index

def extracted_log_cia(closest_index, log_sigma_q):
    
    '''
    Author: Tonmoy Deka
    Parameters: 
        closest_index: array of closest indices of the nu_model_grid values.
        log_sigma_q: original log_sigma cross-section values.
        
    Returns: 
        log_sigma array but only for the model fine wavenumber grid. On this we perform
        interpolation on pressure and temperature axis.
    '''
    
    sigma_subset = log_sigma_q[ :, closest_index]
    
    
    return sigma_subset


def sigma_subset_wl(wl_model, log_sigma_cia, nu_cia):
    
    '''
  Extract sigma values based on model wavelength grid.

  Parameters:
      wl_model (array): Model wavelength array.
      log_sigma_cia (array): Log_sigma cross-section values.
      nu_cia (array): Wavenumber array.

  Returns:
      array: Sigma subset array based on model wavelength grid.
     '''
    
    nu_model = 1.0e4/wl_model
    
    N_nu_cia = len(nu_cia)
    
    N_nu = len(nu_model)
    
    closest_index = []
    
    for k in range(N_nu):
        
        closest_index.append(nearest_index(nu_model[k], nu_cia[0], nu_cia[-1], N_nu_cia))
    
    sigma_cia_subset = extracted_log_cia(closest_index, log_sigma_cia)
    
    return sigma_cia_subset


def T_interp_cia(sigma_cia_subset, Temperature_cia , T_fine):
    
    '''
    Perform T interpolation using numpy temperature interpolation.

    Parameters:
        sigma_cia_subset (array): Subset of sigma values.
        Temperature_cia (array): Temperature array.
        T_fine (array): Fine temperature array for interpolation.

    Returns:
        array: Interpolated sigma values based on temperature grid.
    '''
    
    N_T_old, N_nu = sigma_cia_subset.shape
    N_T_fine = len(T_fine)

    sigma_cia = np.zeros((N_T_fine, N_nu))

    for j in range(N_nu):
        sigma_cia[:, j] = np.interp(T_fine, Temperature_cia, sigma_cia_subset[:, j], left=0.0, right=0.0)

    return 10**sigma_cia

    
    




