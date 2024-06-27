import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
  

def read_fastchem_grid(chem_species, input_path='/home/seps05/POSEIDON/inputs/chemistry_grids/fastchem_database.hdf5'):
    """
    Reads the FastChem grid from the specified HDF5 file and loads the data for the given chemical species.

    Parameters:
    - chem_species: list of chemical species to load from the grid.
    - input_path: path to the HDF5 file containing the FastChem grid data.

    Returns:
    - fastchem_grid: dictionary containing the loaded grid data.
    """
    # Open the HDF5 file
    data = h5py.File(input_path, 'r')
    
    # Load the grids from the HDF5 file
    T_grid = np.array(data['Info/T grid'])
    P_grid = np.array(data['Info/P grid'])
    Metal_grid = np.array(data['Info/M/H grid'])
    c_to_o_grid = np.array(data['Info/C/O grid'])
    
    # Initialize the log_X_grid array to store the chemical species data
    log_X_grid = np.zeros((len(chem_species), len(Metal_grid), len(c_to_o_grid),
                           len(T_grid), len(P_grid)))
    
    # Load the data for each chemical species
    for q, chem in enumerate(chem_species):
        raw_array = np.array(data[f'{chem}/log(X)'])
        reshaped_array = raw_array.reshape(len(Metal_grid), len(c_to_o_grid),
                                           len(T_grid), len(P_grid))
        log_X_grid[q, :, :, :, :] = reshaped_array
    
    # Close the HDF5 file
    data.close()
    
    # Create a dictionary to store the loaded grid data
    fastchem_grid = {
        'log_X_grid': log_X_grid,
        'T_grid': T_grid,
        'P_grid': P_grid,
        'Metal_grid': Metal_grid,
        'c_to_o_grid': c_to_o_grid
    }
    
    return fastchem_grid


    
#fastchem_grid1 = read_fastchem_grid(chem_species=['CO'])


def intp_grid(P, T, c_to_o, metal, chem_species, fastchem_grid):
    
    
    
    log_X_grid = fastchem_grid['log_X_grid']
    T_grid =  fastchem_grid['T_grid']
    P_grid =  fastchem_grid['P_grid']
    Metal_grid = fastchem_grid['Metal_grid']
    c_to_o_grid =  fastchem_grid['c_to_o_grid']
    
    len_P, len_T, len_c_to_o, len_metal = np.array(P).size, np.array(T).size, \
                          np.array(c_to_o).size, np.array(metal).size


    max_len = max(len_P, len_T, len_c_to_o, metal)
    
    if len_P == 1:
        P = np.full(max_len, P)
    if len_T == 1:
        T = np.full(max_len, T)
    if len_c_to_o == 1:
        c_to_o = np.full(max_len, c_to_o)
    if len_metal == 1:
        metal = np.full(max_len, metal)   

    def interpolation(species):
        
        q = chem_species.index(species)
        
        intp = RegularGridInterpolator((Metal_grid, c_to_o_grid,
                                        T_grid, P_grid), 
                                        log_X_grid[q,:,:,:,:])
        
        x = intp(np.vstack((np.expand_dims(metal,0), 
                            np.expand_dims(c_to_o,0),
                            np.expand_dims(T,0), 
                            np.expand_dims(P,0))).T).T

        return x

    if isinstance(chem_species, str):
        
        return interpolation(chem_species)
    
    log_X = []
    
    for _,species in enumerate(chem_species):
        log_X.append(interpolation(species))
    
    log_X_intp_array = np.array(log_X)
    
    return log_X_intp_array.T
    
'''
# Test
chem_species=['CO','H2O']
fastchem_grid = read_fastchem_grid(chem_species)
c_to_o = 0.55
metal = 10

from tau_vert_update import read_sigmas, generate_pressure_grid

P = generate_pressure_grid(P_min=1.0e-7, P_max=100, N_layers=100)

from path_dist_copy import Temp_profile

T = Temp_profile(P, T_iso=1000)

log_X_intp_array = intp_grid(P, T, c_to_o, metal, chem_species, fastchem_grid)

import matplotlib.pyplot as plt
plt.yscale('log')
plt.plot(log_X_intp_array[:,0],P)
'''


    