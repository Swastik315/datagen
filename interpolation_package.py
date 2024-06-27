
'''
Various Interpolations
'''

from numba import njit, jit
import numpy as np
from scipy.interpolate import interp1d
from closest_index import prior_index
from scipy.interpolate import CubicSpline
import h5py



def clip(value, min_val, max_val):
    """
    Simulate the np.clip function using basic arithmetic operations for Numba compatibility.
    
    Parameters:
        value: Value to be clipped.
        min_val: Minimum value.
        max_val: Maximum value.
        
    Returns:
        Clipped value between min_val and max_val.
    """
    # Simulate the clip operation using min and max functions
    clipped_value = min(max(value, min_val), max_val)
    
    return clipped_value


def round_nearest(value):
    """
    Round a value to the nearest integer using basic arithmetic operations for Numba compatibility.
    
    Parameters:
        value: Value to be rounded.
        
    Returns:
        Nearest integer to the input value.
    """
    # Compute the integer part of the value
    int_part = int(value)
    
    # Compute the fractional part of the value
    frac_part = value - int_part
    
    # Round the fractional part to the nearest integer
    rounded_frac = int(frac_part + 0.5)
    
    # Combine the integer part and the rounded fractional part
    rounded_value = int_part + rounded_frac
    
    return rounded_value


def nearest_index(value, grid_start, grid_end, N_grid):
    
    '''
    Author: Tonmoy Deka
    
    Parameters:
        Value: Model grid value
        grid_start: original grid start value
        grid_end: original end value
        N_grid: number of values in original grid
    
    Returns:
        Index number of model value to the closest orignal value.
    '''
    
    # Single value grids only have one element, so return 0
    if N_grid == 1:
        return 0
    
    # Clip the value to the range of the grid
    value = clip(value, grid_start, grid_end)
    
    # Compute the index using linear interpolation
    i = (N_grid - 1) * ((value - grid_start) / (grid_end - grid_start))
    
    # Round to the nearest integer index
    closest_idx = round_nearest(i)
    
    return closest_idx



def extracted_log_sigmas(closest_index, log_sigma_q):
    
    '''
    Author: Tonmoy Deka
    Parameters: 
        closest_index: array of closest indices of the nu_model_grid values.
        log_sigma_q: original log_sigma cross-section values.
        
    Returns: 
        log_sigma_q array but only for the model fine wavenumber grid. On this we perform
        interpolation on pressure and temperature axis.
    '''
    
    sigma_subset = log_sigma_q[:, :, closest_index]
    
    return sigma_subset
 
    

def interp_P(wl_model, log_P_fine, nu_q, log_sigma_q):

    '''
    Author: Tonmoy Deka
    Parameters:
        log_P_fine = fine pressure array on which to interpolate log_sigma values
        wl_model: model wavelength array
        
    Returns:
        sigma values in linear scale interpolated in pressure grids.
    '''

    nu_model = 1.0e4/wl_model
    
    N_nu_q = len(nu_q)
    
    N_nu = len(nu_model)
    
    closest_index = []
    
    for k in range(N_nu):
        
        closest_index.append(nearest_index(nu_model[k], nu_q[0], nu_q[-1], N_nu_q))
          
    sigma_subset = log_sigma_q[:, :, closest_index] 
    
    interpolated_sigma = np.zeros((len(log_P_fine), sigma_subset.shape[1], sigma_subset.shape[2]))
    
    for i in range(sigma_subset.shape[1]):  
        
        for j in range(sigma_subset.shape[2]):
            
            f = CubicSpline(range(sigma_subset.shape[0]), sigma_subset[:, i, j])  # Cubic spline interpolation
            
            interpolated_sigma[:, i, j] = f(np.linspace(0, sigma_subset.shape[0] - 1, len(log_P_fine)))
            
    reversed_interpolated_sigma = interpolated_sigma[:, :, ::-1] # for output in increasing wavelength.

    interp_sigma_P = 10**(reversed_interpolated_sigma) # log to linear.
    
    return interp_sigma_P


def interp_T(interp_sigma_P, T_fine, T_grid):
    
    '''
    Returns:
        Interpolated sigmas along the fine temperature grid.
    '''
    N_P, N_T_old, N_nu = interp_sigma_P.shape
    N_T_fine = len(T_fine)

    sigma_interpolated = np.zeros((N_P, N_T_fine, N_nu))

    for i in range(N_P):
        for j in range(N_nu):
            sigma_interpolated[i, :, j] = np.interp(T_fine, T_grid, interp_sigma_P[i, :, j], left=0.0, right=0.0)

    sigma_interpolated = sigma_interpolated[:, :, ::-1]
    return sigma_interpolated


#------------------------AEROSOL-------------------------------------------------

def extract_aero(file_path, aerosol, size):
    """
    Extract coefficients for the specified size from the cross-section dataset in an HDF5 file.

    Parameters:
    - file_path (str): Path to the HDF5 files.
    - aerosol (str): Aerosol name.
    - size (float): Size for which extinction coefficients are to be extracted.

    Returns:
    - tuple: A tuple containing:
      - np.ndarray: 1D array of coefficients for the specified aerosol and size.
      - np.ndarray: Array of wavelengths.
    """
    file = f"{aerosol}.h5"
    with h5py.File(file_path + file, 'r') as f:
        cross_section_2d = f['sigma'][:]
        sizes = f['sizes'][:]
        current_wavelengths = f['wavelengths'][:]*1e6  # converting to microns
        wavelengths = current_wavelengths[:, 0]  # just some formatting

    size_indices = {size: idx for idx, size in enumerate(sizes)}

    if size in size_indices:
        index = size_indices[size]
        coefficients = cross_section_2d[:, index]
    else:
        raise ValueError(f"Specified size {size} not found in the sizes array for {aerosol}.")

    return coefficients, wavelengths


def extracted_aero(file_path, aero_sizes, wl_model, aero_species):
        
    sigma_subset = np.zeros((len(wl_model),len(aero_species)))
    
    #closest_index = []
    
    for q, species in enumerate(aero_species):
        
        species = aero_species[q]
        
        sizes = aero_sizes[q]
        
        coefficients, wavelengths = extract_aero(file_path, species, sizes)
        
        for k in range(len(wl_model)):
            
            closest_index = nearest_index(wl_model[k], wavelengths[0], wavelengths[-1], len(wavelengths))
            
            sigma_subset[k,q] = coefficients[closest_index]
        
    return sigma_subset


#-----------------------------------------------------------------------------------------------------------------------------------------
# The below codes of opacity sampling have been adapted from POSEIDON.....................................................................

def prior_index_V2(value, grid_start, grid_end, N_grid):
    ''' 
    Find the previous index of a *uniformly spaced* grid closest to a specified 
    value. When a uniform grid can be assumed, this function is much faster 
    than 'prior_index' due to there being no need for a loop. However, 
    for non-uniform grids one should still default to 'prior_index'.
    This function assumes the input grid monotonically increases.

    Args:
        value (float):
            The value for which the prior grid index is desired.
        grid_start (float):
            The value at the left edge of the uniform grid (array[0]).
        grid_start (float):
            The value at the right edge of the uniform grid (array[-1]).
        N_grid (int):
            The number of points on the uniform grid.

    Returns:
        (int):
            Prior index of the grid corresponding to the value.

    '''
    
    # Set to lower boundary
    if (value < grid_start):
        return 0
    
    # Set to upper boundary
    elif (value > grid_end):
        return N_grid-1
    
    # Use the equation of a straight line, then round down to integer.
    else:
        i = (N_grid-1) * ((value - grid_start) / (grid_end - grid_start))
        return int(i)



def closest_index(value, grid_start, grid_end, N_grid):
    '''
    Same as 'prior_index_V2', but for the closest index (i.e. can also round up).

    Args:
        val (float): 
            The value for which closest index is desired.
        grid_start (float):
            The value at the left edge of the uniform grid (array[0]).
        grid_start (float):
            The value at the right edge of the uniform grid (array[-1]).
        N_grid (int):
            The number of points on the uniform grid.

    Returns:
        (int):
            The index of the uniform grid closest to 'value'.

    '''

    # Single value grids only have one element, so return 0
    if (N_grid == 1):
        return 0

    # Set to lower boundary
    if (value < grid_start): 
        return 0
    
    # Set to upper boundary
    elif (value > grid_end):
        return N_grid-1
    
    # Use the equation of a straight line, then round to nearest integer.
    else:
        i = (N_grid-1) * ((value - grid_start) / (grid_end - grid_start))
        if ((i%1) <= 0.5):
            return int(i)     # Round down
            
        else:
            return int(i)+1   # Round up
 
           
def weight_factors(log_P_fine):
    
    '''
    Adapted from POSEIDON
    
    weight factors for nearest index interpolation
    
    log_P_fine = array of user-made fine model pressures.
    
    Output: weight factors x,b1,b2
    '''
     
    N_P_fine = len(log_P_fine)
    
    log_P_q = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2])
         
    N_P = len(log_P_q) 
   
    N_P = len(log_P_q)
    
    # Initialise array of indices on pre-calculated pressure opacity grid prior to defined atmosphere layer pressures
    x = np.zeros(N_P_fine, dtype=np.int64)
    
    b1 = np.zeros(shape=(N_P_fine))
    
    b2 = np.zeros(shape=(N_P_fine))
    
    w_p = np.zeros(N_P_fine)
    
    for i in range(N_P_fine):
        
        # Pressure below minimum then don't interpolate
        
        if (log_P_fine[i] < log_P_q[0]):
            
            x[i] = -1
            
            w_p[i] = 0.0
            
        # If pressure above maximum, do not interpolate
        elif (log_P_fine[i] >= log_P_q[-1]):
           
            x[i] = -2 
            
            w_p[i] = 0.0
            
        else:
            
            x[i] = prior_index_V2(log_P_fine[i], log_P_q[0], log_P_q[-1], N_P) # using prior_index_V2 and not closes_index
            
            w_p[i] = (log_P_fine[i]-log_P_q[x[i]])/(log_P_q[x[i]+1]-log_P_q[x[i]])     
            
           # Precalculate interpolation pre-factors to reduce computation overhead
        b1[i] = (1.0-w_p[i])
         
        b2[i] = w_p[i]  
        
    return x, b1, b2 
      


def intp_sigma(N_P_fine, N_T, N_P, N_wl, log_sigma_q, x, nu_model, b1, b2, nu_q, N_nu, T_q):
    
   '''
   Adapted from POSEIDON
   
   Interpolation of raw sigma to fine P and model Wl grid (from POSEIDON)
   
   Note: This interpolation method uses the nearest neighbour sampling method along wl and pressure.
   
   '''
    
   sigma_pre_inp = np.zeros(shape=(N_P_fine, N_T, N_wl))
    
   N_nu_q = len(nu_q)   # Number of wavenumber points in CIA array
   
   for k in range(N_nu): # Note that the k here is looping over wavenumber

     
        
     # Find closest indices in pre-computed wavenumber array to desired wavenumber grid
     z = closest_index(nu_model[k], nu_q[0], nu_q[-1], N_nu_q)

       
     for i in range(N_P_fine):
         
            for j in range(N_T):
            
                # If nu (wl) point out of range of opacity grid, set opacity to zero
                if ((z == 0) or (z == (N_nu_q-1))):
                    
                    sigma_pre_inp[i, j, ((N_wl-1)-k)] = 0.0
                
                else:
                                        
                        # If pressure below minimum, set to value at min pressure
                        if (x[i] == -1):
                            sigma_pre_inp[i, j, ((N_wl-1)-k)] = 10 ** (log_sigma_q[0, j, z])
                            
                        # If pressure above maximum, set to value at max pressure
                        elif (x[i] == -2):
                            sigma_pre_inp[i, j, ((N_wl-1)-k)] = 10 ** (log_sigma_q[(N_P-1), j, z])
            
                        # Interpolate sigma in logsace, then power to get interp array
                        else:
                            reduced_sigma = log_sigma_q[x[i]:x[i]+2, j, z]
                            
                            sigma_pre_inp[i, j, ((N_wl-1)-k)] =  10 ** (b1[i]*(reduced_sigma[0]) +
                                                                        b2[i]*(reduced_sigma[1]))
                                               
   return sigma_pre_inp
                    


    
def interpolate_temperature_sigma(sigma_pre_inp, T_grid, T_fine):
    
    '''
    T interpolation formula using numpy temperature interpolation
    
    sigma_pre_inp: interpolated sigmas in wl and P.
    T_grid: temperature array extracted from the chemical species opacity file.
    T_fine: fine temperature array in which to interpolate.
    
    Output: Fully interpolated sigmas in wl, P and T. ( this will be used as the opacity)
    '''
    N_P, N_T_old, N_nu = sigma_pre_inp.shape
    N_T_fine = len(T_fine)

    sigma_interp = np.zeros((N_P, N_T_fine, N_nu))

    for i in range(N_P):
        for j in range(N_nu):
            sigma_interp[i, :, j] = np.interp(T_fine, T_grid, sigma_pre_inp[i, :, j], left=0.0, right=0.0)

    return sigma_interp

#--------------------------

def T_interpolation_init(N_T_fine, T_grid, T_fine, y):
    ''' 
    Precomputes the T interpolation weight factors, so this does not
    need to be done multiple times across all species.
        
    '''
    
    w_T = np.zeros(N_T_fine)
        
    # Find T index in cross section arrays prior to fine temperature value
    for j in range(N_T_fine):
        
        if (T_fine[j] < T_grid[0]):   # If fine temperature point falls off LHS of temperature grid
            y[j] = -1                 # Special value (-1) stored, interpreted in interpellator
            w_T[j] = 0.0              # Weight not used in this case
            
        elif (T_fine[j] >= T_grid[-1]):   # If fine temperature point falls off RHS of temperature grid
            y[j] = -2                     # Special value (-2) stored, interpreted in interpellator
            w_T[j] = 0.0                  # Weight not used in this case
        
        else:
            
            # Have to use prior_index (V1) here as T_grid is not uniformly spaced
            y[j] = prior_index(T_fine[j], T_grid, 0)       # For cross section T interpolation
            
            # Pre-computed temperatures to left and right of fine temperature value
            T1 = T_grid[y[j]]
            T2 = T_grid[y[j]+1]
            
            # Precompute temperature weight factor
            w_T[j] = (1.0/((1.0/T2) - (1.0/T1)))
  
    return w_T


def T_interpolate_sigma(N_P_fine, N_T_fine, N_T, N_wl, sigma_pre_inp, T_grid, 
                        T_fine,y, w_T):
    ''' 
    Interpolates pre-processed cross section onto the fine T grid.
       
    Note: input sigma has format cross_sec[log(P)_pre, T_grid, wl_model], 
          whilst output has format cross_sec[log(P)_pre, T_fine, wl_model].
             
    Output is the interpolated cross section as a 3D array.
        
    '''
    
    
    sigma_inp = np.zeros(shape=(N_P_fine, N_T_fine, N_wl))

    for i in range(N_P_fine):       # Loop over pressures
        for j in range(N_T_fine):   # Loop over temperatures
            
            T = T_fine[j]           # Temperature we wish to interpolate to
            T1 = T_grid[y[j]]
            T2 = T_grid[y[j]+1]
            
            for k in range(N_wl):   # Loop over wavelengths
                
                # If T_fine below min value (100 K), set sigma to value at min T
                if (y[j] == -1):
                    sigma_inp[i, j, k] = sigma_pre_inp[i, 0, k]
                    
                # If T_fine above max value (3500 K), set sigma to value at max T
                elif (y[j] == -2):
                    sigma_inp[i, j, k] = sigma_pre_inp[i, (N_T-1), k]
            
                # Interpolate sigma to fine temperature grid value
                else: 
                    sig_reduced = sigma_pre_inp[i, y[j]:y[j]+2, k]
                    sig_1, sig_2 = sig_reduced[0], sig_reduced[1]    # sigma(T1)[i,k], sigma(T2)[i,k]
                    
                    sigma_inp[i, j, k] =  (sig_1**(w_T[j]*((1.0/T2) - (1.0/T)))) * sig_2** (w_T[j]*((1.0/T) - (1.0/T1)))
            
    return sigma_inp



#-------------------------------------------------------------------------------

